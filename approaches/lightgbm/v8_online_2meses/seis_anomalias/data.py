"""Data preparation and the same causal feature path used at inference."""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from config import (COLUMNAS_SENSORES, INYECCION_CONFIG, SEED_INYECCION_TRAIN,
                    SEED_INYECCION_VAL, SEED_INYECCION_TEST)
from src.features import cargar_y_regularizar_datos, split_temporal_81dias
from src.injection import inyectar_anomalias_sinteticas
from src.multiclass_features import EXTRA_FEATURES, SharedFeatureExtractor

from .settings import BASE_FEATURES, DATASET, MASK_COLUMNS, canonical

SENSORS = tuple(COLUMNAS_SENSORES)
FEATURES = BASE_FEATURES + EXTRA_FEATURES + MASK_COLUMNS
SEEDS = {"train": SEED_INYECCION_TRAIN, "val": SEED_INYECCION_VAL,
         "test": SEED_INYECCION_TEST}


def absence_flags(raw: np.ndarray) -> np.ndarray:
    """Per-sample evidence: NaN or a zero within a mixed partial-loss pattern.

    No future sample or synthetic truth is consulted. A stand-alone physical
    zero, including night radiation, remains a valid zero.
    """
    values = np.asarray(raw, np.float32)
    if values.shape[-1] != len(SENSORS):
        raise ValueError("Expected nine sensor values")
    missing = np.isnan(values)
    zero = np.isfinite(values) & (values == 0)
    mixed = (missing.sum(axis=-1) >= 3) & (zero.sum(axis=-1) >= 4)
    return (missing | (zero & np.expand_dims(mixed, -1))).astype(np.uint8)


def decorate_masks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    flags = absence_flags(out[list(SENSORS)].to_numpy(np.float32))
    for j, column in enumerate(MASK_COLUMNS):
        out[column] = flags[:, j]
    return out


def prepare_splits() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    raw = cargar_y_regularizar_datos(str(DATASET))
    raw["Fecha"] = pd.to_datetime(raw["Fecha"])
    warmup, train, val, test = split_temporal_81dias(raw)
    originals = {"warmup": decorate_masks(warmup)}
    injected = {}
    for split, frame in (("train", train), ("val", val), ("test", test)):
        clean = decorate_masks(frame)
        # The inherited injector prints its eight origin labels. Show the
        # operational six-class contract after regrouping instead.
        with contextlib.redirect_stdout(io.StringIO()):
            product = inyectar_anomalias_sinteticas(
                frame, sensor_cols=list(SENSORS), config=INYECCION_CONFIG,
                seed=SEEDS[split])
        source = product["tipo_anomalia"].astype(str).copy()
        product["tipo_origen"] = source
        product["tipo_anomalia"] = source.map(canonical)
        product = decorate_masks(product)
        original_raw = clean[list(SENSORS)].to_numpy(np.float32)
        changed = ~(np.isclose(product[list(SENSORS)].to_numpy(np.float32),
                               original_raw, rtol=0, atol=0, equal_nan=True))
        product["origen_dato"] = np.where(
            product["target_anomalia"].to_numpy() == 0, "normal",
            np.where(changed.any(axis=1), "sintetico", "real"))
        print(f"{split.upper()} — seis clases:", product["tipo_anomalia"].value_counts().to_dict())
        originals[split], injected[split] = clean, product
    return originals, injected


def arrays(frame: pd.DataFrame):
    raw = np.ascontiguousarray(frame[list(SENSORS)].to_numpy(np.float32))
    times = pd.to_datetime(frame["Fecha"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    mask = np.ascontiguousarray(frame[list(MASK_COLUMNS)].to_numpy(np.uint8))
    return raw, times, mask


def stream_matrix(frame: pd.DataFrame, history: pd.DataFrame,
                  fills: np.ndarray) -> np.ndarray:
    raw, times, masks = arrays(frame)
    old, old_times, old_masks = arrays(history)
    if not len(raw) or not len(old):
        raise ValueError("Streaming features require nonempty split and history")
    extractor = SharedFeatureExtractor(
        BASE_FEATURES, fills, old[-360:], old_times[-360:],
        old_masks[-360:], times[0])
    matrix = np.empty((len(raw), len(FEATURES)), np.float32)
    for i in range(len(raw)):
        matrix[i, :len(BASE_FEATURES) + len(EXTRA_FEATURES)] = extractor.step(
            raw[i], times[i], masks[i])
        matrix[i, -len(SENSORS):] = masks[i]
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite feature matrix")
    return matrix


def build_matrices(originals: dict, frames: dict):
    # Zero is the fixed fallback for invalid causal features in this variant.
    fills = np.zeros(len(BASE_FEATURES), np.float32)
    matrices = {"train": stream_matrix(frames["train"], originals["warmup"], fills)}
    matrices["val"] = stream_matrix(frames["val"], frames["train"].iloc[-360:], fills)
    matrices["test"] = stream_matrix(frames["test"], frames["val"].iloc[-360:], fills)
    return matrices, fills


def usable(frame: pd.DataFrame) -> np.ndarray:
    if "usar_en_train" not in frame:
        return np.ones(len(frame), bool)
    return frame["usar_en_train"].fillna(False).astype(bool).to_numpy()
