"""Deferred, sensor-scoped correction following the v7 method."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANGOS_FISICOS
from src.artifacts import file_sha256

from .data import SENSORS
from .settings import CLASSES, MASK_COLUMNS, SCHEMA

MAX_MISSING_ROWS = 20                 # 10 minutes at 30 s
MAX_ANCHOR_DISTANCE_ROWS = 60         # 30 minutes
DELTA_MAX = {
    "PCO2EXT": 80.0, "PHEXT": 8.0, "PRAD": 250.0,
    "PRGINT": 150.0, "PTEXT": 2.5, "PVV": 3.0,
    "XCO2I": 120.0, "XHINV": 8.0, "XTINV": 2.5,
}


def _segments(mask: np.ndarray, times: np.ndarray):
    positions = np.flatnonzero(mask)
    if not len(positions):
        return
    cuts = np.flatnonzero((np.diff(positions) != 1) |
                          (np.diff(times[positions]) != 30_000_000_000)) + 1
    for segment in np.split(positions, cuts):
        yield segment


def _sensor_attributed(series: pd.Series, sensor: str) -> np.ndarray:
    return series.fillna("").astype(str).map(
        lambda text: sensor in text.split(",")).to_numpy(bool)


def correct_frame(trace: pd.DataFrame, clean: np.ndarray | None = None,
                  synthetic: np.ndarray | None = None):
    """Return corrected data and a per-cell audit without consulting clean truth.

    clean and synthetic are only used later for metrics by summarize_correction.
    """
    del clean, synthetic
    required = {"Fecha", "pred_binaria", "tipo_predicho", "sensor_inferencia"}
    required.update(SENSORS)
    required.update(MASK_COLUMNS)
    if not required.issubset(trace.columns):
        raise ValueError(f"Missing correction inputs: {sorted(required-set(trace.columns))}")
    output = trace.copy()
    times = pd.to_datetime(trace["Fecha"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    labels = trace["tipo_predicho"].to_numpy()
    detected = trace["pred_binaria"].to_numpy(np.uint8) == 1
    logs = []
    for sensor, flag_col in zip(SENSORS, MASK_COLUMNS):
        original = pd.to_numeric(trace[sensor], errors="coerce").to_numpy(float)
        values = original.copy()
        marked = trace[flag_col].to_numpy(bool)
        attributed = _sensor_attributed(trace["sensor_inferencia"], sensor)
        targets = detected & attributed & np.isin(labels, CLASSES)
        # A mask carries the evidence for a zero or NaN missing reading.
        targets &= np.where(labels == "Datos Faltantes", marked,
                            np.isfinite(original))
        lo = RANGOS_FISICOS[sensor]["min"]
        hi = RANGOS_FISICOS[sensor]["max"]
        if not targets.any():
            output[sensor + "_corregido"] = values
            continue
        bad_anchor = targets | marked | ~np.isfinite(original) | (original < lo) | (original > hi)
        clean_anchor = ~bad_anchor
        for seg in _segments(targets, times):
            start, end = int(seg[0]), int(seg[-1])
            kind = str(labels[start])
            reason = ""
            if kind == "Datos Faltantes" and len(seg) > MAX_MISSING_ROWS:
                reason = "hueco_mayor_10_min"
            left_candidates = np.flatnonzero(clean_anchor[max(0, start-MAX_ANCHOR_DISTANCE_ROWS):start])
            left = (max(0, start-MAX_ANCHOR_DISTANCE_ROWS) +
                    int(left_candidates[-1])) if len(left_candidates) else None
            right_candidates = np.flatnonzero(
                clean_anchor[end+1:min(len(values), end+MAX_ANCHOR_DISTANCE_ROWS+1)])
            right = (end+1+int(right_candidates[0])) if len(right_candidates) else None
            if not reason and (left is None or right is None):
                reason = "sin_anclas_limpias"
            if not reason and (times[right] - times[left] !=
                               (right-left)*30_000_000_000):
                reason = "salto_temporal"
            if not reason and kind == "Datos Faltantes" and (
                    abs(original[right] - original[left]) > DELTA_MAX[sensor]):
                reason = "cambio_excesivo"
            if not reason and kind == "Valores Fuera de Rango" and not np.any(
                    (original[seg] < lo) | (original[seg] > hi)):
                reason = "sin_violacion_fisica"
            if not reason and kind == "Ruido":
                local = original[max(0, start-30):min(len(values), end+31)]
                local = local[np.isfinite(local)]
                if len(local) >= 5:
                    scale = max(float(np.std(local)), 1.0)
                    if not np.any(np.abs(original[seg] - np.median(local)) > 2*scale):
                        reason = "ruido_no_confirmado"
            if not reason:
                estimate = np.interp(times[seg], [times[left], times[right]],
                                     [original[left], original[right]])
                values[seg] = np.clip(estimate, lo, hi)
            for pos in seg:
                logs.append({
                    "Fecha": str(trace["Fecha"].iloc[pos]), "sensor": sensor,
                    "tipo_predicho": str(labels[pos]), "valor_original": original[pos],
                    "valor_corregido": values[pos],
                    "corregido": not bool(reason), "motivo": reason or "interpolacion",
                    "ancla_previa": "" if left is None else str(trace["Fecha"].iloc[left]),
                    "ancla_posterior": "" if right is None else str(trace["Fecha"].iloc[right]),
                })
        output[sensor + "_corregido"] = values
    log = pd.DataFrame(logs, columns=(
        "Fecha", "sensor", "tipo_predicho", "valor_original",
        "valor_corregido", "corregido", "motivo", "ancla_previa", "ancla_posterior"))
    return output, log


def summarize_correction(trace: pd.DataFrame, corrected: pd.DataFrame,
                         clean: np.ndarray, synthetic: np.ndarray,
                         injected: np.ndarray) -> pd.DataFrame:
    """Synthetic-only, changed-cell reconstruction metrics by true type/sensor."""
    raw = trace[list(SENSORS)].to_numpy(np.float64)
    result = corrected[[s + "_corregido" for s in SENSORS]].to_numpy(np.float64)
    clean = np.asarray(clean, np.float64)
    synthetic = np.asarray(synthetic, bool)
    injected = np.asarray(injected, np.float64)
    if injected.shape != raw.shape:
        raise ValueError("Injected truth does not align")
    if clean.shape != raw.shape or result.shape != raw.shape or synthetic.shape != (len(raw),):
        raise ValueError("Truth arrays do not align")
    changed = ~np.isclose(injected, clean, rtol=0, atol=0, equal_nan=True)
    truth = trace["tipo_real_auditoria"].to_numpy()
    records = []
    for kind in CLASSES:
        for j, sensor in enumerate(SENSORS):
            selected = synthetic & (truth == kind) & changed[:, j] & np.isfinite(clean[:, j])
            count = int(selected.sum())
            if not count:
                continue
            before = raw[selected, j]
            after = result[selected, j]
            target = clean[selected, j]
            recovered = np.isfinite(after)
            corrected_count = int((recovered & ~np.isclose(before, after, equal_nan=True)).sum())
            before_error = np.abs(before-target)
            after_error = np.abs(after-target)
            records.append({
                "tipo_real": kind, "sensor": sensor, "celdas": count,
                "corregidas": corrected_count, "cobertura": corrected_count/count,
                "mae_original": float(np.nanmean(before_error)) if np.isfinite(before_error).any() else None,
                "mae_corregido": float(np.nanmean(after_error)) if np.isfinite(after_error).any() else None,
                "rmse_corregido": float(np.sqrt(np.nanmean((after-target)**2)))
                if np.isfinite(after_error).any() else None,
            })
    # Any changed cell outside an attributed target is an implementation error.
    untouched = ~np.isin(np.arange(len(raw)), np.flatnonzero(
        trace["pred_binaria"].to_numpy(np.uint8) == 1))
    if not np.isclose(raw[untouched], result[untouched], equal_nan=True).all():
        raise ValueError("Correction modified a non-detected row")
    return pd.DataFrame(records)


def correct(results: Path) -> Path:
    results = Path(results)
    manifest = json.loads((results / "result_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Wrong evaluation schema")
    for name, digest in manifest["artifacts"].items():
        if file_sha256(results / name) != digest:
            raise ValueError(f"Evaluation artifact hash mismatch: {name}")
    trace = pd.read_csv(results / "test_trace.csv")
    with np.load(results / "truth_clean_test.npz") as data:
        clean = data["clean"].copy()
        synthetic = data["synthetic"].copy()
        injected = data["injected"].copy()
    corrected, log = correct_frame(trace)
    metrics = summarize_correction(trace, corrected, clean, synthetic, injected)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = results / "corrections" / stamp
    output.mkdir(parents=True)
    corrected.to_csv(output / "corrected_test.csv", index=False)
    log.to_csv(output / "correction_log.csv", index=False)
    metrics.to_csv(output / "reconstruction_metrics.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA,
        "evaluation_manifest_sha256": file_sha256(results / "result_manifest.json"),
        "artifacts": {name: file_sha256(output / name) for name in (
            "corrected_test.csv", "correction_log.csv", "reconstruction_metrics.csv")},
    }, indent=2), encoding="utf-8")
    return output
