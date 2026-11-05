"""Causal adaptation of the Agroconnect feature flow for the shared experiment.

The imported source notebooks remain intact. Their Model 2 centred windows,
negative shifts and two-sided interpolation require future observations, so
this separately versioned run uses past-only replacements with the same model
architectures. See docs/comparison_protocol.md before interpreting metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from shared.contract import ROOT, SPLITS, ROWS, load_split, validate_splits

SOURCE_ROOT = Path(__file__).resolve().parent

from approaches.lstm.src.feature_engineering import engineer_features  # noqa: E402
from approaches.lstm.src.scaling import MODEL_FEATURES  # noqa: E402
from approaches.lstm.src.config_injection import SENSORES_STUCK  # noqa: E402
from approaches.lstm.src.diagnostic_features_model2 import (  # noqa: E402
    TARGET_SENSORS, add_physical_diagnostic_features, add_relationship_diagnostic_features,
)

ALL_SENSORS = tuple(MODEL_FEATURES[:12])
STUCK_SENSORS = tuple(SENSORES_STUCK)
M2_REDUNDANT = {"PRGINT_minus_PRAD", "ventilation_min", "ventilation_max"}
for _sensor in TARGET_SENSORS:
    M2_REDUNDANT.update({f"{_sensor}_distance_to_min", f"{_sensor}_distance_to_max",
                         f"{_sensor}_rolling_range"})
M2_EXCLUDE = {"FECHA", "sensor_anomalia", "anomalia_id", "tipo_anomalia", "fuente_anomalia",
              "valor_original_anomalia", "valor_inyectado_anomalia", "_original_observation",
              "gap_category", "in_short_gap", "in_medium_gap", "in_long_gap", "sequence_unsafe",
              "has_unresolved_nan", "tipo_canonico", "_split", "_row"}


def _causal_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for sensor in TARGET_SENSORS:
        series = output[sensor]
        delta = series.diff()
        prior_delta = delta.shift(1)
        output[f"{sensor}_delta_prev"] = delta
        output[f"{sensor}_delta_prev_lag1"] = prior_delta
        output[f"{sensor}_abs_delta_prev"] = delta.abs()
        output[f"{sensor}_abs_delta_prev_lag1"] = prior_delta.abs()
        rolling = series.rolling(window=5, min_periods=2)
        output[f"{sensor}_rolling_std"] = rolling.std()
        output[f"{sensor}_rolling_range"] = rolling.max() - rolling.min()
        output[f"{sensor}_change_balance"] = (delta.abs() - prior_delta.abs()).abs()
        output[f"{sensor}_spike_score"] = (delta.abs() + prior_delta.abs()) / rolling.std().replace(0, np.nan)
        output[f"{sensor}_local_change_mean"] = (delta.abs() + prior_delta.abs()) / 2
    output = add_physical_diagnostic_features(output, sensors=TARGET_SENSORS)
    return add_relationship_diagnostic_features(output)


def _stuck_run_lengths(frame: pd.DataFrame, *, cap: int = 20) -> pd.DataFrame:
    """Consecutive identical readings ending at each row, using past data only."""
    result = pd.DataFrame(index=frame.index)
    for sensor in STUCK_SENSORS:
        values = frame[sensor]
        same = values.eq(values.shift()) & values.notna() & values.shift().notna()
        groups = (~same).cumsum()
        result[f"{sensor}_stuck_run_length"] = (
            same.groupby(groups).cumcount().add(1).clip(upper=cap).astype(np.float32)
        )
    return result


def prepare_features(*, stuck_context: bool = False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, pd.DataFrame], dict]:
    contract = validate_splits()
    physical = {name: load_split(name) for name in SPLITS}
    joined = pd.concat([physical[name].assign(_split=name, _row=np.arange(len(physical[name])))
                        for name in SPLITS], ignore_index=True)
    if joined.FECHA.diff().dt.total_seconds().iloc[1:].gt(60).any() or joined.sequence_unsafe.any():
        raise ValueError("The frozen timeline contains an unsafe LSTM window")
    missing = joined.loc[:, ALL_SENSORS].isna()
    joined["any_sensor_was_missing"] = missing.any(axis=1).astype(np.float32)
    for sensor in ALL_SENSORS:
        joined[f"{sensor}_was_missing"] = missing[sensor].astype(np.float32)
    # Past-only imputation. Training medians are used only where no prior value exists.
    medians = joined.loc[joined._split.eq("train"), ALL_SENSORS].median()
    joined.loc[:, ALL_SENSORS] = joined.loc[:, ALL_SENSORS].ffill().fillna(medians)
    if joined.loc[:, ALL_SENSORS].isna().any().any():
        raise ValueError("Causal imputation left missing sensor values")
    engineered = engineer_features(joined)
    if stuck_context:
        run_lengths = _stuck_run_lengths(engineered)
    m1_values = engineered.loc[:, MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    m1_fill = m1_values.loc[engineered._split.eq("train")].median()
    m1_values = m1_values.fillna(m1_fill)
    if m1_values.isna().any().any():
        raise ValueError("Model 1 has undefined features")
    train_mask = engineered._split.eq("train").to_numpy()
    scaler1 = StandardScaler().fit(m1_values.loc[train_mask])
    matrix1 = np.column_stack([scaler1.transform(m1_values).astype(np.float32),
                               engineered.any_sensor_was_missing.to_numpy(np.float32)])
    if stuck_context:
        matrix1 = np.column_stack([matrix1, run_lengths.to_numpy(np.float32) / 20.0])
    if matrix1.shape[1] != (27 if stuck_context else 19):
        raise ValueError("Unexpected Model 1 input width")
    diagnostics = _causal_diagnostics(engineered)
    if stuck_context:
        diagnostics = pd.concat([diagnostics, run_lengths], axis=1)
    selected = [column for column in diagnostics.columns
                if column not in M2_EXCLUDE and column not in M2_REDUNDANT
                and not column.endswith("_out_of_range") and not column.endswith("_interpolated")
                and pd.api.types.is_numeric_dtype(diagnostics[column])]
    matrix2_df = diagnostics.loc[:, selected].replace([np.inf, -np.inf], np.nan).copy()
    ratios = [column for column in ("PRGINT_PRAD_ratio", "PRGINT_PRAD_relative_difference", "co2_per_ventilation")
              if column in matrix2_df]
    for column in ratios:
        matrix2_df[f"{column}_was_undefined"] = matrix2_df[column].isna().astype(np.float32)
    m2_fill = matrix2_df.loc[train_mask].median().fillna(0)
    matrix2_df = matrix2_df.fillna(m2_fill)
    if matrix2_df.isna().any().any():
        raise ValueError("Model 2 has undefined features")
    scaler2 = StandardScaler().fit(matrix2_df.loc[train_mask])
    matrix2 = scaler2.transform(matrix2_df).astype(np.float32)
    offsets = np.cumsum([0] + [ROWS[name] for name in SPLITS])
    m1 = {name: matrix1[offsets[i]:offsets[i + 1]] for i, name in enumerate(SPLITS)}
    m2 = {name: matrix2[offsets[i]:offsets[i + 1]] for i, name in enumerate(SPLITS)}
    metadata = {"model1_features": list(MODEL_FEATURES) + ["any_sensor_was_missing"] + (list(run_lengths.columns) if stuck_context else []),
                "model2_features": list(matrix2_df.columns), "model2_feature_count": matrix2.shape[1],
                "data_manifest_id": contract["manifest_id"], "imputation": "past-only forward fill, then training median",
                "model2_temporal_features": "past-only replacements for negative shifts and centred windows"}
    if stuck_context:
        metadata["stuck_context"] = "consecutive identical readings by source eligible sensor, capped at 20; Model 1 divided by 20"
    return m1, m2, physical, metadata


def windows(matrices: dict[str, np.ndarray], split: str, *, length: int = 20,
            targets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if split not in SPLITS:
        raise ValueError(split)
    order = list(SPLITS)
    index = order.index(split)
    history = (np.concatenate([matrices[name][-(length - 1):] for name in order[:index]], axis=0)[-(length - 1):]
               if index else np.empty((0, matrices[split].shape[1]), np.float32))
    first = length - 1 - len(history)
    target_rows = np.arange(max(0, first), len(matrices[split]), dtype=np.int64) if targets is None else np.asarray(targets, np.int64)
    if (target_rows < max(0, first)).any() or (target_rows >= len(matrices[split])).any():
        raise ValueError("Target lacks causal history or lies outside its split")
    indices = target_rows[:, None] - np.arange(length - 1, -1, -1)[None, :]
    result = matrices[split][np.clip(indices, 0, len(matrices[split]) - 1)]
    from_history = indices < 0
    if from_history.any():
        result[from_history] = history[indices[from_history]]
    return np.ascontiguousarray(result, dtype=np.float32), target_rows
