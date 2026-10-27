"""Common trace construction and strict routing validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.contract import ANOMALIES, CLASSES, canonical_label

TRACE_COLUMNS = ("timestamp", "split", "ground_truth_binary", "ground_truth_type", "binary_score",
                 "binary_prediction", "stage_2_executed", "stage_2_prediction", "final_prediction",
                 "episode_id", "model_version", "data_manifest_id")


def episode_ids(physical: pd.DataFrame, split: str) -> pd.Series:
    """A source ID broken by a normal row or ID change starts a new episode."""
    if "anomalia_id" not in physical:
        return pd.Series([None] * len(physical), dtype=object)
    source = physical.anomalia_id.reset_index(drop=True)
    label = physical.tipo_anomalia.reset_index(drop=True)
    if source.notna().ne(label.notna()).any():
        raise ValueError("Source anomaly IDs and labels do not cover the same rows")
    starts = source.ne(source.shift()) | source.isna()
    run = starts.cumsum()
    return pd.Series([f"{split}:{value}:{int(segment)}" if pd.notna(value) else None
                      for value, segment in zip(source, run)], dtype=object)


def make_trace(physical: pd.DataFrame, *, split: str, scores: np.ndarray, binary: np.ndarray,
               stage2: np.ndarray, model_version: str, data_manifest_id: str,
               extra: dict[str, object] | None = None) -> pd.DataFrame:
    n = len(physical)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    binary = np.asarray(binary).astype(bool).reshape(-1)
    stage2 = np.asarray(stage2, dtype=object).reshape(-1)
    if len(scores) != n or len(binary) != n or len(stage2) != n:
        raise ValueError("Trace arrays must match physical rows")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("binary_score must be a finite probability")
    if any(label not in ANOMALIES for label in stage2[binary]):
        raise ValueError("Positive Model 1 rows need a six-class Model 2 prediction")
    if any(pd.notna(label) for label in stage2[~binary]):
        raise ValueError("Model 2 cannot be executed for Model 1 normal rows")
    truth = physical.tipo_anomalia.map(canonical_label).to_numpy()
    trace = pd.DataFrame({
        "timestamp": physical.FECHA.to_numpy(), "split": split,
        "ground_truth_binary": np.where(truth == "Normal", "Normal", "Anomalous"),
        "ground_truth_type": truth, "binary_score": scores,
        "binary_prediction": np.where(binary, "Anomalous", "Normal"),
        "stage_2_executed": binary, "stage_2_prediction": stage2,
        "final_prediction": np.where(binary, stage2, "Normal"),
        "episode_id": episode_ids(physical, split),
        "model_version": model_version, "data_manifest_id": data_manifest_id,
    })
    if extra:
        for name, value in extra.items():
            if name in TRACE_COLUMNS:
                raise ValueError(f"Cannot overwrite trace contract column: {name}")
            trace[name] = value
    validate_trace(trace)
    return trace


def validate_trace(trace: pd.DataFrame) -> None:
    missing = set(TRACE_COLUMNS) - set(trace.columns)
    if missing:
        raise ValueError(f"Missing trace fields: {sorted(missing)}")
    if not trace.timestamp.is_unique or trace.timestamp.isna().any() or not trace.timestamp.is_monotonic_increasing:
        raise ValueError("Trace target timestamps must be ordered and unique")
    if not trace.ground_truth_type.isin(CLASSES).all() or not trace.final_prediction.isin(CLASSES).all():
        raise ValueError("Unknown seven-class label")
    if not trace.ground_truth_binary.eq(np.where(trace.ground_truth_type.eq("Normal"), "Normal", "Anomalous")).all():
        raise ValueError("Binary truth conflicts with type truth")
    routed = trace.binary_prediction.eq("Anomalous")
    if not trace.stage_2_executed.eq(routed).all():
        raise ValueError("Stage 2 routing conflicts with Model 1")
    if not trace.loc[~routed, "stage_2_prediction"].isna().all():
        raise ValueError("Stage 2 predicted on a normal decision")
    if not trace.loc[routed, "stage_2_prediction"].isin(ANOMALIES).all():
        raise ValueError("Stage 2 has an invalid anomaly prediction")
    if not trace.final_prediction.eq(np.where(routed, trace.stage_2_prediction, "Normal")).all():
        raise ValueError("Final prediction does not follow the cascade")
