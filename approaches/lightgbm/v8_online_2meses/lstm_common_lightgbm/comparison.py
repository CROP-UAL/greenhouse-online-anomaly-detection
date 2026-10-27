"""Read-only alignment and paired metrics against an exported LSTM CSV."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from seis_anomalias.settings import CLASSES
from .pipeline import _json, file_sha256

def compare_with_lstm(result: str | Path, lstm_csv: str | Path) -> Path:
    """Require LSTM's 20-step targets to match LightGBM FECHA exactly."""
    result, lstm_csv = Path(result), Path(lstm_csv)
    trace = pd.read_parquet(result / "test_trace.parquet")
    lstm = pd.read_csv(lstm_csv)
    needed = {"FECHA", "y_true_binary", "model1_predicted_anomaly", "model2_predicted_type"}
    if missing := needed - set(lstm.columns):
        raise ValueError(f"LSTM prediction CSV lacks {sorted(missing)}")
    dates = pd.to_datetime(lstm.FECHA)
    if dates.dt.tz is not None or dates.isna().any() or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Invalid LSTM FECHA")
    # workflow.ipynb exports one target per test sequence, starting at row 20.
    expected = trace.iloc[19:].reset_index(drop=True)
    if len(lstm) != len(expected) or not dates.reset_index(drop=True).equals(expected.FECHA):
        raise ValueError("LSTM CSV does not match the 20-step test targets by FECHA")
    binary_lstm = lstm.model1_predicted_anomaly.to_numpy()
    binary_truth = lstm.y_true_binary.to_numpy()
    if not set(np.unique(binary_lstm)).issubset({0, 1}) or not set(np.unique(binary_truth)).issubset({0, 1}):
        raise ValueError("Invalid LSTM binary labels")
    if not np.array_equal(binary_truth, expected.target_anomalia.to_numpy()):
        raise ValueError("LSTM and physical parquet truth disagree")
    types = lstm.model2_predicted_type.copy()
    if types[binary_lstm == 1].isna().any() or types[binary_lstm == 0].notna().any():
        raise ValueError("LSTM routing and multiclass output disagree")
    unknown = set(types.dropna()) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown LSTM predicted types: {sorted(unknown)}")
    final_lstm = types.fillna("Normal").to_numpy()
    labels = ("Normal",) + CLASSES
    y = expected.tipo_real.to_numpy()
    lightgbm_binary = expected.pred_binaria.to_numpy()
    lightgbm_final = expected.tipo_predicho.to_numpy()
    metrics = {
        "rows": len(expected), "first_FECHA": str(expected.FECHA.iloc[0]),
        "last_FECHA": str(expected.FECHA.iloc[-1]),
        "lstm_csv_sha256": file_sha256(lstm_csv),
        "lightgbm_trace_sha256": file_sha256(result / "test_trace.parquet"),
        "lightgbm": {
            "binary_f1": float(f1_score(binary_truth, lightgbm_binary)),
            "macro_f1_seven_outputs": float(f1_score(y, lightgbm_final, labels=labels, average="macro", zero_division=0)),
            "report": classification_report(y, lightgbm_final, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, lightgbm_final, labels=labels).tolist(),
        },
        "lstm": {
            "binary_f1": float(f1_score(binary_truth, binary_lstm)),
            "macro_f1_seven_outputs": float(f1_score(y, final_lstm, labels=labels, average="macro", zero_division=0)),
            "report": classification_report(y, final_lstm, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, final_lstm, labels=labels).tolist(),
        },
        "labels": list(labels),
    }
    target = result / "comparison_lstm_paired.json"
    if target.exists():
        raise FileExistsError(target)
    _json(target, metrics)
    paired = expected[["FECHA", "tipo_real", "target_anomalia", "pred_binaria", "tipo_predicho"]].copy()
    paired["lstm_pred_binaria"] = binary_lstm
    paired["lstm_tipo_predicho"] = final_lstm
    paired.to_parquet(result / "comparison_lstm_paired.parquet", index=False)
    return target

