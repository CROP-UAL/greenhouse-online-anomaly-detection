"""Evaluate distinct binary, isolated, conditional and cascade populations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from shared.contract import ANOMALIES, CLASSES, ROOT
from shared.trace import validate_trace


def _classification(truth, predicted, labels: tuple[str, ...]) -> dict:
    return {"rows": len(truth), "labels": list(labels),
            "per_class": classification_report(truth, predicted, labels=list(labels), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=list(labels)).tolist()}


def evaluate_one(trace_path: Path, diagnostic_path: Path) -> dict:
    trace = pd.read_parquet(trace_path)
    isolated = pd.read_parquet(diagnostic_path)
    validate_trace(trace)
    if trace.split.nunique() != 1 or trace.split.iloc[0] != "test":
        raise ValueError("Evaluation requires one complete test trace")
    if trace.data_manifest_id.nunique() != 1 or isolated.data_manifest_id.nunique() != 1 or trace.data_manifest_id.iloc[0] != isolated.data_manifest_id.iloc[0]:
        raise ValueError("Trace and isolated predictions use different physical data")
    if trace.model_version.nunique() != 1 or isolated.model_version.nunique() != 1 or trace.model_version.iloc[0] != isolated.model_version.iloc[0]:
        raise ValueError("Trace and isolated predictions use different models")
    anomaly = trace.ground_truth_type.ne("Normal")
    expected = trace.loc[anomaly, ["timestamp", "ground_truth_type"]].reset_index(drop=True)
    actual = isolated[["timestamp", "ground_truth_type"]].reset_index(drop=True)
    if not expected.equals(actual) or not isolated.stage_2_prediction.isin(ANOMALIES).all():
        raise ValueError("Model 2 isolated trace must cover every true anomalous target exactly once")
    routed = trace.stage_2_executed
    conditional = anomaly & routed
    binary = _classification(trace.ground_truth_binary, trace.binary_prediction, ("Normal", "Anomalous"))
    stage2_isolated = _classification(isolated.ground_truth_type, isolated.stage_2_prediction, ANOMALIES)
    stage2_conditional = _classification(trace.loc[conditional, "ground_truth_type"],
                                         trace.loc[conditional, "stage_2_prediction"], ANOMALIES)
    cascade = _classification(trace.ground_truth_type, trace.final_prediction, CLASSES)
    report = {"model_version": trace.model_version.iloc[0], "data_manifest_id": trace.data_manifest_id.iloc[0],
              "binary_full_test": binary, "model2_isolated_true_anomalies": stage2_isolated,
              "model2_conditional_detected_true_anomalies": stage2_conditional, "cascade_seven_class_full_test": cascade,
              "false_alarms_model1": int((~anomaly & routed).sum()),
              "missed_anomalies_model1": int((anomaly & ~routed).sum()),
              "stage2_invocations": int(routed.sum())}
    if trace.loc[anomaly, "episode_id"].notna().all():
        grouped = trace.loc[anomaly].groupby("episode_id", sort=False)
        report["episodes"] = {"count": int(grouped.ngroups),
                              "detected_at_least_once": int(grouped.stage_2_executed.any().sum()),
                              "all_rows_correct_final": int(grouped.apply(lambda g: g.final_prediction.eq(g.ground_truth_type).all(), include_groups=False).sum()),
                              "definition": "contiguous rows with the same source anomalia_id inside one split"}
    return report


def evaluate_pair(lightgbm_trace: Path, lightgbm_isolated: Path, lstm_trace: Path, lstm_isolated: Path,
                  output: Path | None = None) -> dict:
    first = pd.read_parquet(lightgbm_trace)
    second = pd.read_parquet(lstm_trace)
    keys = ["timestamp", "split", "ground_truth_binary", "ground_truth_type", "episode_id", "data_manifest_id"]
    if not first[keys].reset_index(drop=True).equals(second[keys].reset_index(drop=True)):
        raise ValueError("Paired traces have different physical targets or labels")
    result = {"lightgbm": evaluate_one(lightgbm_trace, lightgbm_isolated),
              "lstm": evaluate_one(lstm_trace, lstm_isolated), "paired_test_rows": len(first)}
    if output is None:
        output = ROOT / "results" / "metrics" / "paired_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
