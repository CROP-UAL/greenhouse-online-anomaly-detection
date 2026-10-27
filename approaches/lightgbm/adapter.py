"""Adapter around the frozen v8 physical-data pipeline; no model changes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shared.contract import ROOT, FILES, ANOMALIES, SOURCE_TO_CANONICAL, validate_splits
from shared.trace import make_trace

UPSTREAM = Path(__file__).resolve().parent / "v8_online_2meses"
RUN = UPSTREAM / "reference_run"


def _upstream():
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from lstm_common_lightgbm import pipeline
    pipeline.PARQUETS = {"train": FILES["train"], "val": FILES["validation"], "test": FILES["test"]}
    return pipeline


def export_test() -> tuple[Path, Path]:
    manifest = validate_splits()["manifest_id"]
    pipeline = _upstream()
    binary_model, calibration, multiclass = pipeline.load_run(RUN)
    frames, table, matrices = pipeline._data(include_test=True)
    physical = pd.read_parquet(FILES["test"])
    test = table["test"]
    if not physical.FECHA.reset_index(drop=True).equals(test.FECHA.reset_index(drop=True)):
        raise ValueError("LightGBM target timestamps differ from frozen test")
    scores, detected, rules = pipeline._decide(binary_model, calibration, matrices["test"], test,
                                                frames["val"].iloc[-pipeline.HISTORY_ROWS:])
    selected = np.flatnonzero(detected)
    stage2 = np.full(len(test), None, dtype=object)
    if len(selected):
        predictions = multiclass.predict(matrices["test"][selected], num_threads=1)
        stage2[selected] = [SOURCE_TO_CANONICAL[pipeline.CLASSES[i]] for i in np.argmax(predictions, axis=1)]
    version = "lightgbm-v8-lstm-common-20261021T120316012126Z"
    trace = make_trace(physical, split="test", scores=scores, binary=detected, stage2=stage2,
                       model_version=version, data_manifest_id=manifest,
                       extra={"binary_rule_code": rules})
    anomalies = np.flatnonzero(physical.tipo_anomalia.notna().to_numpy())
    isolated_scores = multiclass.predict(matrices["test"][anomalies], num_threads=1)
    diagnostic = pd.DataFrame({"timestamp": physical.FECHA.iloc[anomalies].to_numpy(),
                               "split": "test", "ground_truth_type": trace.ground_truth_type.iloc[anomalies].to_numpy(),
                               "stage_2_prediction": [SOURCE_TO_CANONICAL[pipeline.CLASSES[i]] for i in np.argmax(isolated_scores, axis=1)],
                               "model_version": version, "data_manifest_id": manifest})
    out = ROOT / "results" / "traces"
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "lightgbm_test.parquet"
    diagnostic_path = out / "lightgbm_model2_isolated_test.parquet"
    trace.to_parquet(trace_path, index=False)
    diagnostic.to_parquet(diagnostic_path, index=False)
    return trace_path, diagnostic_path
