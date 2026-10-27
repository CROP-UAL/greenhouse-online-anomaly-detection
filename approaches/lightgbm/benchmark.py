"""Normalize the frozen v8 online benchmark into the common edge schema."""
from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone

import pandas as pd
import lightgbm

from approaches.lightgbm.adapter import RUN, _upstream
from shared.benchmarking.report import save_report
from shared.contract import ROOT, validate_splits


def run_benchmark(*, rows: int = 2000, repeats: int = 5):
    if rows <= 0 or repeats <= 0:
        raise ValueError("Rows and repetitions must be positive")
    pipeline = _upstream()
    scratch = ROOT / "experiments" / "runs" / "lightgbm_benchmark" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    scratch.mkdir(parents=True, exist_ok=False)
    for name in ("binary_model.txt", "binary_calibration.json", "binary_manifest.json", "binary_validation.json",
                 "multiclass_model.txt", "multiclass_validation.json", "manifest.json"):
        shutil.copy2(RUN / name, scratch / name)
    paired_trace = ROOT / "results" / "traces" / "lightgbm_test.parquet"
    if paired_trace.exists():
        trace = pd.read_parquet(paired_trace)
        inverse = {"Normal": "Normal", "Contextual": "Contextual", "Noise": "Ruido",
                   "Correlation deviation": "Desviacion de Correlacion", "Missing data": "Datos Faltantes",
                   "Out-of-range": "Valores Fuera de Rango", "Stuck sensor": "Sensor Atascado"}
        path = scratch / "results" / "paired"
        path.mkdir(parents=True)
        pd.DataFrame({"tipo_predicho": trace.final_prediction.map(inverse)}).to_parquet(path / "test_trace.parquet", index=False)
    original = json.loads(pipeline.benchmark(scratch, rows=rows, repeats=repeats).read_text(encoding="utf-8"))
    report = {"hardware": platform.machine(), "target_hardware": "NVIDIA Jetson Orin Nano",
              "operating_system": platform.platform(), "python_version": platform.python_version(),
              "framework": "LightGBM", "framework_version": lightgbm.__version__,
              "model_version": "lightgbm-v8-lstm-common-20261021T120316012126Z",
              "data_manifest_id": validate_splits()["manifest_id"], "warmup_policy": "one full untimed pass",
              "repetitions": repeats, "prediction_unit": "target observation",
              "mean_latency_ms": original["mean_ms"], "median_latency_ms": original["p50_ms"],
              "p95_latency_ms": original["p95_ms"], "throughput_per_second": original["throughput_rows_per_second"],
              "memory_usage_bytes": None, "timing_scope": original["timing_scope"],
              "parity_with_test_trace": original["parity_with_test_trace"], "rows_per_pass": original["rows_per_pass"]}
    output = ROOT / "results" / "reports" / f"lightgbm_benchmark_{scratch.name}.json"
    save_report(report, output)
    return output
