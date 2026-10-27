"""Sequential warm benchmark of precomputed causal LSTM windows."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from approaches.lstm.causal_data import prepare_features, windows
from shared.benchmarking.report import build_report, save_report
from shared.contract import ROOT, sha256


def run_benchmark(run: Path, *, rows: int = 2000, repeats: int = 5) -> Path:
    import tensorflow as tf
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if sha256(run / "model1.keras") != manifest["model1_sha256"] or sha256(run / "model2.keras") != manifest["model2_sha256"]:
        raise ValueError("LSTM models differ from run manifest")
    model1 = tf.keras.models.load_model(run / "model1.keras", compile=False)
    model2 = tf.keras.models.load_model(run / "model2.keras", compile=False)
    m1, m2, _, metadata = prepare_features()
    if metadata["data_manifest_id"] != manifest["data_manifest_id"]:
        raise ValueError("Data manifest changed")
    n = min(rows, len(m1["test"]))
    if n <= 0 or repeats <= 0:
        raise ValueError("Rows and repetitions must be positive")
    x1, targets = windows(m1, "test", targets=np.arange(n))
    x2, _ = windows(m2, "test", targets=targets)
    threshold = manifest["binary_threshold_validation_f1"]
    for i in range(n):
        value = float(model1(x1[i:i + 1], training=False).numpy()[0, 0])
        if value >= threshold:
            model2(x2[i:i + 1], training=False)
    durations = []
    start_passes = time.perf_counter()
    for _ in range(repeats):
        for i in range(n):
            start = time.perf_counter_ns()
            value = float(model1(x1[i:i + 1], training=False).numpy()[0, 0])
            if value >= threshold:
                model2(x2[i:i + 1], training=False)
            durations.append((time.perf_counter_ns() - start) / 1e6)
    elapsed = time.perf_counter() - start_passes
    report = build_report(hardware=platform.machine(), framework="TensorFlow/Keras", framework_version=tf.__version__,
                          model_version=f"lstm-causal-v1-{run.name}", data_manifest_id=metadata["data_manifest_id"],
                          warmup_policy="one full untimed pass", repetitions=repeats,
                          prediction_unit="one 20-observation sequence ending at one target observation",
                          durations_ms=np.asarray(durations), elapsed_seconds=elapsed, memory_usage_bytes=None,
                          timing_scope="Model 1 + conditional Model 2 on precomputed scaled windows; excludes feature extraction, scaling and I/O")
    output = ROOT / "results" / "reports" / f"lstm_benchmark_{run.name}.json"
    save_report(report, output)
    return output
