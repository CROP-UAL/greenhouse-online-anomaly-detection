"""Common report shape for edge measurements."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

REQUIRED = ("hardware", "operating_system", "python_version", "framework", "framework_version",
            "model_version", "data_manifest_id", "warmup_policy", "repetitions", "prediction_unit",
            "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "throughput_per_second",
            "memory_usage_bytes", "timing_scope")


def build_report(*, hardware: str, framework: str, framework_version: str, model_version: str,
                 data_manifest_id: str, warmup_policy: str, repetitions: int, prediction_unit: str,
                 durations_ms: np.ndarray, elapsed_seconds: float, memory_usage_bytes: int | None,
                 timing_scope: str, target_hardware: str = "NVIDIA Jetson Orin Nano") -> dict:
    values = np.asarray(durations_ms, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or elapsed_seconds <= 0:
        raise ValueError("Benchmark needs measured, finite durations and positive elapsed time")
    return {"hardware": hardware, "target_hardware": target_hardware,
            "operating_system": platform.platform(), "python_version": platform.python_version(),
            "framework": framework, "framework_version": framework_version,
            "model_version": model_version, "data_manifest_id": data_manifest_id,
            "warmup_policy": warmup_policy, "repetitions": repetitions,
            "prediction_unit": prediction_unit, "mean_latency_ms": float(values.mean()),
            "median_latency_ms": float(np.median(values)), "p95_latency_ms": float(np.percentile(values, 95)),
            "throughput_per_second": float(len(values) / elapsed_seconds),
            "memory_usage_bytes": memory_usage_bytes, "timing_scope": timing_scope}


def save_report(report: dict, destination: Path) -> None:
    missing = set(REQUIRED) - set(report)
    if missing:
        raise ValueError(f"Benchmark report lacks fields: {sorted(missing)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
