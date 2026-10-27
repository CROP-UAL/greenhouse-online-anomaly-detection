"""Sequential online benchmark for the 68-feature night/context experiment."""
from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter_ns

import lightgbm as lgb
import numpy as np
import pandas as pd

from approaches.lightgbm.adapter import RUN, _upstream
from approaches.lightgbm.night_context import ADDED_FEATURES
from shared.benchmarking.report import build_report, save_report
from shared.contract import ROOT, SOURCE_TO_CANONICAL, sha256, validate_splits


def run_benchmark(run: Path, *, rows: int = 2000, repeats: int = 5) -> Path:
    """Time causal extraction, both model decisions and LightGBM's rules per target."""
    if rows <= 0 or repeats <= 0:
        raise ValueError("Rows and repetitions must be positive")
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["data_manifest_id"] != validate_splits()["manifest_id"] or manifest["feature_names"][-2:] != list(ADDED_FEATURES):
        raise ValueError("Night/context run uses a different feature or data contract")
    if any(sha256(run / f"model{i}.txt") != manifest[f"model{i}_sha256"] for i in (1, 2)):
        raise ValueError("Night/context model hash mismatch")
    pipeline = _upstream()
    frames = pipeline._frames(include_test=True)
    test = frames["test"]
    history = frames["val"].iloc[-pipeline.HISTORY_ROWS:]
    n = min(rows, len(test))
    raw = np.ascontiguousarray(test.loc[:, pipeline.SENSORS].to_numpy(np.float32)[:n])
    times = test.Fecha.to_numpy("datetime64[ns]").astype(np.int64)[:n]
    physical_times = test.FECHA.to_numpy("datetime64[ns]").astype(np.int64)[:n]
    masks = np.ascontiguousarray(test.loc[:, pipeline.MASK_COLUMNS].to_numpy(np.uint8)[:n])
    old = np.ascontiguousarray(history.loc[:, pipeline.SENSORS].to_numpy(np.float32))
    old_times = history.Fecha.to_numpy("datetime64[ns]").astype(np.int64)
    old_masks = np.ascontiguousarray(history.loc[:, pipeline.MASK_COLUMNS].to_numpy(np.uint8))
    lower = np.asarray([pipeline.RANGOS_FISICOS[s]["min"] for s in pipeline.SENSORS], np.float32)
    upper = np.asarray([pipeline.RANGOS_FISICOS[s]["max"] for s in pipeline.SENSORS], np.float32)
    policy = pipeline.configured_rules()
    model1 = lgb.Booster(model_file=str(run / "model1.txt"))
    model2 = lgb.Booster(model_file=str(run / "model2.txt"))
    calibration = pipeline.FrozenCalibration.load(RUN / "binary_calibration.json")
    co2_inside = pipeline.SENSORS.index("XCO2I")
    co2_outside = pipeline.SENSORS.index("PCO2EXT")
    samples, elapsed_ns = [], 0
    last = None
    for pass_index in range(repeats + 1):
        extractor = pipeline.SharedFeatureExtractor(pipeline.BASE_FEATURES,
            np.zeros(len(pipeline.BASE_FEATURES), np.float32), old, old_times, old_masks, times[0])
        rule_state = pipeline.RuleState()
        for values, instant in zip(old, old_times):
            pipeline.rule_one(values, np.isnan(values).astype(np.uint8), int(instant), rule_state,
                              policy, lower, upper)
        long_count = short_count = 0
        predictions = np.empty(n, dtype=object)
        durations = np.empty(n, dtype=np.int64)
        pass_start = perf_counter_ns()
        for index in range(n):
            start = perf_counter_ns()
            vector = np.empty(len(manifest["feature_names"]), np.float32)
            cut = len(pipeline.BASE_FEATURES) + len(pipeline.EXTRA_FEATURES)
            vector[:cut] = extractor.step(raw[index], times[index], masks[index])
            vector[cut:len(pipeline.FEATURES)] = masks[index]
            physical_ns = int(physical_times[index]) % pipeline.DAY_NS
            vector[pipeline.COS_HOUR_INDEX] = np.float32(np.cos(2 * np.pi * physical_ns / pipeline.DAY_NS))
            vector[-2] = np.float32(physical_ns >= 20 * 3600 * 1_000_000_000 or physical_ns < 7 * 3600 * 1_000_000_000)
            vector[-1] = raw[index, co2_inside] - raw[index, co2_outside]
            probability = float(model1.predict(vector.reshape(1, -1), num_threads=1)[0])
            detected, long_count, short_count = pipeline.debounce_one(probability,
                calibration.thresh_std, calibration.k_long, calibration.thresh_high,
                calibration.k_short, long_count, short_count)
            rule = pipeline.rule_one(raw[index], np.isnan(raw[index]).astype(np.uint8),
                                     int(times[index]), rule_state, policy, lower, upper)
            if detected or rule != pipeline.RULE_NONE:
                source_label = pipeline.CLASSES[int(np.argmax(model2.predict(vector.reshape(1, -1), num_threads=1)[0]))]
                predictions[index] = SOURCE_TO_CANONICAL[source_label]
            else:
                predictions[index] = "Normal"
            durations[index] = perf_counter_ns() - start
        if pass_index == 0:
            continue
        elapsed_ns += perf_counter_ns() - pass_start
        samples.append(durations)
        if last is not None and not np.array_equal(last, predictions):
            raise ValueError("Sequential benchmark is nondeterministic")
        last = predictions.copy()
    trace_path = ROOT / "results" / "traces" / "lightgbm_night_test.parquet"
    parity = None
    if trace_path.exists():
        trace = pd.read_parquet(trace_path, columns=["final_prediction"])
        parity = bool(np.array_equal(last, trace.final_prediction.to_numpy()[:n]))
        if not parity:
            raise ValueError("Sequential night/context benchmark differs from exported trace")
    report = build_report(hardware=platform.machine(), framework="LightGBM", framework_version=lgb.__version__,
        model_version=f"lightgbm-night-context-v1-{run.name}", data_manifest_id=manifest["data_manifest_id"],
        warmup_policy="one full untimed pass", repetitions=repeats, prediction_unit="target observation",
        durations_ms=np.concatenate(samples) / 1e6, elapsed_seconds=elapsed_ns / 1e9,
        memory_usage_bytes=None,
        timing_scope="causal features including night and CO2 gradient + binary + rules + conditional multiclass; excludes loading and I/O")
    report["parity_with_test_trace"] = parity
    report["rows_per_pass"] = n
    output = ROOT / "results" / "reports" / f"lightgbm_night_benchmark_{run.name}.json"
    save_report(report, output)
    return output
