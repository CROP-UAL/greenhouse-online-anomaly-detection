"""Matched, warm, complete causal test-replay benchmark for Jetson.

This measures throughput from preloaded physical observations to final labels.
It leaves the trained models, feature definitions, thresholds and routing intact.
LSTM feature preparation is vectorized over the test prefix, so the resulting
ratio describes full replay throughput, not live per-observation latency.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd

from approaches.lightgbm.night_benchmark import run_benchmark as benchmark_lightgbm
from approaches.lstm.adapter import SOURCE_CLASSES
from approaches.lstm.causal_data import prepare_features, transform_test_for_benchmark
from shared.contract import ROOT, SOURCE_TO_CANONICAL, sha256


def _lstm_replay(run: Path, *, rows: int, repeats: int, expected_version: str,
                 expected_manifest: str) -> dict:
    import tensorflow as tf

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != "lstm-causal-v2":
        raise ValueError("Full replay benchmark expects the paired causal v2 LSTM")
    if manifest["data_manifest_id"] != expected_manifest:
        raise ValueError("LSTM run uses another physical data manifest")
    if expected_version != f"{manifest['schema_version']}-{run.name}":
        raise ValueError("LSTM run does not match the paired quality report")
    for name in ("model1", "model2"):
        if sha256(run / f"{name}.keras") != manifest[f"{name}_sha256"]:
            raise ValueError(f"LSTM {name} hash differs from run manifest")

    # Data loading, scaler fitting and model loading are setup, as for LightGBM.
    m1, m2, physical, metadata, state = prepare_features(
        stuck_context=True, return_benchmark_state=True)
    if metadata["data_manifest_id"] != expected_manifest:
        raise ValueError("LSTM data manifest changed")
    n = min(rows, len(physical["test"]))
    raw_test = physical["test"].iloc[:n]
    reference = pd.read_parquet(ROOT / "results" / "traces" / "lstm_test.parquet")
    if len(reference) < n or not raw_test.FECHA.reset_index(drop=True).equals(
            reference.timestamp.iloc[:n].reset_index(drop=True)):
        raise ValueError("LSTM benchmark targets differ from the paired trace")
    if reference.model_version.nunique() != 1 or reference.model_version.iloc[0] != expected_version:
        raise ValueError("LSTM trace uses another model run")
    if reference.data_manifest_id.nunique() != 1 or reference.data_manifest_id.iloc[0] != expected_manifest:
        raise ValueError("LSTM trace uses another data manifest")

    model1 = tf.keras.models.load_model(run / "model1.keras", compile=False)
    model2 = tf.keras.models.load_model(run / "model2.keras", compile=False)
    threshold = float(manifest["binary_threshold_validation_f1"])
    reference_labels = reference.final_prediction.iloc[:n].to_numpy()
    durations = []
    maximum_feature_difference = 0.0
    stage2_invocations = None
    for pass_number in range(repeats + 1):
        start = perf_counter_ns()
        test_m1, test_m2 = transform_test_for_benchmark(raw_test, state)
        # Window context assembly belongs to the measured preprocessing path.
        sequence1 = np.concatenate([state["history_model1"], test_m1])
        sequence2 = np.concatenate([state["history_model2"], test_m2])
        prepared = perf_counter_ns()
        difference = float(np.max(np.abs(test_m2 - m2["test"][:n])))
        maximum_feature_difference = max(maximum_feature_difference, difference)
        if not np.array_equal(test_m1, m1["test"][:n]) or difference > 1e-3:
            raise ValueError(f"LSTM benchmark changed the feature contract: maximum difference {difference}")
        # Context is the same 19 scaled validation observations used by export_test.
        predictions = np.empty(n, dtype=object)
        routed = 0
        inference_start = perf_counter_ns()
        for index in range(n):
            score = float(model1(sequence1[index:index + 20][None], training=False).numpy()[0, 0])
            if score >= threshold:
                probabilities = model2(sequence2[index:index + 20][None], training=False).numpy()[0]
                predictions[index] = SOURCE_TO_CANONICAL[SOURCE_CLASSES[int(np.argmax(probabilities))]]
                routed += 1
            else:
                predictions[index] = "Normal"
        finished = perf_counter_ns()
        if not np.array_equal(predictions, reference_labels):
            different = int(np.count_nonzero(predictions != reference_labels))
            raise ValueError(f"LSTM benchmark differs from the paired trace on {different} targets")
        if stage2_invocations is None:
            stage2_invocations = routed
        elif routed != stage2_invocations:
            raise ValueError("LSTM routing changed between timed passes")
        if pass_number:
            durations.append({"preprocessing_ns": prepared - start,
                              "inference_ns": finished - inference_start})

    preprocessing = sum(item["preprocessing_ns"] for item in durations)
    inference = sum(item["inference_ns"] for item in durations)
    total = preprocessing + inference
    pass_seconds = [(item["preprocessing_ns"] + item["inference_ns"]) / 1e9 for item in durations]
    median_pass = float(np.median(pass_seconds))
    pass_relative_span = (max(pass_seconds) - min(pass_seconds)) / median_pass
    if pass_relative_span > 0.20:
        raise ValueError(f"LSTM pass timing span {pass_relative_span:.1%} exceeds 20%; repeat under stable power/thermal conditions")
    del model1, model2
    tf.keras.backend.clear_session()
    gc.collect()
    return {"framework": "TensorFlow/Keras", "framework_version": tf.__version__,
            "model_version": expected_version, "rows_per_pass": n, "repetitions": repeats,
            "warmup_policy": "one full untimed pass", "stage2_invocations_per_pass": stage2_invocations,
            "test_preprocessing_seconds": preprocessing / 1e9,
            "model_inference_seconds": inference / 1e9,
            "total_seconds": total / 1e9,
            "pass_seconds": pass_seconds,
            "pass_relative_span": pass_relative_span,
            "mean_ms_per_target": total / (n * repeats * 1e6),
            "throughput_targets_per_second": n * repeats * 1e9 / total,
            "maximum_model2_feature_difference": maximum_feature_difference,
            "parity_with_test_trace": True}


def run_pair(lightgbm_run: Path, lstm_run: Path, *, rows: int = 2000,
             repeats: int = 5, output: Path | None = None,
             quality_path: Path | None = None) -> Path:
    if rows <= 0 or repeats <= 0:
        raise ValueError("Rows and repetitions must be positive")
    if platform.machine().lower() != "aarch64" or "tegra" not in platform.release().lower():
        raise RuntimeError("The publishable paired benchmark must run on the Jetson Orin Nano")
    model_path = Path("/proc/device-tree/model")
    device_model = model_path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace") if model_path.is_file() else ""
    if "Orin Nano" not in device_model:
        raise RuntimeError(f"Expected Jetson Orin Nano, found {device_model or 'unknown device'}")
    release_path = Path("/etc/nv_tegra_release")
    l4t_release = release_path.read_text(encoding="utf-8", errors="replace").strip() if release_path.is_file() else None
    try:
        jetpack_query = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
                                       capture_output=True, text=True, timeout=5, check=False)
        jetpack_version = jetpack_query.stdout.strip() if jetpack_query.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        jetpack_version = None
    try:
        power_query = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                                     timeout=5, check=False)
        power_mode = (power_query.stdout + power_query.stderr).strip() or None
    except (OSError, subprocess.TimeoutExpired):
        power_mode = None
    lightgbm_run = Path(lightgbm_run).resolve()
    lstm_run = Path(lstm_run).resolve()
    for path in (lightgbm_run / "manifest.json", lightgbm_run / "model1.txt",
                 lightgbm_run / "model2.txt", lstm_run / "manifest.json",
                 lstm_run / "model1.keras", lstm_run / "model2.keras"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired run artifact: {path}")
    if quality_path is None:
        quality_path = ROOT / "results" / "metrics" / "paired_test.json"
    quality = json.loads(Path(quality_path).read_text(encoding="utf-8"))
    if quality["paired_test_rows"] != 31680:
        raise ValueError("Paired quality report lacks the full test")
    light_version = quality["lightgbm"]["model_version"]
    lstm_version = quality["lstm"]["model_version"]
    manifest_id = quality["lightgbm"]["data_manifest_id"]
    if quality["lstm"]["data_manifest_id"] != manifest_id:
        raise ValueError("Paired runs use different physical data")
    if light_version != f"lightgbm-night-context-v1-{lightgbm_run.name}":
        raise ValueError("LightGBM run does not match the paired quality report")
    if lstm_version != f"lstm-causal-v2-{lstm_run.name}":
        raise ValueError("LSTM run does not match the paired quality report")
    light_manifest = json.loads((lightgbm_run / "manifest.json").read_text(encoding="utf-8"))
    lstm_manifest = json.loads((lstm_run / "manifest.json").read_text(encoding="utf-8"))
    try:
        commit_query = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True, timeout=5, check=False)
        code_commit = commit_query.stdout.strip() if commit_query.returncode == 0 else None
        status_query = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                      capture_output=True, text=True, timeout=5, check=False)
        code_dirty = bool(status_query.stdout.strip()) if status_query.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        code_commit, code_dirty = None, None

    report_dir = ROOT / "results" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    before_path = report_dir / f"full_replay_lightgbm_before_{stamp}.json"
    after_path = report_dir / f"full_replay_lightgbm_after_{stamp}.json"
    before_path = benchmark_lightgbm(lightgbm_run, rows=rows, repeats=repeats,
                                     output=before_path, masks_inside_timer=True)
    light_before = json.loads(before_path.read_text(encoding="utf-8"))
    lstm = _lstm_replay(lstm_run, rows=rows, repeats=repeats,
                        expected_version=lstm_version, expected_manifest=manifest_id)
    after_path = benchmark_lightgbm(lightgbm_run, rows=rows, repeats=repeats,
                                    output=after_path, masks_inside_timer=True)
    light_after = json.loads(after_path.read_text(encoding="utf-8"))
    for report in (light_before, light_after):
        if (report["parity_with_test_trace"] is not True or
                report["rows_per_pass"] != lstm["rows_per_pass"] or
                report.get("masks_inside_timer") is not True):
            raise ValueError("LightGBM replay lacks parity or matched target count")
        if report["model_version"] != light_version or report["data_manifest_id"] != manifest_id:
            raise ValueError("LightGBM replay uses another model or physical data")
    before_ms = 1000.0 / light_before["throughput_per_second"]
    after_ms = 1000.0 / light_after["throughput_per_second"]
    mean_light_ms = (before_ms + after_ms) / 2
    drift = abs(before_ms - after_ms) / mean_light_ms
    if drift > 0.10:
        raise ValueError(f"LightGBM before/after timing drift {drift:.1%} exceeds 10%; repeat under stable power/thermal conditions")
    result = {
        "schema_version": "matched-causal-full-replay-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": platform.platform(), "jetson_model": device_model,
        "jetpack_version": jetpack_version, "l4t_release": l4t_release,
        "nvpmodel_query": power_mode,
        "architecture": platform.machine(),
        "code_commit": code_commit, "code_dirty": code_dirty,
        "python_version": platform.python_version(), "data_manifest_id": manifest_id,
        "model_hashes": {"lightgbm": {"model1": light_manifest["model1_sha256"],
                                     "model2": light_manifest["model2_sha256"]},
                         "lstm": {"model1": lstm_manifest["model1_sha256"],
                                  "model2": lstm_manifest["model2_sha256"]}},
        "prediction_unit": "one final decision per identical physical test timestamp",
        "rows_per_pass": lstm["rows_per_pass"], "repetitions": repeats,
        "warmup_policy": "one full untimed pass per framework",
        "timing_scope": "preloaded physical test prefix through causal feature preparation, Model 1, original routing, conditional Model 2 and final label; excludes file I/O, model loading and train-fitted scaler setup",
        "execution_policy": "LightGBM extracts features per target; LSTM prepares the causal test prefix vectorially then infers each target sequentially; ratio measures full replay throughput, not live single-target latency",
        "lightgbm": {"model_version": light_version, "framework_version": light_before["framework_version"],
                     "before_ms_per_target": before_ms, "after_ms_per_target": after_ms,
                     "mean_ms_per_target": mean_light_ms, "before_after_relative_drift": drift,
                     "throughput_targets_per_second": 1000.0 / mean_light_ms,
                     "stage2_invocations_per_pass": int(pd.read_parquet(ROOT / "results" / "traces" / "lightgbm_night_test.parquet", columns=["stage_2_executed"]).stage_2_executed.iloc[:lstm["rows_per_pass"]].sum()),
                     "parity_with_test_trace": True},
        "lstm": lstm,
        "speed_ratio_lightgbm_over_lstm_full_replay": lstm["mean_ms_per_target"] / mean_light_ms,
        "ratio_interpretation": "LightGBM complete causal test replay throughput is this many times the LSTM replay throughput on the matched target prefix",
        "lightgbm_before_report": before_path.name, "lightgbm_after_report": after_path.name,
    }
    if output is None:
        output = report_dir / f"full_pipeline_replay_{stamp}.json"
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightgbm-run", type=Path, required=True)
    parser.add_argument("--lstm-run", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quality-report", type=Path)
    args = parser.parse_args()
    result = run_pair(args.lightgbm_run, args.lstm_run, rows=args.rows,
                      repeats=args.repeats, output=args.output,
                      quality_path=args.quality_report)
    print(result)
    report = json.loads(result.read_text(encoding="utf-8"))
    print(f"Full replay speed ratio LightGBM/LSTM: {report['speed_ratio_lightgbm_over_lstm_full_replay']:.3f}x")


if __name__ == "__main__":
    main()
