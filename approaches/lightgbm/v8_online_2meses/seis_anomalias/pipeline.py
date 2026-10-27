"""Training, causal inference, immutable artifacts, and device benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import COLUMNAS_SENSORES, RANGOS_FISICOS
from src.artifacts import file_sha256
from src.binary_postprocess import (
    FrozenCalibration, RULE_METHODS, RULE_NONE, binary_metrics,
    calibrate_binary, configured_rules, debounce_one, debounce_probabilities,
    rule_codes, rule_one,
)
from src.multiclass_features import LOW, HIGH, SharedFeatureExtractor

from .data import (FEATURES, SENSORS, absence_flags, arrays, build_matrices, prepare_splits,
                   stream_matrix, usable)
from .settings import BASE_FEATURES, CLASSES, DATASET, MASK_COLUMNS, RUNS, SCHEMA


def _json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               default=lambda x: x.item() if isinstance(x, np.generic)
                               else str(x)), encoding="utf-8")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _check_dataset(manifest: dict) -> None:
    if file_sha256(DATASET) != manifest["dataset_sha256"]:
        raise ValueError("Dataset differs from the trained run")


def _split_days(frame: pd.DataFrame) -> float:
    return max(len(frame) / 2880.0, 1e-9)


def _rules(frame: pd.DataFrame, history: pd.DataFrame) -> np.ndarray:
    raw, times, _ = arrays(frame)
    past, old_times, _ = arrays(history)
    policy = configured_rules()
    _, state = rule_codes(past, np.isnan(past).astype(np.uint8), old_times,
                          policy, LOW, HIGH)
    codes, _ = rule_codes(raw, np.isnan(raw).astype(np.uint8), times,
                          policy, LOW, HIGH, state=state)
    return codes


def train_binary() -> Path:
    originals, frames = prepare_splits()
    matrices, fills = build_matrices(originals, frames)
    train, val = frames["train"], frames["val"]
    train_mask, val_mask = usable(train), usable(val)
    y_train = train["target_anomalia"].to_numpy(np.uint8)
    y_val = val["target_anomalia"].to_numpy(np.uint8)
    if len(np.unique(y_train[train_mask])) != 2:
        raise ValueError("Binary TRAIN needs both labels")
    counts = np.bincount(y_train[train_mask], minlength=2)
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.04,
        "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 25,
        "feature_fraction": 0.7, "verbosity": -1, "num_threads": 1,
        "seed": 42, "deterministic": True, "force_col_wise": True,
        "scale_pos_weight": float(counts[0] / max(counts[1], 1)),
    }
    ds_train = lgb.Dataset(matrices["train"][train_mask],
                           label=y_train[train_mask], feature_name=list(FEATURES))
    ds_val = lgb.Dataset(matrices["val"][val_mask], label=y_val[val_mask],
                         reference=ds_train)
    model = lgb.train(params, ds_train, num_boost_round=400,
                      valid_sets=[ds_val],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
    probabilities = np.asarray(model.predict(matrices["val"], num_threads=1),
                               np.float32)
    guards = (_rules(val, train.iloc[-360:]) != RULE_NONE).astype(np.uint8)
    calibration, val_score = calibrate_binary(
        probabilities, y_val, days=_split_days(val),
        std_grid=np.asarray([0.20, 0.30, 0.40, 0.50], np.float32),
        long_grid=np.asarray([2, 3, 4], np.int32),
        high_grid=np.asarray([0.75, 0.85, 0.95], np.float32),
        short_grid=np.asarray([1, 2], np.int32),
        max_false_alarms_per_day=10.0, forced_positive=guards)
    RUNS.mkdir(parents=True, exist_ok=True)
    run = RUNS / _stamp()
    run.mkdir()
    model.save_model(str(run / "binary_model.txt"))
    calibration.save(run / "binary_calibration.json")
    _json(run / "binary_validation.json", {
        "metrics": val_score, "train_counts": counts.tolist(),
        "val_counts": np.bincount(y_val, minlength=2).tolist(),
        "best_iteration": model.best_iteration,
    })
    _json(run / "binary_manifest.json", {
        "schema_version": SCHEMA, "stage": "binary",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": file_sha256(DATASET),
        "features": FEATURES, "base_fills": fills.tolist(),
        "mask_columns": MASK_COLUMNS,
        "artifacts": {
            name: file_sha256(run / name) for name in
            ("binary_model.txt", "binary_calibration.json", "binary_validation.json")
        },
    })
    return run


def load_binary(run: Path):
    run = Path(run)
    manifest = _read(run / "binary_manifest.json")
    if manifest.get("schema_version") != SCHEMA or manifest.get("stage") != "binary":
        raise ValueError("Wrong binary schema")
    _check_dataset(manifest)
    if tuple(manifest["features"]) != FEATURES or tuple(manifest["mask_columns"]) != MASK_COLUMNS:
        raise ValueError("Wrong binary feature schema")
    for name, digest in manifest["artifacts"].items():
        if file_sha256(run / name) != digest:
            raise ValueError(f"Binary artifact hash mismatch: {name}")
    model = lgb.Booster(model_file=str(run / "binary_model.txt"))
    if model.num_feature() != len(FEATURES) or tuple(model.feature_name()) != FEATURES:
        raise ValueError("Binary model feature order mismatch")
    return manifest, model, FrozenCalibration.load(run / "binary_calibration.json")


def train_multiclass(run: Path) -> Path:
    run = Path(run)
    binary_manifest, _, _ = load_binary(run)
    if (run / "manifest.json").exists():
        raise FileExistsError(run / "manifest.json")
    originals, frames = prepare_splits()
    matrices, _ = build_matrices(originals, frames)
    chosen = {}
    y = {}
    for split in ("train", "val", "test"):
        frame = frames[split]
        chosen[split] = (frame["target_anomalia"].to_numpy(np.uint8) == 1) & usable(frame)
        labels = frame.loc[chosen[split], "tipo_anomalia"].to_numpy()
        y[split] = np.asarray([CLASSES.index(name) for name in labels], np.int32)
    counts = np.bincount(y["train"], minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError("Six-class TRAIN has a missing class")
    weights = len(y["train"]) / (len(CLASSES) * counts)
    params = {
        "objective": "multiclass", "num_class": len(CLASSES),
        "metric": "multi_logloss", "learning_rate": 0.05, "num_leaves": 31,
        "max_depth": 6, "min_data_in_leaf": 30, "feature_fraction": 0.9,
        "verbosity": -1, "seed": 42, "deterministic": True,
        "force_col_wise": True, "num_threads": 1,
    }
    ds_train = lgb.Dataset(matrices["train"][chosen["train"]],
                           label=y["train"], weight=weights[y["train"]],
                           feature_name=list(FEATURES))
    ds_val = lgb.Dataset(matrices["val"][chosen["val"]], label=y["val"],
                         reference=ds_train)
    model = lgb.train(params, ds_train, num_boost_round=500,
                      valid_sets=[ds_val],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    val_scores = np.asarray(model.predict(matrices["val"][chosen["val"]],
                                           num_threads=1))
    val_f1 = float(f1_score(y["val"], val_scores.argmax(axis=1),
                            labels=range(len(CLASSES)), average="macro",
                            zero_division=0))
    model.save_model(str(run / "multiclass_model.txt"))
    _json(run / "multiclass_validation.json", {
        "val_macro_f1": val_f1, "best_iteration": model.best_iteration,
        "class_weights": dict(zip(CLASSES, weights.tolist())),
        "support": {split: dict(zip(CLASSES,
                     np.bincount(y[split], minlength=len(CLASSES)).tolist()))
                    for split in y},
    })
    _json(run / "manifest.json", {
        "schema_version": SCHEMA, "stage": "complete",
        "classes": CLASSES, "features": FEATURES,
        "dataset_sha256": binary_manifest["dataset_sha256"],
        "binary_manifest_sha256": file_sha256(run / "binary_manifest.json"),
        "artifacts": {
            name: file_sha256(run / name) for name in
            ("multiclass_model.txt", "multiclass_validation.json")
        },
    })
    return run


def load_run(run: Path):
    run = Path(run)
    binary_manifest, binary_model, calibration = load_binary(run)
    manifest = _read(run / "manifest.json")
    if (manifest.get("schema_version") != SCHEMA or
            tuple(manifest.get("classes", ())) != CLASSES or
            tuple(manifest.get("features", ())) != FEATURES or
            manifest.get("dataset_sha256") != binary_manifest["dataset_sha256"] or
            manifest.get("binary_manifest_sha256") !=
            file_sha256(run / "binary_manifest.json")):
        raise ValueError("Incompatible six-class manifest")
    for name, digest in manifest["artifacts"].items():
        if file_sha256(run / name) != digest:
            raise ValueError(f"Multiclass artifact hash mismatch: {name}")
    model = lgb.Booster(model_file=str(run / "multiclass_model.txt"))
    if model.num_feature() != len(FEATURES) or tuple(model.feature_name()) != FEATURES:
        raise ValueError("Multiclass feature order mismatch")
    probe = np.asarray(model.predict(np.zeros((1, len(FEATURES)), np.float32),
                                     num_threads=1))
    if probe.shape != (1, len(CLASSES)):
        raise ValueError("Not a six-class model")
    return binary_model, calibration, model


def _binary_predictions(model, calibration, matrix, frame, history):
    probabilities = np.asarray(model.predict(matrix, num_threads=1), np.float32)
    ml = debounce_probabilities(
        probabilities, calibration.thresh_std, calibration.k_long,
        calibration.thresh_high, calibration.k_short)
    rules = _rules(frame, history)
    detected = np.maximum(ml, (rules != RULE_NONE).astype(np.uint8))
    return probabilities, detected, rules


def evaluate(run: Path) -> Path:
    run = Path(run)
    binary, calibration, multiclass = load_run(run)
    originals, frames = prepare_splits()
    matrices, _ = build_matrices(originals, frames)
    frame = frames["test"]
    original = originals["test"]
    matrix = matrices["test"]
    binary_prob, detected, rules = _binary_predictions(
        binary, calibration, matrix, frame, frames["val"].iloc[-360:])
    predictions = np.full(len(frame), "Normal", dtype=object)
    scores = np.full((len(frame), len(CLASSES)), np.nan, np.float32)
    picked = np.flatnonzero(detected)
    if len(picked):
        scores[picked] = multiclass.predict(matrix[picked], num_threads=1)
        predictions[picked] = np.asarray(CLASSES, object)[scores[picked].argmax(axis=1)]
    truth = frame["tipo_anomalia"].to_numpy()
    labels = ("Normal",) + CLASSES
    report = {
        "binary": binary_metrics(frame["target_anomalia"].to_numpy(np.uint8),
                                 detected, _split_days(frame)),
        "end_to_end_macro_f1": float(f1_score(truth, predictions, labels=labels,
                                             average="macro", zero_division=0)),
        "end_to_end_confusion_matrix": confusion_matrix(
            truth, predictions, labels=labels).tolist(),
        "end_to_end_report": classification_report(
            truth, predictions, labels=labels, output_dict=True,
            zero_division=0),
    }
    source = frame["tipo_origen"].to_numpy()
    methods = np.asarray(
        [RULE_METHODS[int(code)] if code else
         ("lightgbm_debouncing" if hit else "normal")
         for code, hit in zip(rules, detected)], object)
    trace = pd.DataFrame({
        "Fecha": frame["Fecha"].to_numpy(),
        "pred_binaria": detected, "probabilidad_binaria": binary_prob,
        "tipo_predicho": predictions, "metodo_deteccion_binaria": methods,
        "target_anomalia_auditoria": frame["target_anomalia"].to_numpy(),
        "tipo_real_auditoria": truth, "tipo_origen_auditoria": source,
        "origen_dato_auditoria": frame["origen_dato"].to_numpy(),
    })
    for sensor, column in zip(SENSORS, MASK_COLUMNS):
        trace[sensor] = frame[sensor].to_numpy()
        trace[column] = frame[column].to_numpy(np.uint8)
    trace["sensor_inferencia"] = _attribute(trace)
    results = run / "results" / _stamp()
    results.mkdir(parents=True)
    trace.to_csv(results / "test_trace.csv", index=False)
    np.savez_compressed(results / "truth_clean_test.npz",
                        clean=original[list(SENSORS)].to_numpy(np.float32),
                        injected=frame[list(SENSORS)].to_numpy(np.float32),
                        source=source.astype(str),
                        synthetic=(frame["origen_dato"].to_numpy() == "sintetico"))
    np.savez_compressed(results / "type_probabilities.npz", scores=scores,
                        classes=np.asarray(CLASSES))
    _json(results / "metrics.json", report)
    _json(results / "result_manifest.json", {
        "schema_version": SCHEMA,
        "run_manifest_sha256": file_sha256(run / "manifest.json"),
        "artifacts": {name: file_sha256(results / name) for name in (
            "test_trace.csv", "truth_clean_test.npz",
            "type_probabilities.npz", "metrics.json")},
    })
    return results


def _attribute(trace: pd.DataFrame) -> np.ndarray:
    """Observed per-sensor evidence; uncertain classes may abstain."""
    raw = trace[list(SENSORS)].to_numpy(np.float32)
    masks = trace[list(MASK_COLUMNS)].to_numpy(bool)
    labels = trace["tipo_predicho"].to_numpy()
    out = np.full(len(trace), "", object)
    low = np.asarray([RANGOS_FISICOS[s]["min"] for s in SENSORS])
    high = np.asarray([RANGOS_FISICOS[s]["max"] for s in SENSORS])
    for i, label in enumerate(labels):
        if label == "Datos Faltantes":
            out[i] = ",".join(np.asarray(SENSORS)[masks[i]])
        elif label == "Valores Fuera de Rango":
            out[i] = ",".join(np.asarray(SENSORS)[
                np.isfinite(raw[i]) & ((raw[i] < low) | (raw[i] > high))])
        elif label == "Contextual":
            out[i] = "PRAD" if np.isfinite(raw[i, 2]) and (
                pd.Timestamp(trace["Fecha"].iloc[i]).hour >= 23 or
                pd.Timestamp(trace["Fecha"].iloc[i]).hour <= 4) and raw[i, 2] > 20 else ""
        elif label == "Desviacion de Correlacion":
            out[i] = "XTINV" if np.isfinite(raw[i, 8]) and np.isfinite(raw[i, 4]) and (
                abs(float(raw[i, 8] - raw[i, 4])) > 8) else ""
        elif label == "Sensor Atascado":
            if i >= 9:
                tail = raw[i-9:i+1]
                candidates = np.isfinite(tail).all(axis=0) & (
                    np.ptp(tail, axis=0) < 0.05)
                # Night radiation and zero wind are naturally flat.
                candidates[[2, 3, 5]] = False
                if candidates.any():
                    out[i] = SENSORS[int(np.flatnonzero(candidates)[0])]
        elif label == "Ruido" and i > 0 and i + 1 < len(trace):
            center = raw[i]
            neighbours = (raw[i-1] + raw[i+1]) / 2
            diff = np.abs(center - neighbours)
            if np.isfinite(diff).any():
                out[i] = SENSORS[int(np.nanargmax(diff))]
    return out


def latest_run() -> Path:
    paths = sorted(RUNS.glob("*/manifest.json"))
    if not paths:
        raise FileNotFoundError("Train the six-class models first")
    return paths[-1].parent


def latest_results(run: Path) -> Path:
    paths = sorted((Path(run) / "results").glob("*/result_manifest.json"))
    if not paths:
        raise FileNotFoundError("Run evaluate first")
    return paths[-1].parent


def benchmark(run: Path, *, rows: int | None = None, repeats: int = 3) -> Path:
    run = Path(run)
    binary, calibration, multiclass = load_run(run)
    originals, frames = prepare_splits()
    frame, history = frames["test"], frames["val"].iloc[-360:]
    raw, times, flags = arrays(frame)
    old, old_times, _ = arrays(history)
    n = len(raw) if rows is None else min(len(raw), rows)
    if n <= 0 or repeats <= 0:
        raise ValueError("rows and repeats must be positive")
    durations = []
    last_predictions = None
    for _ in range(repeats):
        extractor = SharedFeatureExtractor(
            BASE_FEATURES, np.zeros(len(BASE_FEATURES), np.float32),
            old, old_times, absence_flags(old), times[0])
        _, rule_state = rule_codes(old, np.isnan(old).astype(np.uint8),
                                   old_times, configured_rules(), LOW, HIGH)
        long_count = short_count = np.int32(0)
        predicted = np.empty(n, object)
        for i in range(n):
            start = perf_counter_ns()
            base = extractor.step(raw[i], times[i], flags[i])
            features = np.concatenate((base, flags[i].astype(np.float32)))
            p = float(binary.predict(features.reshape(1, -1), num_threads=1)[0])
            ml, long_count, short_count = debounce_one(
                p, calibration.thresh_std, calibration.k_long,
                calibration.thresh_high, calibration.k_short,
                long_count, short_count)
            code = rule_one(raw[i], np.isnan(raw[i]).astype(np.uint8),
                            int(times[i]), rule_state, configured_rules(), LOW, HIGH)
            if ml or code != RULE_NONE:
                score = multiclass.predict(features.reshape(1, -1), num_threads=1)[0]
                predicted[i] = CLASSES[int(np.argmax(score))]
            else:
                predicted[i] = "Normal"
            durations.append(perf_counter_ns() - start)
        if last_predictions is not None and not np.array_equal(predicted, last_predictions):
            raise ValueError("Benchmark replay is nondeterministic")
        last_predictions = predicted
    traces = sorted((run / "results").glob("*/test_trace.csv"))
    parity = None
    if traces:
        expected = pd.read_csv(traces[-1], usecols=["tipo_predicho"])["tipo_predicho"].to_numpy()[:n]
        parity = bool(np.array_equal(last_predictions, expected))
        if not parity:
            raise ValueError("Sequential benchmark differs from saved TEST predictions")
    latencies = np.asarray(durations, np.float64) / 1e6
    report = {
        "parity_with_test_trace": parity,
        "platform": platform.platform(), "machine": platform.machine(),
        "backend": "lightgbm_native",
        "schema_version": SCHEMA, "run_manifest_sha256": file_sha256(run / "manifest.json"),
        "rows_per_pass": n, "repeats": repeats, "invocations_m2": int(
            np.isin(last_predictions, CLASSES).sum()),
        "mean_ms": float(latencies.mean()), "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_rows_per_second": float(1000 / latencies.mean()),
    }
    output = run / "benchmarks" / (_stamp() + ".json")
    output.parent.mkdir(exist_ok=True)
    _json(output, report)
    return output


def benchmark_binary(run: Path, *, rows: int | None = 2000,
                     repeats: int = 3) -> Path:
    """Measure only the binary online path; useful before fitting stage two."""
    run = Path(run)
    manifest, model, calibration = load_binary(run)
    originals, frames = prepare_splits()
    frame, history = frames["test"], frames["val"].iloc[-360:]
    raw, times, flags = arrays(frame)
    old, old_times, _ = arrays(history)
    n = len(raw) if rows is None else min(len(raw), rows)
    if n <= 0 or repeats <= 0:
        raise ValueError("rows and repeats must be positive")
    durations = []
    predictions = None
    for _ in range(repeats):
        extractor = SharedFeatureExtractor(
            BASE_FEATURES, np.asarray(manifest["base_fills"], np.float32),
            old, old_times, absence_flags(old), times[0])
        _, rule_state = rule_codes(old, np.isnan(old).astype(np.uint8),
                                   old_times, configured_rules(), LOW, HIGH)
        long_count = short_count = np.int32(0)
        current = np.empty(n, np.uint8)
        for i in range(n):
            start = perf_counter_ns()
            base = extractor.step(raw[i], times[i], flags[i])
            features = np.concatenate((base, flags[i].astype(np.float32)))
            probability = float(model.predict(features.reshape(1, -1),
                                              num_threads=1)[0])
            ml, long_count, short_count = debounce_one(
                probability, calibration.thresh_std, calibration.k_long,
                calibration.thresh_high, calibration.k_short,
                long_count, short_count)
            code = rule_one(raw[i], np.isnan(raw[i]).astype(np.uint8),
                            int(times[i]), rule_state, configured_rules(), LOW, HIGH)
            current[i] = int(bool(ml or code != RULE_NONE))
            durations.append(perf_counter_ns() - start)
        if predictions is not None and not np.array_equal(predictions, current):
            raise ValueError("Binary replay is nondeterministic")
        predictions = current
    traces = sorted((run / "results").glob("*/test_trace.csv"))
    parity = None
    if traces:
        expected = pd.read_csv(traces[-1], usecols=["pred_binaria"])["pred_binaria"].to_numpy(np.uint8)[:n]
        parity = bool(np.array_equal(predictions, expected))
        if not parity:
            raise ValueError("Sequential binary benchmark differs from saved TEST decisions")
    latency = np.asarray(durations, np.float64) / 1e6
    report = {
        "parity_with_test_trace": parity,
        "platform": platform.platform(), "machine": platform.machine(),
        "backend": "lightgbm_native",
        "schema_version": SCHEMA,
        "binary_manifest_sha256": file_sha256(run / "binary_manifest.json"),
        "rows_per_pass": n, "repeats": repeats,
        "metrics": binary_metrics(
            frame["target_anomalia"].to_numpy(np.uint8)[:n],
            predictions, n / 2880.0),
        "mean_ms": float(latency.mean()),
        "p50_ms": float(np.percentile(latency, 50)),
        "p95_ms": float(np.percentile(latency, 95)),
        "p99_ms": float(np.percentile(latency, 99)),
        "throughput_rows_per_second": float(1000 / latency.mean()),
    }
    output = run / "benchmarks" / ("binary_" + _stamp() + ".json")
    output.parent.mkdir(exist_ok=True)
    _json(output, report)
    return output


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    from .correction import correct
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("binary", "multiclass", "evaluate",
                                            "correct", "benchmark", "benchmark-binary"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.action == "binary":
        print(train_binary())
        return
    run = args.run or latest_run()
    if args.action == "benchmark-binary":
        print(benchmark_binary(run, rows=args.rows, repeats=args.repeats))
    elif args.action == "multiclass":
        print(train_multiclass(run))
    elif args.action == "evaluate":
        print(evaluate(run))
    elif args.action == "correct":
        print(correct(args.results or latest_results(run)))
    else:
        print(benchmark(run, rows=args.rows, repeats=args.repeats))


if __name__ == "__main__":
    main()
