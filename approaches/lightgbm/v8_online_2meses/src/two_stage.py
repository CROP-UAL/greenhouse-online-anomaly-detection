"""Reproducible eight-class training and two-stage streaming evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)

from config import (DATA_RAW_PATH, COLUMNAS_SENSORES, INYECCION_CONFIG,
                    RANGOS_FISICOS, SEED_INYECCION_TRAIN,
                    SEED_INYECCION_VAL, SEED_INYECCION_TEST)
from src.artifacts import file_sha256, load_manifest
from src.binary_postprocess import (FrozenCalibration, RuleConfig, RULE_METHODS,
                                    RULE_NONE, debounce_one, rule_codes, rule_one)
from src.features import cargar_y_regularizar_datos, split_temporal_81dias
from src.injection import inyectar_anomalias_sinteticas
from src.multiclass_features import (EXTRA_FEATURES, RULE_PROXIMAL_FEATURES,
                                     LOW, HIGH, SharedFeatureExtractor,
                                     raw_arrays, stream_matrix)

CLASSES = (
    "Caida Sistema", "Fallo Parcial Sistema", "Datos Faltantes",
    "Sensor Atascado", "Ruido", "Valores Fuera de Rango",
    "Desviacion de Correlacion", "Contextual",
)
ALIASES = {
    "Caída Sistema": "Caida Sistema",
    "Desviación de Correlación": "Desviacion de Correlacion",
}
SCHEMA = "two-stage-multiclass-v1"
DEFAULT_BINARY = Path(__file__).resolve().parents[1] / "data/optimized/20261006T093721Z/manifest.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/two_stage"
SEEDS = {"train": SEED_INYECCION_TRAIN, "val": SEED_INYECCION_VAL,
         "test": SEED_INYECCION_TEST}
FORBIDDEN = {"Fecha", "target_anomalia", "tipo_anomalia", "pred_binaria",
             "probabilidad_anomalia", "anomaly_probability", "tipo_sugerido_regla"}
BASE_PARAMS = {
    "objective": "multiclass", "num_class": 8, "metric": "multi_logloss",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "min_data_in_leaf": 30, "feature_fraction": 0.9,
    "verbosity": -1, "seed": 42, "feature_fraction_seed": 42,
    "bagging_seed": 42, "deterministic": True, "force_col_wise": True,
    "num_threads": 1,
}


def canonical(value):
    item = ALIASES.get(str(value), str(value))
    if item != "Normal" and item not in CLASSES:
        raise ValueError(f"Unknown anomaly type: {value}")
    return item


def encode(labels):
    mapping = {name: i for i, name in enumerate(CLASSES)}
    return np.asarray([mapping[canonical(value)] for value in labels], np.int32)


def load_binary(path=DEFAULT_BINARY):
    path = Path(path).resolve()
    manifest = load_manifest(path)
    root = path.parent
    meta = json.loads((root / manifest["artifacts"]["features"]).read_text(encoding="utf-8"))
    names = tuple(meta["features"])
    fills = np.asarray(meta["fill_values"], np.float32)
    if len(names) != 50 or len(fills) != 50 or len(set(names)) != 50:
        raise ValueError("Expected frozen 50-feature binary model")
    if FORBIDDEN.intersection(names):
        raise ValueError("Binary feature schema contains an output or label")
    model = lgb.Booster(model_file=str(root / manifest["artifacts"]["model_lightgbm"]))
    if model.num_feature() != 50:
        raise ValueError("Binary model feature count mismatch")
    calibration = FrozenCalibration.load(root / manifest["artifacts"]["calibration"])
    with np.load(root / manifest["artifacts"]["runtime_test"]) as data:
        runtime = {key: data[key] for key in data.files}
    return {"path": path, "manifest": manifest, "names": names, "fills": fills,
            "model": model, "calibration": calibration, "runtime": runtime}


def prepare_frames(binary):
    """Recreate original injections without filtering out timeline rows."""
    raw = cargar_y_regularizar_datos(DATA_RAW_PATH)
    raw["Fecha"] = pd.to_datetime(raw["Fecha"])
    warmup, train, val, test = split_temporal_81dias(raw)
    originals = {"train": train, "val": val, "test": test}
    frames, origins = {}, {}
    policy = RuleConfig.from_dict(binary["manifest"]["rule_policy"])
    for split, original in originals.items():
        original_values, original_times, original_missing = raw_arrays(original)
        codes, _ = rule_codes(original_values, original_missing, original_times,
                              policy, LOW, HIGH)
        injected = inyectar_anomalias_sinteticas(
            original, sensor_cols=COLUMNAS_SENSORES,
            config=INYECCION_CONFIG, seed=SEEDS[split])
        injected["tipo_anomalia"] = injected["tipo_anomalia"].map(canonical)
        # The injector changes only previously normal rows. A raw rule-positive
        # remains a real acquisition anomaly, regardless of its rule label.
        origins[split] = np.where(injected["target_anomalia"].to_numpy() == 0,
                                  "normal", np.where(codes != RULE_NONE,
                                  "real", "sintetico"))
        frames[split] = injected
    verify_frozen_test(binary, frames["test"])
    return warmup, frames, origins


def verify_frozen_test(binary, test):
    """Fail before fitting if regenerated TEST differs from the frozen binary run."""
    runtime = binary["runtime"]
    raw, times, missing = raw_arrays(test)
    expected = runtime["raw_test"]
    if (raw.shape != expected.shape or
            not np.array_equal(raw, expected, equal_nan=True) or
            not np.array_equal(times, runtime["test_timestamps"]) or
            not np.array_equal(missing, runtime["test_missing"]) or
            not np.array_equal(test["target_anomalia"].to_numpy(np.uint8),
                               runtime["y_test"])):
        raise ValueError("Regenerated TEST differs from frozen binary runtime")
    trace = pd.read_csv(
        binary["path"].parent / binary["manifest"]["artifacts"]["test_decision_trace"])
    if (not np.array_equal(pd.to_datetime(trace["Fecha"]).to_numpy(dtype="datetime64[ns]").astype(np.int64), times)
            or not np.array_equal(trace["tipo_anomalia_auditoria"].map(canonical).to_numpy(),
                                  test["tipo_anomalia"].to_numpy())):
        raise ValueError("Regenerated TEST labels differ from frozen binary trace")


def audit_support(frames, origins):
    records = []
    for split, frame in frames.items():
        labels = frame["tipo_anomalia"].to_numpy()
        times = pd.to_datetime(frame["Fecha"]).to_numpy(dtype="datetime64[ns]")
        usable = (frame["usar_en_train"].fillna(False).astype(bool).to_numpy()
                  if "usar_en_train" in frame else np.ones(len(frame), bool))
        for name in CLASSES:
            indices = np.flatnonzero(labels == name)
            lengths = []
            if len(indices):
                breaks = np.flatnonzero((np.diff(indices) != 1) |
                    ((times[indices[1:]] - times[indices[:-1]]) != np.timedelta64(30, "s")))
                lengths = np.diff(np.r_[0, breaks + 1, len(indices)]).tolist()
            days = len(np.unique(times[indices].astype("datetime64[D]")))
            record = {
                "split": split.upper(), "clase": name, "filas": len(indices),
                "episodios": len(lengths), "duracion_min_muestras": min(lengths, default=0),
                "duracion_mediana_muestras": float(np.median(lengths)) if lengths else 0,
                "duracion_max_muestras": max(lengths, default=0),
                "dias_distintos": days,
                "porcentaje_anomalias": 100*len(indices) /
                    max(1, int((labels != "Normal").sum())),
                "filas_reales": int(np.sum(origins[split][indices] == "real")),
                "filas_sinteticas": int(np.sum(origins[split][indices] == "sintetico")),
                "filas_utilizables": int(usable[indices].sum()),
                "filas_excluidas_no_modelables": int((~usable[indices]).sum()),
            }
            records.append(record)
    audit = pd.DataFrame(records)
    insufficient = audit[
        (audit["filas_utilizables"] < audit["split"].map(
            {"TRAIN": 200, "VAL": 50, "TEST": 50})) |
        (audit["episodios"] < 5) | (audit["dias_distintos"] < 3)]
    if not insufficient.empty:
        raise ValueError("Insufficient eight-class support:\n" +
                         insufficient.to_string(index=False))
    return audit


def build_matrices(warmup, frames, binary):
    names, fills = binary["names"], binary["fills"]
    train = stream_matrix(frames["train"], warmup, names, fills)
    val = stream_matrix(frames["val"], frames["train"].iloc[-360:], names, fills)
    test = stream_matrix(frames["test"], frames["val"].iloc[-360:], names, fills)
    matrices = {"train": train, "val": val, "test": test}
    expected = binary["runtime"]["batch_features"]
    if test[:, :50].shape != expected.shape or not np.allclose(
            test[:, :50], expected, rtol=1e-5, atol=1e-4):
        raise ValueError("Shared extractor differs from frozen binary TEST features")
    return matrices


def selection(frame):
    anomaly = frame["target_anomalia"].to_numpy(np.uint8) == 1
    usable = (frame["usar_en_train"].fillna(False).astype(bool).to_numpy()
              if "usar_en_train" in frame else np.ones(len(frame), bool))
    return anomaly & usable


def fit_candidates(frames, matrices, binary):
    names = binary["names"] + EXTRA_FEATURES
    if len(set(names)) != len(names) or FORBIDDEN.intersection(names):
        raise ValueError("Forbidden or repeated model feature")
    selected = {split: selection(frame) for split, frame in frames.items()}
    y_train = encode(frames["train"].loc[selected["train"], "tipo_anomalia"])
    y_val = encode(frames["val"].loc[selected["val"], "tipo_anomalia"])
    counts = np.bincount(y_train, minlength=8)
    if np.any(counts == 0):
        raise ValueError("Missing TRAIN class")
    weights = len(y_train) / (8 * counts)
    candidates = {
        "binary_50": tuple(range(50)),
        "expanded_57": tuple(range(len(names))),
        "ablation_without_rule_proxies": tuple(
            i for i, name in enumerate(names) if name not in RULE_PROXIMAL_FEATURES),
    }
    results, models = [], {}
    for name, indices in candidates.items():
        columns = [names[i] for i in indices]
        train_data = lgb.Dataset(
            matrices["train"][selected["train"]][:, indices], label=y_train,
            weight=weights[y_train], feature_name=columns, free_raw_data=False)
        val_data = lgb.Dataset(
            matrices["val"][selected["val"]][:, indices], label=y_val,
            reference=train_data, feature_name=columns, free_raw_data=False)
        model = lgb.train(
            BASE_PARAMS, train_data, num_boost_round=500,
            valid_sets=[val_data], valid_names=["VAL"],
            callbacks=[lgb.early_stopping(50, verbose=False)])
        prediction = np.asarray(model.predict(
            matrices["val"][selected["val"]][:, indices],
            num_threads=1)).argmax(axis=1)
        score = float(f1_score(y_val, prediction, labels=range(8), average="macro",
                               zero_division=0))
        results.append({"candidate": name, "feature_count": len(indices),
                        "val_macro_f1": score, "best_iteration": model.best_iteration})
        models[name] = model
    ranking = [r for r in results if r["candidate"] != "ablation_without_rule_proxies"]
    winner = sorted(ranking, key=lambda r: (-r["val_macro_f1"], r["feature_count"]))[0]
    return models[winner["candidate"]], candidates[winner["candidate"]], results, weights
class EpisodeTracker:
    """Emit the final type only when a binary episode closes."""

    def __init__(self):
        self.active = None
        self.events = []

    def step(self, timestamp_ns, detected, predicted_type):
        timestamp_ns = int(timestamp_ns)
        if self.active is not None and (
                not detected or timestamp_ns - self.active["last_ns"] != 30_000_000_000):
            self.events.append({
                "inicio": str(np.datetime64(self.active["start_ns"], "ns")),
                "fin": str(np.datetime64(self.active["last_ns"], "ns")),
                "filas": self.active["rows"],
                "tipo_final": self.active["last_type"],
                "cambios_tipo": self.active["changes"],
            })
            self.active = None
        if detected:
            if predicted_type not in CLASSES:
                raise ValueError("A detected anomaly must have a canonical type")
            if self.active is None:
                self.active = {"start_ns": timestamp_ns, "last_ns": timestamp_ns,
                               "last_type": predicted_type, "rows": 1, "changes": 0}
            else:
                self.active["changes"] += int(predicted_type != self.active["last_type"])
                self.active["last_ns"] = timestamp_ns
                self.active["last_type"] = predicted_type
                self.active["rows"] += 1

    def flush(self):
        if self.active is not None:
            self.step(self.active["last_ns"] + 30_000_000_000, False, "Normal")
        return pd.DataFrame(self.events)


def evaluate_types(truth, predictions, labels):
    truth = np.asarray(truth)
    predictions = np.asarray(predictions)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=labels,
                                    average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
        "report": classification_report(truth, predictions, labels=labels,
                                         output_dict=True, zero_division=0),
    }


def isolated_evaluation(model, indices, frame, matrix):
    mask = selection(frame)
    truth = frame.loc[mask, "tipo_anomalia"].to_numpy()
    probs = np.asarray(model.predict(matrix[mask][:, indices], num_threads=1))
    if probs.shape != (len(truth), 8) or not np.isfinite(probs).all():
        raise ValueError("Invalid eight-class probability matrix")
    if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    chosen = np.asarray(CLASSES, object)[probs.argmax(axis=1)]
    return evaluate_types(truth, chosen, CLASSES)


def stream_test(binary, model, indices):
    """Replay frozen TEST one sample at a time, without reading truth to gate M2."""
    runtime = binary["runtime"]
    raw = runtime["raw_test"]
    times = runtime["test_timestamps"]
    missing = runtime["test_missing"]
    extractor = SharedFeatureExtractor(
        binary["names"], binary["fills"], runtime["history"],
        runtime["history_timestamps"], runtime["history_missing"], times[0])
    policy = RuleConfig.from_dict(binary["manifest"]["rule_policy"])
    _, rule_state = rule_codes(
        runtime["history"], runtime["history_missing"],
        runtime["history_timestamps"], policy, LOW, HIGH)
    calibration = binary["calibration"]
    tracker = EpisodeTracker()
    n = len(raw)
    probabilities = np.full((n, 8), np.nan, np.float32)
    bin_probs = np.empty(n, np.float32)
    decisions = np.empty(n, np.uint8)
    methods = np.empty(n, object)
    types = np.empty(n, object)
    long_count = short_count = np.int32(0)
    multi_ns = 0
    started_total = perf_counter_ns()
    for i in range(n):
        row = extractor.step(raw[i], times[i], missing[i])
        probability = float(binary["model"].predict(
            row[:50].reshape(1, -1), num_threads=1)[0])
        decision, long_count, short_count = debounce_one(
            probability, calibration.thresh_std, calibration.k_long,
            calibration.thresh_high, calibration.k_short,
            long_count, short_count)
        code = rule_one(raw[i], missing[i], int(times[i]), rule_state,
                        policy, LOW, HIGH)
        detected = bool(decision or code != RULE_NONE)
        bin_probs[i] = probability
        decisions[i] = int(detected)
        methods[i] = RULE_METHODS[code] if code != RULE_NONE else (
            "lightgbm_debouncing" if detected else "normal")
        if detected:
            started_multi = perf_counter_ns()
            scores = np.asarray(model.predict(
                row[list(indices)].reshape(1, -1), num_threads=1)[0], np.float32)
            multi_ns += perf_counter_ns() - started_multi
            if (scores.shape != (8,) or not np.isfinite(scores).all() or
                    np.any(scores < 0) or abs(float(scores.sum()) - 1) > 1e-5):
                raise ValueError("Invalid online eight-class probabilities")
            probabilities[i] = scores
            types[i] = CLASSES[int(scores.argmax())]
        else:
            types[i] = "Normal"
        tracker.step(times[i], detected, types[i])
    total_ns = perf_counter_ns() - started_total
    frozen_trace = pd.read_csv(
        binary["path"].parent /
        binary["manifest"]["artifacts"]["test_decision_trace"])
    if (not np.array_equal(decisions,
            frozen_trace["pred_target_anomalia"].to_numpy(np.uint8)) or
            not np.allclose(bin_probs,
            frozen_trace["probabilidad_binaria"].to_numpy(float), atol=1e-5)):
        raise ValueError("Online binary decisions differ from frozen TEST trace")
    trace = pd.DataFrame({
        "Fecha": pd.to_datetime(times, unit="ns"),
        "pred_binaria": decisions, "tipo_predicho": types,
        "estado_clasificacion": np.where(decisions == 1, "clasificado", "normal"),
        "metodo_deteccion_binaria": methods,
        "target_anomalia_auditoria": runtime["y_test"],
        "tipo_real_auditoria": frozen_trace["tipo_anomalia_auditoria"].map(canonical),
    })
    episodes = tracker.flush()
    benchmark = {
        "platform": "PC", "test_rows": n,
        "invocations_m2": int(decisions.sum()),
        "invocation_percentage": 100 * float(decisions.mean()),
        "total_two_stage_ms": total_ns / 1e6,
        "total_multiclass_predict_ms": multi_ns / 1e6,
        "additional_multiclass_us_per_test_row": multi_ns / n / 1e3,
        "multiclass_us_per_invocation": multi_ns / max(1, int(decisions.sum())) / 1e3,
    }
    return trace, probabilities, episodes, benchmark


def end_to_end_evaluation(trace, episodes):
    truth_binary = trace["target_anomalia_auditoria"].to_numpy(np.uint8)
    detected = trace["pred_binaria"].to_numpy(np.uint8)
    truth_type = trace["tipo_real_auditoria"].to_numpy()
    chosen = trace["tipo_predicho"].to_numpy()
    true_positive = (truth_binary == 1) & (detected == 1)
    false_positive = (truth_binary == 0) & (detected == 1)
    false_negative = (truth_binary == 1) & (detected == 0)
    report = evaluate_types(truth_type, chosen, ("Normal",) + CLASSES)
    report["binary_false_negatives"] = int(false_negative.sum())
    report["binary_false_positives"] = int(false_positive.sum())
    report["detected_anomalies"] = int(true_positive.sum())
    report["correct_type_among_detected"] = int(
        np.sum(chosen[true_positive] == truth_type[true_positive]))
    report["type_errors_among_detected"] = int(
        np.sum(chosen[true_positive] != truth_type[true_positive]))
    report["m2_invocations"] = int(detected.sum())
    report["m2_invocation_percentage"] = 100 * float(detected.mean())
    report["episodes"] = len(episodes)
    report["episodes_with_type_changes"] = int((episodes["cambios_tipo"] > 0).sum())
    return report


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def _manifest_digest(payload):
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_run(output_root, binary, model, indices, candidates, weights, audit,
             isolated, end_to_end, trace, probabilities, episodes, benchmark, matrix_rows):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / stamp
    run_dir.mkdir(exist_ok=False)
    names = binary["names"] + EXTRA_FEATURES
    selected_names = [names[i] for i in indices]
    selected_fills = [float(binary["fills"][i]) if i < 50 else 0.0 for i in indices]
    model.save_model(str(run_dir / "model.txt"))
    _write_json(run_dir / "encoder.json",
                {"classes": CLASSES, "mapping": {name: i for i, name in enumerate(CLASSES)}})
    _write_json(run_dir / "features.json",
                {"binary_features": binary["names"],
                 "binary_fills": binary["fills"].tolist(),
                 "extra_features": EXTRA_FEATURES,
                 "selected_features": selected_names,
                 "selected_fills": selected_fills,
                 "selected_indices": indices})
    _write_json(run_dir / "training.json",
                {"parameters": BASE_PARAMS, "class_weights": dict(zip(CLASSES, weights.tolist())),
                 "candidate_results": candidates, "seeds": SEEDS,
                 "best_iteration": model.best_iteration,
                 "rule_proxy_ablation": list(RULE_PROXIMAL_FEATURES),
                 "matrix_rows_before_anomaly_selection": matrix_rows,
                 "selection_after_feature_generation": True})
    audit.to_csv(run_dir / "support_audit.csv", index=False)
    _write_json(run_dir / "isolated_test_metrics.json", isolated)
    _write_json(run_dir / "end_to_end_test_metrics.json", end_to_end)
    trace.to_csv(run_dir / "test_trace.csv", index=False)
    episodes.to_csv(run_dir / "episode_summary.csv", index=False)
    np.savez_compressed(run_dir / "diagnostic_probabilities.npz",
                        type_probabilities=probabilities,
                        classes=np.asarray(CLASSES))
    _write_json(run_dir / "benchmark_pc.json", benchmark)
    artifacts = {
        "model": "model.txt", "encoder": "encoder.json", "features": "features.json",
        "training": "training.json", "support_audit": "support_audit.csv",
        "isolated_test_metrics": "isolated_test_metrics.json",
        "end_to_end_test_metrics": "end_to_end_test_metrics.json",
        "test_trace": "test_trace.csv", "episode_summary": "episode_summary.csv",
        "diagnostic_probabilities": "diagnostic_probabilities.npz",
        "benchmark_pc": "benchmark_pc.json",
    }
    manifest = {
        "schema_version": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "two_stage_eight_class_anomaly_classification",
        "classes": CLASSES, "binary_manifest": str(binary["path"]),
        "binary_manifest_file_sha256": file_sha256(binary["path"]),
        "raw_dataset_sha256": file_sha256(DATA_RAW_PATH),
        "binary_manifest_hash": binary["manifest"]["manifest_hash"],
        "binary_feature_schema_version": binary["manifest"]["feature_schema_version"],
        "model_feature_count": len(indices),
        "artifacts": artifacts,
        "hashes": {key: file_sha256(run_dir / value) for key, value in artifacts.items()},
    }
    manifest["manifest_hash"] = _manifest_digest(manifest)
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir / "manifest.json"


def load_run(path, binary_path=None):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_hash")
    if _manifest_digest(manifest) != digest or manifest.get("schema_version") != SCHEMA:
        raise ValueError("Invalid two-stage manifest")
    manifest["manifest_hash"] = digest
    if tuple(manifest["classes"]) != CLASSES:
        raise ValueError("Class order differs from canonical order")
    for key, relative in manifest["artifacts"].items():
        if file_sha256(path.parent / relative) != manifest["hashes"][key]:
            raise ValueError(f"Artifact hash mismatch: {key}")
    binary = load_binary(binary_path or manifest["binary_manifest"])
    if (file_sha256(binary["path"]) != manifest["binary_manifest_file_sha256"] or
            binary["manifest"]["manifest_hash"] != manifest["binary_manifest_hash"] or
            binary["manifest"]["feature_schema_version"] !=
            manifest["binary_feature_schema_version"]):
        raise ValueError("Different binary manifest dependency")
    features = json.loads((path.parent / manifest["artifacts"]["features"]).read_text())
    encoder = json.loads((path.parent / manifest["artifacts"]["encoder"]).read_text())
    if (tuple(encoder["classes"]) != CLASSES or
            encoder["mapping"] != {name: i for i, name in enumerate(CLASSES)} or
            tuple(features["binary_features"]) != binary["names"] or
            not np.array_equal(features["binary_fills"], binary["fills"]) or
            tuple(features["extra_features"]) != EXTRA_FEATURES or
            not np.array_equal(features["selected_fills"], [
                float(binary["fills"][i]) if i < 50 else 0.0
                for i in features["selected_indices"]]) or
            tuple(features["selected_features"]) != tuple(
                (binary["names"] + EXTRA_FEATURES)[i] for i in features["selected_indices"])):
        raise ValueError("Incompatible class or feature schema")
    model = lgb.Booster(model_file=str(path.parent / manifest["artifacts"]["model"]))
    if (model.num_feature() != len(features["selected_features"]) or
            model.feature_name() != features["selected_features"] or
            np.asarray(model.predict(np.zeros((1, model.num_feature()),
                                              np.float32), num_threads=1)).shape != (1, 8)):
        raise ValueError("Model feature order mismatch")
    return manifest, binary, model, tuple(features["selected_indices"])


def run_training(binary_path=DEFAULT_BINARY, output_root=DEFAULT_OUTPUT):
    binary = load_binary(binary_path)
    warmup, frames, origins = prepare_frames(binary)
    audit = audit_support(frames, origins)
    matrices = build_matrices(warmup, frames, binary)
    model, indices, candidates, weights = fit_candidates(frames, matrices, binary)
    isolated = isolated_evaluation(model, indices, frames["test"], matrices["test"])
    trace, probabilities, episodes, benchmark = stream_test(binary, model, indices)
    end_to_end = end_to_end_evaluation(trace, episodes)
    path = save_run(output_root, binary, model, indices, candidates, weights,
                    audit, isolated, end_to_end, trace, probabilities, episodes,
                    benchmark, {key.upper(): len(value) for key, value in frames.items()})
    load_run(path)
    return path


