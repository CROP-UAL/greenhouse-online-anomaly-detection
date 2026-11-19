"""Train and export the separately versioned causal Agroconnect LSTM run."""
from __future__ import annotations

import json
import gc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from approaches.lstm.causal_data import prepare_features, windows
from shared.contract import ROOT, ANOMALIES, SOURCE_TO_CANONICAL, SPLITS, sha256
from shared.trace import make_trace

SOURCE_CLASSES = tuple(SOURCE_TO_CANONICAL)
RUNS = ROOT / "experiments" / "runs" / "lstm_causal_v2"
VERSION = "lstm-causal-v2"
BATCH_SIZE = 256


def _tensorflow():
    import tensorflow as tf
    return tf


def _binary_model(tf, features: int = 20):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20, features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2), tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def _multiclass_model(tf, features: int):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20, features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2), tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(6, activation="softmax"),
    ])


def _weights(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    values = compute_class_weight("balanced", classes=classes, y=y)
    return {int(k): float(v) for k, v in zip(classes, values)}


def _training_dataset(tf, matrices: dict[str, np.ndarray], split: str, targets: np.ndarray,
                      labels: np.ndarray, *, class_weights: dict[int, float] | None = None):
    """Yield only one window batch at a time so Keras cannot copy all windows to GPU."""
    targets = np.asarray(targets, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int32)
    if len(targets) != len(labels) or not len(targets):
        raise ValueError("Training targets and labels must be nonempty and aligned")
    rng = np.random.default_rng(42)
    features = matrices[split].shape[1]
    weights = (np.asarray([class_weights[int(y)] for y in labels], dtype=np.float32)
               if class_weights is not None else None)

    def batches():
        order = rng.permutation(len(targets)) if weights is not None else np.arange(len(targets))
        for start in range(0, len(order), BATCH_SIZE):
            positions = order[start:start + BATCH_SIZE]
            x, _ = windows(matrices, split, targets=targets[positions])
            if weights is None:
                yield x, labels[positions]
            else:
                yield x, labels[positions], weights[positions]

    signature = (tf.TensorSpec((None, 20, features), tf.float32), tf.TensorSpec((None,), tf.int32))
    if weights is not None:
        signature += (tf.TensorSpec((None,), tf.float32),)
    return tf.data.Dataset.from_generator(batches, output_signature=signature).prefetch(1)


def _predict_rows(model, matrices: dict[str, np.ndarray], split: str,
                  targets: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    if not len(targets):
        return np.empty((0, model.output_shape[-1]), dtype=np.float32)
    predictions = []
    for start in range(0, len(targets), batch_size):
        x, _ = windows(matrices, split, targets=targets[start:start + batch_size])
        predictions.append(np.asarray(model.predict_on_batch(x)))
    return np.concatenate(predictions, axis=0)


def train() -> Path:
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(42)
    m1, m2, physical, metadata = prepare_features(stuck_context=True)
    run = RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=False)
    train_rows = np.arange(19, len(m1["train"]), dtype=np.int64)
    val_rows = np.arange(len(m1["validation"]), dtype=np.int64)
    y1_train = physical["train"].tipo_anomalia.notna().to_numpy(np.int8)[train_rows]
    y1_val = physical["validation"].tipo_anomalia.notna().to_numpy(np.int8)[val_rows]
    model1 = _binary_model(tf, m1["train"].shape[1])
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy",
                   metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
                            tf.keras.metrics.AUC(name="roc_auc"), tf.keras.metrics.AUC(name="pr_auc", curve="PR")])
    path1 = run / "model1.keras"
    callbacks1 = [tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True),
                  tf.keras.callbacks.ModelCheckpoint(path1, monitor="val_pr_auc", mode="max", save_best_only=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=3, min_lr=1e-6)]
    train1 = _training_dataset(tf, m1, "train", train_rows, y1_train, class_weights=_weights(y1_train))
    val1 = _training_dataset(tf, m1, "validation", val_rows, y1_val)
    history1 = model1.fit(train1, validation_data=val1, epochs=50, callbacks=callbacks1, verbose=2)
    del model1, train1, val1
    tf.keras.backend.clear_session()
    gc.collect()
    model1 = tf.keras.models.load_model(path1, compile=False)
    val_score = _predict_rows(model1, m1, "validation", val_rows).reshape(-1)
    thresholds = np.arange(0.05, 1.0, 0.05)
    threshold = float(max(thresholds, key=lambda t: (f1_score(y1_val, val_score >= t, zero_division=0), -t)))
    del model1
    tf.keras.backend.clear_session()
    gc.collect()
    anomaly_rows = {name: np.flatnonzero(physical[name].tipo_anomalia.notna().to_numpy()) for name in ("train", "validation")}
    anomaly_rows["train"] = anomaly_rows["train"][anomaly_rows["train"] >= 19]
    train_targets = anomaly_rows["train"]
    val_targets = anomaly_rows["validation"]
    y2_train = np.asarray([SOURCE_CLASSES.index(x) for x in physical["train"].tipo_anomalia.iloc[train_targets]], np.int32)
    y2_val = np.asarray([SOURCE_CLASSES.index(x) for x in physical["validation"].tipo_anomalia.iloc[val_targets]], np.int32)

    class ValidationMacroF1(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            prediction = np.argmax(_predict_rows(self.model, m2, "validation", val_targets), axis=1)
            logs["val_macro_f1"] = float(f1_score(y2_val, prediction, labels=np.arange(6), average="macro", zero_division=0))

    model2 = _multiclass_model(tf, m2["train"].shape[1])
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    path2 = run / "model2.keras"
    callbacks2 = [ValidationMacroF1(),
                  tf.keras.callbacks.ModelCheckpoint(path2, monitor="val_macro_f1", mode="max", save_best_only=True),
                  tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=8, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6)]
    train2 = _training_dataset(tf, m2, "train", train_targets, y2_train, class_weights=_weights(y2_train))
    val2 = _training_dataset(tf, m2, "validation", val_targets, y2_val)
    history2 = model2.fit(train2, validation_data=val2, epochs=50, callbacks=callbacks2, verbose=2)
    manifest = {"schema_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_manifest_id": metadata["data_manifest_id"], "seed": 42, "window_length": 20,
                "binary_threshold_validation_f1": threshold, "tensorflow": tf.__version__,
                "feature_contract": metadata, "class_order_source": list(SOURCE_CLASSES),
                "model1_sha256": sha256(path1), "model2_sha256": sha256(path2),
                "model1_training_history": history1.history, "model2_training_history": history2.history}
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return run


def export_test(run: Path) -> tuple[Path, Path]:
    tf = _tensorflow()
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if sha256(run / "model1.keras") != manifest["model1_sha256"] or sha256(run / "model2.keras") != manifest["model2_sha256"]:
        raise ValueError("LSTM model hashes changed")
    version_name = manifest["schema_version"]
    if version_name not in ("lstm-causal-v1", VERSION):
        raise ValueError(f"Unknown LSTM schema: {version_name}")
    m1, m2, physical, metadata = prepare_features(stuck_context=version_name == VERSION)
    if metadata["data_manifest_id"] != manifest["data_manifest_id"] or metadata["model1_features"] != manifest["feature_contract"]["model1_features"] or metadata["model2_features"] != manifest["feature_contract"]["model2_features"]:
        raise ValueError("LSTM feature or data contract changed")
    model1 = tf.keras.models.load_model(run / "model1.keras", compile=False)
    model2 = tf.keras.models.load_model(run / "model2.keras", compile=False)
    targets = np.arange(len(m1["test"]), dtype=np.int64)
    if len(targets) != len(physical["test"]) or not np.array_equal(targets, np.arange(len(targets))):
        raise ValueError("Test lacks a prediction for every physical target")
    scores = _predict_rows(model1, m1, "test", targets).reshape(-1)
    detected = scores >= manifest["binary_threshold_validation_f1"]
    selected = np.flatnonzero(detected)
    prediction = np.argmax(_predict_rows(model2, m2, "test", selected), axis=1) if len(selected) else np.empty(0, int)
    stage2 = np.full(len(targets), None, dtype=object)
    stage2[selected] = [SOURCE_TO_CANONICAL[SOURCE_CLASSES[i]] for i in prediction]
    version = f"{version_name}-{run.name}"
    trace = make_trace(physical["test"], split="test", scores=scores, binary=detected, stage2=stage2,
                       model_version=version, data_manifest_id=metadata["data_manifest_id"])
    anomalies = np.flatnonzero(physical["test"].tipo_anomalia.notna().to_numpy())
    oracle = np.argmax(_predict_rows(model2, m2, "test", anomalies), axis=1)
    diagnostic = pd.DataFrame({"timestamp": physical["test"].FECHA.iloc[anomalies].to_numpy(), "split": "test",
                               "ground_truth_type": trace.ground_truth_type.iloc[anomalies].to_numpy(),
                               "stage_2_prediction": [SOURCE_TO_CANONICAL[SOURCE_CLASSES[i]] for i in oracle],
                               "model_version": version, "data_manifest_id": metadata["data_manifest_id"]})
    out = ROOT / "results" / "traces"
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "lstm_test.parquet"
    isolated_path = out / "lstm_model2_isolated_test.parquet"
    trace.to_parquet(trace_path, index=False)
    diagnostic.to_parquet(isolated_path, index=False)
    return trace_path, isolated_path
