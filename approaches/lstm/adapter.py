"""Train and export the separately versioned causal CIESOL LSTM run."""
from __future__ import annotations

import json
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
RUNS = ROOT / "experiments" / "runs" / "lstm_causal_v1"


def _tensorflow():
    import tensorflow as tf
    return tf


def _binary_model(tf):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20, 19)),
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


def train() -> Path:
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(42)
    m1, m2, physical, metadata = prepare_features()
    run = RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=False)
    x1_train, train_rows = windows(m1, "train")
    x1_val, val_rows = windows(m1, "validation")
    y1_train = physical["train"].tipo_anomalia.notna().to_numpy(np.int8)[train_rows]
    y1_val = physical["validation"].tipo_anomalia.notna().to_numpy(np.int8)[val_rows]
    model1 = _binary_model(tf)
    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy",
                   metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
                            tf.keras.metrics.AUC(name="roc_auc"), tf.keras.metrics.AUC(name="pr_auc", curve="PR")])
    path1 = run / "model1.keras"
    callbacks1 = [tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True),
                  tf.keras.callbacks.ModelCheckpoint(path1, monitor="val_pr_auc", mode="max", save_best_only=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=3, min_lr=1e-6)]
    history1 = model1.fit(x1_train, y1_train, validation_data=(x1_val, y1_val), epochs=50, batch_size=256,
                          class_weight=_weights(y1_train), callbacks=callbacks1, verbose=2)
    del x1_train
    model1 = tf.keras.models.load_model(path1)
    val_score = model1.predict(x1_val, batch_size=256, verbose=0).reshape(-1)
    thresholds = np.arange(0.05, 1.0, 0.05)
    threshold = float(max(thresholds, key=lambda t: (f1_score(y1_val, val_score >= t, zero_division=0), -t)))
    del x1_val
    anomaly_rows = {name: np.flatnonzero(physical[name].tipo_anomalia.notna().to_numpy()) for name in ("train", "validation")}
    anomaly_rows["train"] = anomaly_rows["train"][anomaly_rows["train"] >= 19]
    x2_train, train_targets = windows(m2, "train", targets=anomaly_rows["train"])
    x2_val, val_targets = windows(m2, "validation", targets=anomaly_rows["validation"])
    y2_train = np.asarray([SOURCE_CLASSES.index(x) for x in physical["train"].tipo_anomalia.iloc[train_targets]], np.int32)
    y2_val = np.asarray([SOURCE_CLASSES.index(x) for x in physical["validation"].tipo_anomalia.iloc[val_targets]], np.int32)

    class ValidationMacroF1(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            prediction = np.argmax(self.model.predict(x2_val, batch_size=256, verbose=0), axis=1)
            logs["val_macro_f1"] = float(f1_score(y2_val, prediction, labels=np.arange(6), average="macro", zero_division=0))

    model2 = _multiclass_model(tf, m2["train"].shape[1])
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    path2 = run / "model2.keras"
    callbacks2 = [ValidationMacroF1(),
                  tf.keras.callbacks.ModelCheckpoint(path2, monitor="val_macro_f1", mode="max", save_best_only=True),
                  tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=8, restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6)]
    history2 = model2.fit(x2_train, y2_train, validation_data=(x2_val, y2_val), epochs=50, batch_size=256,
                          class_weight=_weights(y2_train), callbacks=callbacks2, verbose=2)
    manifest = {"schema_version": "lstm-causal-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
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
    m1, m2, physical, metadata = prepare_features()
    if metadata["data_manifest_id"] != manifest["data_manifest_id"] or metadata["model1_features"] != manifest["feature_contract"]["model1_features"] or metadata["model2_features"] != manifest["feature_contract"]["model2_features"]:
        raise ValueError("LSTM feature or data contract changed")
    model1 = tf.keras.models.load_model(run / "model1.keras")
    model2 = tf.keras.models.load_model(run / "model2.keras", compile=False)
    x1, targets = windows(m1, "test")
    if len(targets) != len(physical["test"]) or not np.array_equal(targets, np.arange(len(targets))):
        raise ValueError("Test lacks a prediction for every physical target")
    scores = model1.predict(x1, batch_size=256, verbose=0).reshape(-1)
    detected = scores >= manifest["binary_threshold_validation_f1"]
    selected = np.flatnonzero(detected)
    x2, _ = windows(m2, "test", targets=selected)
    prediction = np.argmax(model2.predict(x2, batch_size=256, verbose=0), axis=1) if len(selected) else np.empty(0, int)
    stage2 = np.full(len(targets), None, dtype=object)
    stage2[selected] = [SOURCE_TO_CANONICAL[SOURCE_CLASSES[i]] for i in prediction]
    version = f"lstm-causal-v1-{run.name}"
    trace = make_trace(physical["test"], split="test", scores=scores, binary=detected, stage2=stage2,
                       model_version=version, data_manifest_id=metadata["data_manifest_id"])
    anomalies = np.flatnonzero(physical["test"].tipo_anomalia.notna().to_numpy())
    x2_oracle, _ = windows(m2, "test", targets=anomalies)
    oracle = np.argmax(model2.predict(x2_oracle, batch_size=256, verbose=0), axis=1)
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
