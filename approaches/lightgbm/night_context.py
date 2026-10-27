"""New LightGBM experiment adding the LSTM's explicit night indicator.

The imported 66-feature v8 reference and its frozen calibration are untouched.
Both models are retrained with their original tree hyperparameters and the
same physical snapshots; the added features are `is_night` from FECHA and
the signed indoor-minus-outdoor CO2 gradient already used by LSTM.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from approaches.lightgbm.adapter import _upstream, RUN
from shared.contract import ROOT, SOURCE_TO_CANONICAL, FILES, validate_splits, sha256
from shared.trace import make_trace

ADDED_FEATURES = ("is_night", "co2_gradient")
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 7
RUNS = ROOT / "experiments" / "runs" / "lightgbm_night_v1"


def night_flag(timestamps: pd.Series) -> np.ndarray:
    times = pd.to_datetime(timestamps)
    hour = times.dt.hour + times.dt.minute / 60 + times.dt.second / 3600
    return ((hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR)).to_numpy(np.float32)


def co2_gradient(frame: pd.DataFrame) -> np.ndarray:
    """Signed indoor-minus-outdoor CO2 relationship, matching LSTM."""
    return (frame.XCO2I - frame.PCO2EXT).to_numpy(np.float32)


def _features(*, include_test: bool):
    pipeline = _upstream()
    frames, table, matrices = pipeline._data(include_test=include_test)
    feature_names = tuple(pipeline.FEATURES) + ADDED_FEATURES
    augmented = {name: np.column_stack([
        matrix,
        night_flag(table[name].FECHA),
        co2_gradient(table[name]),
    ]).astype(np.float32)
                 for name, matrix in matrices.items()}
    if any(matrix.shape[1] != 68 for matrix in augmented.values()):
        raise ValueError("Context experiment must contain 66 original features plus is_night and co2_gradient")
    return pipeline, frames, table, augmented, feature_names


def train() -> Path:
    pipeline, frames, table, matrix, names = _features(include_test=False)
    reference_calibration = pipeline.FrozenCalibration.load(RUN / "binary_calibration.json")
    run = RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=False)
    train_label = table["train"].target_anomalia.to_numpy(np.uint8)
    val_label = table["val"].target_anomalia.to_numpy(np.uint8)
    counts = np.bincount(train_label, minlength=2)
    binary_params = {"objective": "binary", "metric": "auc", "learning_rate": 0.04,
                     "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 25,
                     "feature_fraction": 0.7, "verbosity": -1, "num_threads": 1,
                     "seed": 42, "deterministic": True, "force_col_wise": True,
                     "scale_pos_weight": float(counts[0] / max(counts[1], 1))}
    dt = lgb.Dataset(matrix["train"], label=train_label, feature_name=list(names))
    dv = lgb.Dataset(matrix["val"], label=val_label, reference=dt)
    binary_model = lgb.train(binary_params, dt, num_boost_round=400, valid_sets=[dv],
                             callbacks=[lgb.early_stopping(40, verbose=False)])
    binary_model.save_model(str(run / "model1.txt"))
    source_classes = pipeline.CLASSES
    chosen = {name: table[name].target_anomalia.to_numpy(np.uint8).astype(bool) for name in ("train", "val")}
    targets = {name: np.asarray([source_classes.index(value) for value in table[name].loc[chosen[name], "tipo_anomalia"]], np.int32)
               for name in ("train", "val")}
    type_counts = np.bincount(targets["train"], minlength=len(source_classes))
    weights = len(targets["train"]) / (len(source_classes) * type_counts)
    multiclass_params = {"objective": "multiclass", "num_class": len(source_classes), "metric": "multi_logloss",
                         "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 30,
                         "feature_fraction": 0.9, "verbosity": -1, "seed": 42,
                         "deterministic": True, "force_col_wise": True, "num_threads": 1}
    mt = lgb.Dataset(matrix["train"][chosen["train"]], label=targets["train"],
                     weight=weights[targets["train"]], feature_name=list(names))
    mv = lgb.Dataset(matrix["val"][chosen["val"]], label=targets["val"], reference=mt)
    multiclass_model = lgb.train(multiclass_params, mt, num_boost_round=500, valid_sets=[mv],
                                 callbacks=[lgb.early_stopping(50, verbose=False)])
    multiclass_model.save_model(str(run / "model2.txt"))
    validation_prediction = multiclass_model.predict(matrix["val"][chosen["val"]], num_threads=1).argmax(axis=1)
    manifest = {"schema_version": "lightgbm-night-context-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_manifest_id": validate_splits()["manifest_id"], "base_run": RUN.name,
                "base_binary_model_sha256": sha256(RUN / "binary_model.txt"),
                "feature_names": list(names), "night_hours": {"start": NIGHT_START_HOUR, "end": NIGHT_END_HOUR},
                "binary_params": binary_params, "multiclass_params": multiclass_params,
                "binary_calibration": reference_calibration.canonical(),
                "model1_sha256": sha256(run / "model1.txt"), "model2_sha256": sha256(run / "model2.txt"),
                "validation_model2_macro_f1": float(f1_score(targets["val"], validation_prediction,
                                                               labels=range(len(source_classes)), average="macro", zero_division=0))}
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run


def export_test(run: Path) -> tuple[Path, Path]:
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if sha256(run / "model1.txt") != manifest["model1_sha256"] or sha256(run / "model2.txt") != manifest["model2_sha256"]:
        raise ValueError("Night LightGBM model hash mismatch")
    pipeline, frames, table, matrices, names = _features(include_test=True)
    if list(names) != manifest["feature_names"] or manifest["data_manifest_id"] != validate_splits()["manifest_id"]:
        raise ValueError("Night LightGBM contract changed")
    physical = pd.read_parquet(FILES["test"])
    model1 = lgb.Booster(model_file=str(run / "model1.txt"))
    model2 = lgb.Booster(model_file=str(run / "model2.txt"))
    calibration = pipeline.FrozenCalibration.load(RUN / "binary_calibration.json")
    scores = model1.predict(matrices["test"], num_threads=1)
    debounced = pipeline.debounce_probabilities(scores, calibration.thresh_std, calibration.k_long,
                                                 calibration.thresh_high, calibration.k_short)
    rules = pipeline._rules(table["test"], frames["val"].iloc[-pipeline.HISTORY_ROWS:])
    detected = np.maximum(debounced, (rules != pipeline.RULE_NONE).astype(np.uint8))
    selected = np.flatnonzero(detected)
    stage2 = np.full(len(physical), None, dtype=object)
    if len(selected):
        prediction = model2.predict(matrices["test"][selected], num_threads=1).argmax(axis=1)
        stage2[selected] = [SOURCE_TO_CANONICAL[pipeline.CLASSES[i]] for i in prediction]
    version = f"lightgbm-night-context-v1-{run.name}"
    trace = make_trace(physical, split="test", scores=scores, binary=detected, stage2=stage2,
                       model_version=version, data_manifest_id=manifest["data_manifest_id"],
                       extra={"binary_rule_code": rules, "is_night": night_flag(physical.FECHA)})
    anomalies = np.flatnonzero(physical.tipo_anomalia.notna().to_numpy())
    oracle = model2.predict(matrices["test"][anomalies], num_threads=1).argmax(axis=1)
    diagnostic = pd.DataFrame({"timestamp": physical.FECHA.iloc[anomalies].to_numpy(), "split": "test",
                               "ground_truth_type": trace.ground_truth_type.iloc[anomalies].to_numpy(),
                               "stage_2_prediction": [SOURCE_TO_CANONICAL[pipeline.CLASSES[i]] for i in oracle],
                               "model_version": version, "data_manifest_id": manifest["data_manifest_id"]})
    out = ROOT / "results" / "traces"
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "lightgbm_night_test.parquet"
    diagnostic_path = out / "lightgbm_night_model2_isolated_test.parquet"
    trace.to_parquet(trace_path, index=False)
    diagnostic.to_parquet(diagnostic_path, index=False)
    return trace_path, diagnostic_path
