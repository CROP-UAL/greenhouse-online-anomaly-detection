"""TRAIN-only TreeSHAP audit and type-aware Top-K selection."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class ShapSelection:
    selected: tuple[str, ...]
    global_ranking: tuple[str, ...]
    type_rankings: dict[str, tuple[str, ...]]
    report: pd.DataFrame

def _sample_indices(labels: np.ndarray, types: np.ndarray, max_per_group: int,
                    seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = ["__normal__"] + sorted(set(types[labels == 1].astype(str)))
    chosen = []
    for group in groups:
        candidates = np.flatnonzero(labels == 0) if group == "__normal__" else np.flatnonzero(
            (labels == 1) & (types.astype(str) == group))
        if len(candidates) > max_per_group:
            candidates = rng.choice(candidates, max_per_group, replace=False)
        chosen.append(np.sort(candidates))
    return np.sort(np.concatenate(chosen))

def tree_shap_rankings(model, x_train: pd.DataFrame, y_train, synthetic_types, *,
                       max_per_group: int = 2000, seed: int = 42):
    """Use LightGBM TreeSHAP. No type classifier is trained."""
    labels = np.asarray(y_train, np.uint8)
    types = np.asarray(synthetic_types, str)
    rng = np.random.default_rng(seed)
    anomaly_types = sorted(set(types[labels == 1]))
    # Global ranking follows TRAIN prevalence through an unstratified sample.
    global_size = min(len(x_train), max_per_group*(len(anomaly_types)+1))
    global_indices = np.sort(rng.choice(
        len(x_train), global_size, replace=False))
    global_matrix = np.asarray(x_train.iloc[global_indices], np.float32)
    contributions = np.asarray(
        model.predict(global_matrix, pred_contrib=True), np.float64)
    if contributions.shape[1] == global_matrix.shape[1] + 1:
        contributions = contributions[:, :-1]
    global_mean = np.abs(contributions).mean(axis=0)
    global_order = np.argsort(-global_mean, kind="stable")
    rankings = {}
    importances = {}
    normal = np.flatnonzero(labels == 0)
    if len(normal) > max_per_group:
        normal = np.sort(rng.choice(normal, max_per_group, replace=False))
    for anomaly_type in anomaly_types:
        anomalous = np.flatnonzero((labels == 1) & (types == anomaly_type))
        if len(anomalous) > max_per_group:
            anomalous = np.sort(rng.choice(
                anomalous, max_per_group, replace=False))
        indices = np.concatenate((normal, anomalous))
        matrix = np.asarray(x_train.iloc[indices], np.float32)
        conditional_shap = np.asarray(
            model.predict(matrix, pred_contrib=True), np.float64)
        if conditional_shap.shape[1] == matrix.shape[1] + 1:
            conditional_shap = conditional_shap[:, :-1]
        conditional = np.abs(conditional_shap).mean(axis=0)
        order = np.argsort(-conditional, kind="stable")
        rankings[anomaly_type] = tuple(x_train.columns[order])
        importances[anomaly_type] = conditional
    return tuple(x_train.columns[global_order]), global_mean, rankings, importances

def select_type_aware_top_k(model, x_train: pd.DataFrame, y_train,
                            synthetic_types, k: int, *, reserve_per_type: int = 5,
                            max_per_group: int = 2000, seed: int = 42) -> ShapSelection:
    global_rank, global_imp, rankings, type_imp = tree_shap_rankings(
        model, x_train, y_train, synthetic_types,
        max_per_group=max_per_group, seed=seed)
    reserved = []
    for anomaly_type in sorted(rankings):
        for feature in rankings[anomaly_type][:reserve_per_type]:
            if feature not in reserved:
                reserved.append(feature)
    if len(reserved) > k:
        raise ValueError(f"K={k} cannot reserve {reserve_per_type} features for every type")
    selected = reserved + [name for name in global_rank if name not in reserved]
    selected = tuple(selected[:k])
    rows = []
    for anomaly_type, ranking in rankings.items():
        importance = type_imp[anomaly_type]
        index = {name: i for i, name in enumerate(x_train.columns)}
        covered_top5 = sum(name in selected for name in ranking[:reserve_per_type])
        total = float(importance.sum())
        covered = float(sum(importance[index[name]] for name in selected))
        rows.append({"anomaly_type": anomaly_type, "k": k,
                     "top5_covered": covered_top5,
                     "top5_required": reserve_per_type,
                     "top5_coverage_pct": 100*covered_top5/reserve_per_type,
                     "shap_importance_covered_pct": 100*covered/total if total else 100.0})
    return ShapSelection(selected, global_rank, rankings, pd.DataFrame(rows))

def save_shap_audit(selection: ShapSelection, directory: str | Path) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    selection.report.to_csv(destination/"shap_coverage_by_type.csv", index=False)
    payload = {"selected": list(selection.selected),
               "global_ranking": list(selection.global_ranking),
               "type_rankings": {k: list(v) for k, v in selection.type_rankings.items()}}
    (destination/"shap_rankings.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def binary_shap_ranking(model, x_train: pd.DataFrame, *,
                        sample_size: int = 14000, seed: int = 42) -> tuple[str, ...]:
    """Select with binary-model contributions only; anomaly type is never read."""
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(
        len(x_train), min(sample_size, len(x_train)), replace=False))
    matrix = np.asarray(x_train.iloc[indices], np.float32)
    contributions = np.asarray(
        model.predict(matrix, pred_contrib=True), np.float64)
    if contributions.shape[1] == matrix.shape[1]+1:
        contributions = contributions[:, :-1]
    importance = np.abs(contributions).mean(axis=0)
    return tuple(x_train.columns[np.argsort(-importance, kind="stable")])
