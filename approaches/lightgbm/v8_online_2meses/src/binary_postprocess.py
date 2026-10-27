"""Binary-only guards, multi-route debouncer and immutable calibration."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate

CALIBRATION_SCHEMA = "binary-debouncer-v1"

@dataclass(frozen=True)
class FrozenCalibration:
    thresh_std: np.float32
    k_long: np.int32
    thresh_high: np.float32
    k_short: np.int32
    schema_version: str = CALIBRATION_SCHEMA

    def canonical(self) -> dict:
        return {"schema_version": self.schema_version,
                "thresh_std": float(self.thresh_std), "k_long": int(self.k_long),
                "thresh_high": float(self.thresh_high), "k_short": int(self.k_short)}

    @property
    def calibration_hash(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return sha256(raw).hexdigest()

    def save(self, path: str | Path) -> None:
        payload = self.canonical() | {"calibration_hash": self.calibration_hash}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrozenCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        item = cls(np.float32(payload["thresh_std"]), np.int32(payload["k_long"]),
                   np.float32(payload["thresh_high"]), np.int32(payload["k_short"]),
                   payload["schema_version"])
        if payload.get("calibration_hash") != item.calibration_hash:
            raise ValueError("Calibration hash mismatch; regenerate the artifact")
        return item

@njit(cache=True)
def debounce_probabilities(probabilities, thresh_std, k_long, thresh_high, k_short):
    """Two generic routes: persistent evidence OR instantaneous high evidence."""
    result = np.zeros(probabilities.shape[0], np.uint8)
    long_count = short_count = 0
    for i in range(probabilities.shape[0]):
        probability = probabilities[i]
        long_count = long_count + 1 if probability >= thresh_std else 0
        short_count = short_count + 1 if probability >= thresh_high else 0
        result[i] = np.uint8(long_count >= k_long or short_count >= k_short)
    return result

@njit(cache=True)
def debounce_one(probability, thresh_std, k_long, thresh_high, k_short,
                 long_count, short_count):
    long_count = long_count + 1 if probability >= thresh_std else 0
    short_count = short_count + 1 if probability >= thresh_high else 0
    decision = np.uint8(long_count >= k_long or short_count >= k_short)
    return decision, np.int32(long_count), np.int32(short_count)

@njit(cache=True)
def guard_one(raw, missing, physical_min, physical_max):
    for i in range(raw.shape[0]):
        if missing[i] != 0 or not np.isfinite(raw[i]):
            return np.uint8(1)
        if raw[i] < physical_min[i] or raw[i] > physical_max[i]:
            return np.uint8(1)
    return np.uint8(0)

@njit(cache=True)
def false_alarm_episodes(y_true, y_pred):
    episodes = 0
    active = False
    for i in range(y_pred.shape[0]):
        false_positive = y_true[i] == 0 and y_pred[i] == 1
        if false_positive and not active:
            episodes += 1
        active = false_positive
    return episodes

@njit(cache=True)
def binary_counts(y_true, y_pred):
    tp = fp = fn = tn = 0
    for i in range(y_true.shape[0]):
        if y_true[i] == 1 and y_pred[i] == 1: tp += 1
        elif y_true[i] == 0 and y_pred[i] == 1: fp += 1
        elif y_true[i] == 1: fn += 1
        else: tn += 1
    return tp, fp, fn, tn

def binary_metrics(y_true, y_pred, days: float) -> dict:
    truth = np.asarray(y_true, np.uint8)
    prediction = np.asarray(y_pred, np.uint8)
    tp, fp, fn, tn = map(int, binary_counts(truth, prediction))
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    episodes = int(false_alarm_episodes(truth, prediction))
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "false_alarm_episodes": episodes,
            "false_alarm_episodes_per_day": episodes/days}

@njit(cache=True)
def _calibration_grid(probabilities, y_true, forced_positive, std_grid, long_grid,
                      high_grid, short_grid, days, max_fa, min_precision,
                      min_recall):
    best = np.array([-1e9, 0.0, 0.0, 0.0, 0.0], np.float64)
    best_cfg = np.zeros(4, np.float64)
    for std in std_grid:
        for klong in long_grid:
            for high in high_grid:
                for kshort in short_grid:
                    pred = debounce_probabilities(probabilities, std, klong, high, kshort)
                    for index in range(pred.shape[0]):
                        if forced_positive[index] != 0:
                            pred[index] = np.uint8(1)
                    tp, fp, fn, _ = binary_counts(y_true, pred)
                    precision = tp/(tp+fp) if tp+fp else 0.0
                    recall = tp/(tp+fn) if tp+fn else 0.0
                    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
                    fa = false_alarm_episodes(y_true, pred)/days
                    feasible = (fa <= max_fa and precision >= min_precision and
                                recall >= min_recall)
                    # Prefer the highest F1 among configurations that satisfy
                    # the operational constraints. The fallback remains useful
                    # for diagnosis when VAL has no feasible configuration.
                    penalty = max(0.0, min_precision-precision)
                    penalty += max(0.0, min_recall-recall)
                    penalty += max(0.0, fa-max_fa)
                    score = f1 if feasible else f1 - 10.0*penalty - 1.0
                    if score > best[0]:
                        best[0] = score
                        best[1] = precision
                        best[2] = recall
                        best[3] = f1
                        best[4] = fa
                        best_cfg[0] = std
                        best_cfg[1] = klong
                        best_cfg[2] = high
                        best_cfg[3] = kshort
    return best_cfg, best

def calibrate_binary(probabilities, y_true, *, days: float,
                     std_grid, long_grid, high_grid, short_grid,
                     max_false_alarms_per_day: float = 10.0,
                     forced_positive=None, min_precision: float = 0.0,
                     min_recall: float = 0.0):
    """Calibrate binary streaming decisions, including deterministic guards when set."""
    if forced_positive is None:
        forced = np.zeros(len(probabilities), np.uint8)
    else:
        forced = np.asarray(forced_positive, np.uint8)
        if forced.shape != np.asarray(probabilities).shape:
            raise ValueError("forced_positive must have one entry per probability")
    cfg, scores = _calibration_grid(
        np.asarray(probabilities, np.float32), np.asarray(y_true, np.uint8), forced,
        np.asarray(std_grid, np.float32), np.asarray(long_grid, np.int32),
        np.asarray(high_grid, np.float32), np.asarray(short_grid, np.int32),
        float(days), float(max_false_alarms_per_day), float(min_precision),
        float(min_recall))
    frozen = FrozenCalibration(np.float32(cfg[0]), np.int32(cfg[1]),
                               np.float32(cfg[2]), np.int32(cfg[3]))
    return frozen, {"objective": float(scores[0]), "precision": float(scores[1]),
                    "recall": float(scores[2]), "f1": float(scores[3]),
                    "false_alarm_episodes_per_day": float(scores[4])}

def deterministic_guards(raw_values, missing_flags, physical_min, physical_max):
    raw = np.asarray(raw_values, np.float32)
    missing = np.asarray(missing_flags, np.uint8)
    return np.any(missing != 0, axis=1) | np.any(~np.isfinite(raw), axis=1) | (
        np.any((raw < physical_min) | (raw > physical_max), axis=1))


# Ordered, causal infrastructure and data-quality rules. Code zero means no
# deterministic decision; it is the only case passed to the ML decision.
RULE_NONE, RULE_CAIDA, RULE_PARCIAL, RULE_MISSING, RULE_RANGE = range(5)
RULE_LABELS = {
    RULE_CAIDA: "Ca\u00edda Sistema",
    RULE_PARCIAL: "Fallo Parcial Sistema",
    RULE_MISSING: "Datos Faltantes",
    RULE_RANGE: "Valores Fuera de Rango",
}
RULE_METHODS = {
    RULE_CAIDA: "regla_caida_sistema",
    RULE_PARCIAL: "regla_fallo_parcial_sistema",
    RULE_MISSING: "regla_datos_faltantes",
    RULE_RANGE: "regla_fuera_rango",
}

@dataclass(frozen=True)
class RuleConfig:
    sampling_seconds: int
    caida_min_minutes: float
    parcial_min_minutes: float
    parcial_min_nan: int
    parcial_min_nan_with_zero: int
    parcial_min_zero: int

    @property
    def caida_samples(self) -> int:
        return int(np.ceil(60*self.caida_min_minutes/self.sampling_seconds))

    @property
    def parcial_samples(self) -> int:
        return int(np.ceil(60*self.parcial_min_minutes/self.sampling_seconds))

    def canonical(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RuleConfig":
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


def configured_rules() -> RuleConfig:
    from config import (
        MUESTREO_SEGUNDOS, CAIDA_SISTEMA_MIN_DURACION_MIN,
        FALLO_PARCIAL_SISTEMA_MIN_DURACION_MIN,
        FALLO_PARCIAL_SISTEMA_MIN_SENSORES_NAN,
        FALLO_PARCIAL_SISTEMA_MIN_SENSORES_NAN_CON_CEROS,
        FALLO_PARCIAL_SISTEMA_MIN_SENSORES_CERO,
    )
    return RuleConfig(
        MUESTREO_SEGUNDOS, CAIDA_SISTEMA_MIN_DURACION_MIN,
        FALLO_PARCIAL_SISTEMA_MIN_DURACION_MIN,
        FALLO_PARCIAL_SISTEMA_MIN_SENSORES_NAN,
        FALLO_PARCIAL_SISTEMA_MIN_SENSORES_NAN_CON_CEROS,
        FALLO_PARCIAL_SISTEMA_MIN_SENSORES_CERO,
    )


@dataclass
class RuleState:
    caida_count: int = 0
    parcial_count: int = 0
    last_timestamp_ns: int | None = None


def rule_one(raw_values, missing_flags, timestamp_ns: int, state: RuleState,
             config: RuleConfig, physical_min, physical_max) -> int:
    """Classify current row using only current and earlier observations.

    A five-minute event at 30 s sampling is confirmed on its tenth row.
    Earlier rows keep their contemporaneous missing-data classification.
    """
    raw = np.asarray(raw_values, np.float32)
    missing = np.asarray(missing_flags, np.uint8)
    if raw.ndim != 1 or missing.shape != raw.shape:
        raise ValueError("raw and missing must be equally sized sensor vectors")
    timestamp_ns = int(timestamp_ns)
    if state.last_timestamp_ns is not None and (
            timestamp_ns-state.last_timestamp_ns != config.sampling_seconds*1_000_000_000):
        state.caida_count = state.parcial_count = 0
    state.last_timestamp_ns = timestamp_ns
    absent = (missing != 0) | np.isnan(raw)
    zero = (~absent) & np.isfinite(raw) & (raw == 0)
    n_nan, n_zero = int(absent.sum()), int(zero.sum())
    total = n_nan == len(raw)
    partial = (not total and (
        n_nan >= config.parcial_min_nan or (
            n_nan >= config.parcial_min_nan_with_zero and
            n_zero >= config.parcial_min_zero)))
    state.caida_count = state.caida_count+1 if total else 0
    state.parcial_count = state.parcial_count+1 if partial else 0
    if state.caida_count >= config.caida_samples:
        return RULE_CAIDA
    if state.parcial_count >= config.parcial_samples:
        return RULE_PARCIAL
    if n_nan:
        return RULE_MISSING
    if np.any(~np.isfinite(raw)) or np.any(raw < physical_min) or np.any(raw > physical_max):
        return RULE_RANGE
    return RULE_NONE


def rule_codes(raw_values, missing_flags, timestamps_ns, config: RuleConfig,
               physical_min, physical_max, *, state: RuleState | None = None):
    """Return rule codes and continuation state, including split boundaries."""
    raw = np.asarray(raw_values, np.float32)
    missing = np.asarray(missing_flags, np.uint8)
    times = np.asarray(timestamps_ns, np.int64)
    if raw.ndim != 2 or missing.shape != raw.shape or len(times) != len(raw):
        raise ValueError("Invalid rule input shapes")
    state = RuleState() if state is None else state
    codes = np.empty(len(raw), np.uint8)
    for i in range(len(raw)):
        codes[i] = rule_one(raw[i], missing[i], times[i], state,
                            config, physical_min, physical_max)
    return codes, state


def compose_decisions(ml_decisions, codes):
    """The type describes only deterministic decisions; ML remains binary."""
    ml = np.asarray(ml_decisions, np.uint8)
    rules = np.asarray(codes, np.uint8)
    if ml.shape != rules.shape:
        raise ValueError("ML decisions and rule codes must align")
    prediction = np.maximum(ml, (rules != RULE_NONE).astype(np.uint8))
    types = np.empty(len(ml), dtype=object)
    methods = np.empty(len(ml), dtype=object)
    for i, code in enumerate(rules):
        if code:
            types[i], methods[i] = RULE_LABELS[int(code)], RULE_METHODS[int(code)]
        elif ml[i]:
            types[i], methods[i] = "Anomal\u00eda binaria", "lightgbm"
        else:
            types[i], methods[i] = "Normal", "normal"
    return prediction, types, methods


def count_episodes(labels, timestamps_ns, sample_seconds: int) -> tuple[int, int]:
    """Count rows and contiguous episodes of one truth or prediction type."""
    mask = np.asarray(labels, bool)
    times = np.asarray(timestamps_ns, np.int64)
    if len(mask) != len(times):
        raise ValueError("mask and timestamps must align")
    rows = int(mask.sum())
    starts = 0
    step = sample_seconds*1_000_000_000
    for i in range(len(mask)):
        if mask[i] and (i == 0 or not mask[i-1] or times[i]-times[i-1] != step):
            starts += 1
    return starts, rows
