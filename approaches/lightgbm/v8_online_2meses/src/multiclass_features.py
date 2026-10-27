"""One causal feature buffer shared by the binary and eight-class models."""
from __future__ import annotations

import numpy as np

from config import COLUMNAS_SENSORES, MUESTREO_SEGUNDOS, RANGOS_FISICOS
from src.optimized_features import compile_feature_plan, extract_one, warm_start_buffer

EXTRA_FEATURES = (
    "numero_sensores_nan", "numero_sensores_cero", "numero_sensores_validos",
    "duracion_todos_nan", "duracion_alguna_ausencia",
    "numero_variables_fuera_rango", "magnitud_fuera_rango",
)
RULE_PROXIMAL_FEATURES = EXTRA_FEATURES[3:]
LOW = np.asarray([RANGOS_FISICOS[c]["min"] for c in COLUMNAS_SENSORES], np.float32)
HIGH = np.asarray([RANGOS_FISICOS[c]["max"] for c in COLUMNAS_SENSORES], np.float32)
STEP_NS = MUESTREO_SEGUNDOS * 1_000_000_000


def raw_arrays(frame):
    raw = np.ascontiguousarray(frame[COLUMNAS_SENSORES], np.float32)
    times = np.asarray(frame["Fecha"], dtype="datetime64[ns]").astype(np.int64)
    return raw, times, np.isnan(raw).astype(np.uint8)


class SharedFeatureExtractor:
    """The optimized extractor owns the only sensor history; evidence has counters."""

    def __init__(self, names, fills, history, history_times, history_missing,
                 next_timestamp):
        self.names = tuple(names)
        self.fills = np.asarray(fills, np.float32)
        if len(self.names) != len(self.fills) or not np.isfinite(self.fills).all():
            raise ValueError("Invalid feature names or fills")
        self.plan = compile_feature_plan(self.names, self.fills)
        if np.any(self.plan.op == 0):
            raise ValueError("Unknown feature in the frozen binary schema")
        self.state = warm_start_buffer(
            history, np.asarray(history_times).astype("datetime64[ns]"),
            history_missing, next_timestamp=np.datetime64(int(next_timestamp), "ns"))
        self.all_missing_run = 0
        self.any_missing_run = 0
        self.last_timestamp_ns = None
        for raw, timestamp, missing in zip(history, history_times, history_missing):
            self._evidence(raw, int(timestamp), missing)

    def _evidence(self, raw, timestamp_ns, missing):
        raw = np.asarray(raw, np.float32)
        missing = np.asarray(missing, np.uint8)
        if raw.shape != LOW.shape or missing.shape != LOW.shape:
            raise ValueError("Expected nine sensors and missing flags")
        if self.last_timestamp_ns is not None and timestamp_ns <= self.last_timestamp_ns:
            raise ValueError("Timestamps must increase")
        if (self.last_timestamp_ns is not None and
                timestamp_ns - self.last_timestamp_ns != STEP_NS):
            self.all_missing_run = self.any_missing_run = 0
        self.last_timestamp_ns = timestamp_ns
        absent = (missing != 0) | ~np.isfinite(raw)
        valid = ~absent
        zero = valid & (raw == 0)
        count_missing = int(absent.sum())
        self.all_missing_run = min(360, self.all_missing_run + 1) if count_missing == 9 else 0
        self.any_missing_run = min(360, self.any_missing_run + 1) if count_missing else 0
        below = valid & (raw < LOW)
        above = valid & (raw > HIGH)
        outside = below | above
        distance = np.maximum(LOW - raw, raw - HIGH)
        distance[~outside] = 0
        magnitude = float(np.max(distance / np.maximum(HIGH - LOW, 1)))
        return np.asarray((count_missing, int(zero.sum()), int(valid.sum()),
                           self.all_missing_run, self.any_missing_run,
                           int(outside.sum()), magnitude), np.float32)

    def step(self, raw, timestamp_ns, missing):
        evidence = self._evidence(raw, int(timestamp_ns), missing)
        base = extract_one(self.state, self.plan, raw,
                           np.datetime64(int(timestamp_ns), "ns"), missing)
        result = np.concatenate((base, evidence))
        if not np.isfinite(result).all():
            raise ValueError("Non-finite feature vector")
        return result


def stream_matrix(frame, history, names, fills):
    raw, times, missing = raw_arrays(frame)
    old, old_times, old_missing = raw_arrays(history)
    extractor = SharedFeatureExtractor(names, fills, old[-360:], old_times[-360:],
                                       old_missing[-360:], times[0])
    matrix = np.empty((len(raw), len(names) + len(EXTRA_FEATURES)), np.float32)
    for i in range(len(raw)):
        matrix[i] = extractor.step(raw[i], times[i], missing[i])
    return matrix


