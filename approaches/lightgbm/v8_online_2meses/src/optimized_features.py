"""Causal float32/Numba feature engine for sequential batch=1 inference."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(function):
            return function
        return decorate

SENSORS = ("PCO2EXT", "PHEXT", "PRAD", "PRGINT", "PTEXT", "PVV",
           "XCO2I", "XHINV", "XTINV")
SENSOR_INDEX = {name: i for i, name in enumerate(SENSORS)}
MAX_HISTORY = 360
SCHEMA_VERSION = "optimized-fe-v4-causal-missing-protected-ratio"
ROLLING_WINDOWS = np.asarray((10, 20, 30, 60, 120, 360), np.int32)
OP_CONST, OP_RAW, OP_ISNA = 0, 1, 2
OP_SIN_HOUR, OP_COS_HOUR, OP_SIN_DOW, OP_COS_DOW = 3, 4, 5, 6
OP_SIN_MONTH, OP_COS_MONTH, OP_IS_DAY = 7, 8, 9
OP_DIFF, OP_ABS_DIFF, OP_LAG = 10, 11, 12
OP_STD_CURRENT, OP_STUCK_CURRENT = 13, 14
OP_PAST_STD, OP_PAST_RANGE, OP_ZERO_DIFF_PAST = 15, 16, 17
OP_RESIDUAL_PAST, OP_MEAN_CURRENT, OP_ZSCORE_CURRENT = 18, 19, 20
OP_DIFF_ABS_PAIR, OP_DIFF_REL_PAIR = 21, 22
OP_COV_PAST, OP_CORR_PAST, OP_RATIO, OP_DELTA = 23, 24, 25, 26

@dataclass(frozen=True)
class FeaturePlan:
    names: tuple[str, ...]
    op: np.ndarray
    a: np.ndarray
    b: np.ndarray
    window: np.ndarray
    fill: np.ndarray
    @property
    def size(self) -> int:
        return len(self.names)

@dataclass
class BufferState:
    values: np.ndarray
    head: np.int32
    count: np.int32
    last_valid: np.ndarray
    rolling_sums: np.ndarray
    rolling_sum_squares: np.ndarray
    rolling_counts: np.ndarray
    @classmethod
    def empty(cls) -> "BufferState":
        return cls(np.zeros((MAX_HISTORY, len(SENSORS)), np.float32),
                   np.int32(0), np.int32(0),
                   np.full(len(SENSORS), np.nan, np.float32),
                   np.zeros((len(ROLLING_WINDOWS), len(SENSORS)), np.float64),
                   np.zeros((len(ROLLING_WINDOWS), len(SENSORS)), np.float64),
                   np.zeros(len(ROLLING_WINDOWS), np.int32))

def compile_feature_plan(feature_names: Sequence[str],
                         fill_values: Sequence[float] | None = None) -> FeaturePlan:
    """Compile names once; unknown/static columns use their TRAIN median."""
    names = tuple(map(str, feature_names))
    fill = np.zeros(len(names), np.float32)
    if fill_values is not None:
        supplied = np.asarray(fill_values, np.float32)
        if supplied.shape != fill.shape:
            raise ValueError("fill_values must contain one value per feature")
        fill[:] = supplied
    op = np.full(len(names), OP_CONST, np.int16)
    aa = np.full(len(names), -1, np.int16)
    bb = np.full(len(names), -1, np.int16)
    windows = np.zeros(len(names), np.int16)
    time_ops = {"sin_hour": OP_SIN_HOUR, "cos_hour": OP_COS_HOUR,
                "sin_dow": OP_SIN_DOW, "cos_dow": OP_COS_DOW,
                "sin_month": OP_SIN_MONTH, "cos_month": OP_COS_MONTH,
                "is_daytime": OP_IS_DAY}
    aliases = {
        "ratio_rad": (OP_RATIO, "PRGINT", "PRAD", 0),
        "delta_temp": (OP_DELTA, "XTINV", "PTEXT", 0),
        "delta_hum": (OP_DELTA, "XHINV", "PHEXT", 0),
        "gradiente_co2": (OP_DELTA, "XCO2I", "PCO2EXT", 0),
        "rolling_corr_20": (OP_CORR_PAST, "PRAD", "PRGINT", 20),
        "rolling_corr_rad_20": (OP_CORR_PAST, "PRAD", "PRGINT", 20),
        "rolling_corr_temp_20": (OP_CORR_PAST, "XTINV", "PTEXT", 20),
        "rolling_corr_hum_20": (OP_CORR_PAST, "XHINV", "PHEXT", 20),
        "rolling_corr_co2_20": (OP_CORR_PAST, "XCO2I", "PCO2EXT", 20)}
    suffixes = {
        "isna": (OP_ISNA, 0), "diff1": (OP_DIFF, 1),
        "abs_diff1": (OP_ABS_DIFF, 1), "diff3": (OP_DIFF, 3),
        "diff10": (OP_DIFF, 10), "diff30": (OP_DIFF, 30),
        "lag1": (OP_LAG, 1), "lag30": (OP_LAG, 30),
        "rstd_5m": (OP_STD_CURRENT, 10), "rstd_15m": (OP_STD_CURRENT, 30),
        "stuck_flag_5m": (OP_STUCK_CURRENT, 10),
        "stuck_flag_15m": (OP_STUCK_CURRENT, 30),
        "rolling_std_20": (OP_PAST_STD, 20),
        "rolling_range_20": (OP_PAST_RANGE, 20),
        "zero_diff_count_20": (OP_ZERO_DIFF_PAST, 20),
        "diurnal_residual": (OP_RESIDUAL_PAST, 120),
        "rmean_30m": (OP_MEAN_CURRENT, 60), "rstd_30m": (OP_STD_CURRENT, 60),
        "rmean_3h": (OP_MEAN_CURRENT, 360), "rstd_3h": (OP_STD_CURRENT, 360),
        "zscore_local": (OP_ZSCORE_CURRENT, 60)}
    pair_prefixes = (("diff_abs_", OP_DIFF_ABS_PAIR, 0),
                     ("diff_rel_", OP_DIFF_REL_PAIR, 0),
                     ("cov_roll10_", OP_COV_PAST, 10),
                     ("corr_roll10_", OP_CORR_PAST, 10),
                     ("cov_roll20_", OP_COV_PAST, 20),
                     ("corr_roll20_", OP_CORR_PAST, 20))
    pairs = (("PRAD", "PRGINT"), ("XTINV", "PTEXT"),
             ("XHINV", "PHEXT"), ("XCO2I", "PCO2EXT"))
    for i, name in enumerate(names):
        if name in SENSOR_INDEX:
            op[i], aa[i] = OP_RAW, SENSOR_INDEX[name]
        elif name in time_ops:
            op[i] = time_ops[name]
        elif name in aliases:
            code, left, right, win = aliases[name]
            op[i], aa[i], bb[i], windows[i] = code, SENSOR_INDEX[left], SENSOR_INDEX[right], win
        else:
            matched = False
            for sensor, sensor_index in SENSOR_INDEX.items():
                prefix = sensor + "_"
                if name.startswith(prefix) and name[len(prefix):] in suffixes:
                    code, win = suffixes[name[len(prefix):]]
                    op[i], aa[i], windows[i] = code, sensor_index, win
                    matched = True
                    break
            if not matched:
                for left, right in pairs:
                    for prefix, code, win in pair_prefixes:
                        if name == f"{prefix}{left}_{right}":
                            op[i], aa[i], bb[i], windows[i] = (
                                code, SENSOR_INDEX[left], SENSOR_INDEX[right], win)
                            matched = True
                            break
                    if matched:
                        break
    return FeaturePlan(names, op, aa, bb, windows, fill)

@njit(cache=True)
def _ring_get(values, head, count, lag, sensor):
    if count <= 0:
        return np.float32(0.0)
    lag = min(lag, count)
    return values[(head - lag) % values.shape[0], sensor]

@njit(cache=True)
def _window_index(window):
    for i in range(ROLLING_WINDOWS.shape[0]):
        if ROLLING_WINDOWS[i] == window:
            return i
    return -1

@njit(cache=True)
def _past_mean_std(values, head, count, rolling_sums, rolling_sum_squares,
                   rolling_counts, sensor, window):
    window_index = _window_index(window)
    n = min(count, window)
    if n <= 0:
        return np.float32(0.0), np.float32(0.0)
    if window_index >= 0:
        total = rolling_sums[window_index, sensor]
        total2 = rolling_sum_squares[window_index, sensor]
        n = rolling_counts[window_index]
    else:
        total = total2 = 0.0
        for lag in range(1, n + 1):
            value = _ring_get(values, head, count, lag, sensor)
            total += value
            total2 += value * value
    mean = total / n
    if n < 2:
        return np.float32(mean), np.float32(0.0)
    variance = max(0.0, (total2 - total * total / n) / (n - 1))
    return np.float32(mean), np.float32(np.sqrt(variance))

@njit(cache=True)
def _current_mean_std(values, head, count, rolling_sums, rolling_sum_squares,
                      rolling_counts, current, sensor, window):
    n_past = min(count, max(0, window - 1))
    window_index = _window_index(window)
    if window_index >= 0:
        total = rolling_sums[window_index, sensor]
        total2 = rolling_sum_squares[window_index, sensor]
        if count >= window:
            oldest = _ring_get(values, head, count, window, sensor)
            total -= oldest
            total2 -= oldest*oldest
    else:
        total = total2 = 0.0
        for lag in range(1, n_past + 1):
            value = _ring_get(values, head, count, lag, sensor)
            total += value
            total2 += value * value
    total += float(current[sensor])
    total2 += float(current[sensor])*float(current[sensor])
    n = n_past + 1
    mean = total / n
    if n < 2:
        return np.float32(mean), np.float32(0.0)
    variance = max(0.0, (total2 - total * total / n) / (n - 1))
    return np.float32(mean), np.float32(np.sqrt(variance))

@njit(cache=True)
def _past_cov_corr(values, head, count, left, right, window):
    n = min(count, window)
    if n < 2:
        return np.float32(0.0), np.float32(0.0)
    sx = sy = sxx = syy = sxy = 0.0
    for lag in range(1, n + 1):
        x = float(_ring_get(values, head, count, lag, left))
        y = float(_ring_get(values, head, count, lag, right))
        sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y
    cov = (sxy - sx*sy/n) / (n-1)
    vx = (sxx - sx*sx/n) / (n-1)
    vy = (syy - sy*sy/n) / (n-1)
    corr = 0.0 if vx <= 0.0 or vy <= 0.0 else cov / np.sqrt(vx*vy)
    return np.float32(cov), np.float32(corr)

@njit(cache=True)
def _extract_and_append(values, head, count, last_valid, rolling_sums,
                        rolling_sum_squares, rolling_counts, raw, missing,
                        seconds_day, day_of_week, month_zero,
                        op, aa, bb, windows, fills, out):
    current = np.empty(raw.shape[0], dtype=np.float32)
    for sensor in range(raw.shape[0]):
        is_missing = missing[sensor] != 0 or not np.isfinite(raw[sensor])
        if is_missing:
            old = last_valid[sensor]
            current[sensor] = np.float32(0.0) if np.isnan(old) else old
        else:
            current[sensor] = raw[sensor]
            last_valid[sensor] = raw[sensor]
    for i in range(op.shape[0]):
        code, left, right, win = op[i], aa[i], bb[i], windows[i]
        value = fills[i]
        if code == OP_RAW:
            value = current[left]
        elif code == OP_ISNA:
            value = np.float32(missing[left] != 0 or np.isnan(raw[left]))
        elif code == OP_SIN_HOUR:
            value = np.float32(np.sin(2*np.pi*seconds_day/86400.0))
        elif code == OP_COS_HOUR:
            value = np.float32(np.cos(2*np.pi*seconds_day/86400.0))
        elif code == OP_SIN_DOW:
            value = np.float32(np.sin(2*np.pi*day_of_week/7.0))
        elif code == OP_COS_DOW:
            value = np.float32(np.cos(2*np.pi*day_of_week/7.0))
        elif code == OP_SIN_MONTH:
            value = np.float32(np.sin(2*np.pi*month_zero/12.0))
        elif code == OP_COS_MONTH:
            value = np.float32(np.cos(2*np.pi*month_zero/12.0))
        elif code == OP_IS_DAY:
            hour = seconds_day // 3600
            value = np.float32(current[2] > 10 or current[3] > 10 or
                               (hour >= 7 and hour <= 19))
        elif code == OP_DIFF:
            value = current[left] - _ring_get(values, head, count, win, left)
        elif code == OP_ABS_DIFF:
            value = abs(current[left] - _ring_get(values, head, count, win, left))
        elif code == OP_LAG:
            value = _ring_get(values, head, count, win, left)
        elif code in (OP_STD_CURRENT, OP_STUCK_CURRENT,
                       OP_MEAN_CURRENT, OP_ZSCORE_CURRENT):
            mean, std = _current_mean_std(
                values, head, count, rolling_sums, rolling_sum_squares,
                rolling_counts, current, left, win)
            if code == OP_STD_CURRENT:
                value = std
            elif code == OP_STUCK_CURRENT:
                value = np.float32(std < 1e-6)
            elif code == OP_MEAN_CURRENT:
                value = mean
            else:
                value = (current[left]-mean)/(std+1e-6)
        elif code == OP_PAST_STD:
            _, value = _past_mean_std(
                values, head, count, rolling_sums, rolling_sum_squares,
                rolling_counts, left, win)
        elif code == OP_PAST_RANGE:
            n = min(count, win)
            if n >= 5:
                lo, hi = np.float32(np.inf), np.float32(-np.inf)
                for lag in range(1, n+1):
                    x = _ring_get(values, head, count, lag, left)
                    lo, hi = min(lo, x), max(hi, x)
                value = hi-lo
            else:
                value = np.float32(0.0)
        elif code == OP_ZERO_DIFF_PAST:
            n = min(max(0, count-1), win)
            value = np.float32(0.0)
            if count >= 5:
                for lag in range(1, n+1):
                    x0 = _ring_get(values, head, count, lag, left)
                    x1 = _ring_get(values, head, count, lag+1, left)
                    if abs(x0-x1) < 1e-4:
                        value += np.float32(1.0)
        elif code == OP_RESIDUAL_PAST:
            mean, _ = _past_mean_std(
                values, head, count, rolling_sums, rolling_sum_squares,
                rolling_counts, left, win)
            value = current[left]-mean if count >= 10 else np.float32(0.0)
        elif code == OP_DIFF_ABS_PAIR:
            value = abs(current[left]-current[right])
        elif code == OP_DIFF_REL_PAIR:
            value = abs(current[left]-current[right]) / (
                abs(current[left])+abs(current[right])+1.0)
        elif code in (OP_COV_PAST, OP_CORR_PAST):
            cov, corr = _past_cov_corr(values, head, count, left, right, win)
            value = cov if code == OP_COV_PAST else corr
        elif code == OP_RATIO:
            denominator = current[right]+np.float32(1.0)
            # Explicit invalid-division policy: 0.0, never NaN or infinity.
            value = (np.float32(0.0) if not np.isfinite(denominator) or
                     abs(denominator) <= 1e-6 else current[left]/denominator)
        elif code == OP_DELTA:
            value = current[left]-current[right]
        if np.isnan(value) or np.isinf(value):
            value = fills[i]
        out[i] = np.float32(value)
    for window_index in range(ROLLING_WINDOWS.shape[0]):
        window = ROLLING_WINDOWS[window_index]
        full = count >= window
        for sensor in range(current.shape[0]):
            if full:
                old = _ring_get(values, head, count, window, sensor)
                rolling_sums[window_index, sensor] -= old
                rolling_sum_squares[window_index, sensor] -= old*old
            rolling_sums[window_index, sensor] += current[sensor]
            rolling_sum_squares[window_index, sensor] += current[sensor]*current[sensor]
        if not full:
            rolling_counts[window_index] += 1
    values[head, :] = current
    return np.int32((head+1) % values.shape[0]), np.int32(min(count+1, values.shape[0]))

def timestamp_parts(timestamp: np.datetime64) -> tuple[np.int32, np.int32, np.int32]:
    ts = np.datetime64(timestamp, "s")
    day = ts.astype("datetime64[D]")
    seconds = int((ts-day)/np.timedelta64(1, "s"))
    dow = (int(day.astype(np.int64))+3) % 7  # 1970-01-01 was Thursday.
    month = int(str(day)[5:7])-1
    return np.int32(seconds), np.int32(dow), np.int32(month)

def extract_one(state: BufferState, plan: FeaturePlan, raw_values: np.ndarray,
                timestamp: np.datetime64, missing_flags: np.ndarray | None = None,
                out: np.ndarray | None = None) -> np.ndarray:
    """Calculate one row and append it; the compiled kernel allocates no mappings."""
    raw = np.ascontiguousarray(raw_values, np.float32)
    if raw.shape != (len(SENSORS),):
        raise ValueError(f"Expected {len(SENSORS)} sensors in order {SENSORS}")
    missing = (np.isnan(raw).astype(np.uint8) if missing_flags is None else
               np.ascontiguousarray(missing_flags, np.uint8))
    if missing.shape != raw.shape:
        raise ValueError("missing_flags has invalid shape")
    result = np.empty(plan.size, np.float32) if out is None else out
    if result.shape != (plan.size,) or result.dtype != np.float32:
        raise ValueError("out must be float32 with one element per feature")
    seconds, dow, month = timestamp_parts(timestamp)
    state.head, state.count = _extract_and_append(
        state.values, state.head, state.count, state.last_valid,
        state.rolling_sums, state.rolling_sum_squares, state.rolling_counts,
        raw, missing,
        seconds, dow, month, plan.op, plan.a, plan.b, plan.window, plan.fill, result)
    return result

def warm_start_buffer(history: np.ndarray, timestamps: Iterable[np.datetime64],
                      missing_flags: np.ndarray | None = None, *,
                      next_timestamp: np.datetime64,
                      required_history: int = MAX_HISTORY) -> BufferState:
    """Load strict past only. This is causal forward-fill, never pandas bfill."""
    values = np.ascontiguousarray(history, np.float32)
    times = np.asarray(list(timestamps), dtype="datetime64[ns]")
    if values.ndim != 2 or values.shape[1] != len(SENSORS):
        raise ValueError(f"history must have shape [n, {len(SENSORS)}]")
    if len(values) != len(times):
        raise ValueError("history and timestamps must have equal length")
    if len(values) < required_history:
        raise ValueError(f"Insufficient history: {len(values)} < {required_history}")
    if np.any(times >= np.datetime64(next_timestamp, "ns")):
        raise ValueError("Cold start history contains t=0 or future samples")
    if np.any(times[1:] <= times[:-1]):
        raise ValueError("Historical timestamps must be strictly increasing")
    values, times = values[-required_history:], times[-required_history:]
    flags = (np.isnan(values).astype(np.uint8) if missing_flags is None else
             np.ascontiguousarray(missing_flags, np.uint8)[-required_history:])
    if flags.shape != values.shape:
        raise ValueError("missing_flags has invalid shape")
    state, empty, scratch = BufferState.empty(), compile_feature_plan(()), np.empty(0, np.float32)
    for row, timestamp, row_flags in zip(values, times, flags):
        extract_one(state, empty, row, timestamp, row_flags, scratch)
    if np.isnan(state.last_valid).any():
        absent = [SENSORS[i] for i in np.flatnonzero(np.isnan(state.last_valid))]
        raise ValueError(f"No causal historical value for {absent}")
    return state

def ensure_numba_compiled(plan: FeaturePlan) -> None:
    """Trigger JIT before any stable timing region."""
    extract_one(BufferState.empty(), plan, np.zeros(len(SENSORS), np.float32),
                np.datetime64("2020-01-01T00:00:00"))
