"""
Agroconnect data preprocessing — preserve real timestamps,

.

Starts from ``data/processed/clean_data.parquet`` (never overwritten).

Design principles
-----------------
* Dominant inter-arrival is ~30–31 s → normal sampling jitter, not missing data.
* Do NOT force the whole series onto a wall-clock 30 s grid.
* Reconstruct a 30 s grid only inside detected missing-record gaps.
* Never overwrite originally observed sensor values.
* No synthetic anomalies, no LSTM training here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import COLUMN_PHYSICAL_RANGES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP_CANDIDATES = ("FECHA", "Fecha", "fecha")

SENSOR_COLUMNS: list[str] = [
    "PCO2EXT",
    "PHEXT",
    "PRAD",
    "PRGINT",
    "PTEXT",
    "PVV",
    "UVENT_cen",
    "UVENT_lN",
    "XCO2I",
    "XHINV",
    "XTINV",
    "XTS",
]

DEFAULT_CLEAN_PATH = Path("data/processed/clean_data.parquet")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_QUALITY_DIR = Path("data/quality")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    """Configurable thresholds — nothing gap-related is hard-coded as magic."""

    # Nominal sampling period
    expected_seconds: float = 30.0

    # Inclusive band treated as normal jitter (Agroconnect: >95% of intervals in 30–31 s).
    # Intervals ABOVE ``normal_interval_max`` become gap candidates.
    # Do not set this so wide that ~60 s looks "normal".
    normal_interval_min: float = 29.0
    normal_interval_max: float = 32.0

    # Gap categories (minutes). Real Agroconnect holes sit near ~5m11s / ~5m13s,
    # so keep the short/medium boundary configurable.
    short_gap_minutes: float = 5.5
    medium_gap_minutes: float = 10.0

    # Chronological split (no shuffle, no overlap)
    train_days: int = 60
    validation_days: int = 10
    test_days: int = 10

    # Paths
    clean_path: str | Path = field(default_factory=lambda: DEFAULT_CLEAN_PATH)
    output_dir: str | Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    quality_dir: str | Path = field(default_factory=lambda: DEFAULT_QUALITY_DIR)

    @property
    def short_gap_seconds(self) -> float:
        return self.short_gap_minutes * 60.0

    @property
    def medium_gap_seconds(self) -> float:
        return self.medium_gap_minutes * 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_timestamp_column(df: pd.DataFrame) -> str:
    for name in TIMESTAMP_CANDIDATES:
        if name in df.columns:
            return name
    raise ValueError(
        f"Timestamp column not found. Expected one of {TIMESTAMP_CANDIDATES}. "
        f"Got: {list(df.columns)}"
    )


def _normalize_timestamp_name(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the timestamp column to FECHA for a stable internal API."""
    col = _resolve_timestamp_column(df)
    if col != "FECHA":
        df = df.rename(columns={col: "FECHA"})
    return df


def _sensor_columns_present(df: pd.DataFrame) -> list[str]:
    return [c for c in SENSOR_COLUMNS if c in df.columns]


def _classify_gap_duration(
    duration_seconds: float,
    cfg: PreprocessConfig,
) -> str:
    """
    Classify an inter-arrival / gap duration.

    Categories:
      - normal_jitter : within configured normal band (29–32 s, not a real gap)
      - timing_jitter : >32–60 s (mild timing jitter / delayed observation, 0 missing samples)
      - short_gap     : >60 s and ≤ short_gap_minutes (genuine hole, reconstructed & interpolated)
      - medium_gap    : > short_gap_minutes and ≤ medium_gap_minutes (reconstructed, kept as NaN)
      - long_gap      : > medium_gap_minutes (reconstructed, kept as NaN)
    """
    if duration_seconds <= cfg.normal_interval_max:
        return "normal_jitter"
    if duration_seconds <= 60.0:
        return "timing_jitter"
    if duration_seconds <= cfg.short_gap_seconds:
        return "short_gap"
    if duration_seconds <= cfg.medium_gap_seconds:
        return "medium_gap"
    return "long_gap"


def _estimate_missing_samples(
    duration_seconds: float,
    expected_seconds: float = 30.0,
) -> int:
    """
    Estimate how many samples are missing between two observations.

    Intervals <= 60 s are timing jitter / delayed observations (0 missing samples).
    Intervals > 60 s have missing samples (e.g. ~313 s -> 10 missing).
    """
    if duration_seconds <= 60.0 or expected_seconds <= 0:
        return 0
    return max(0, int(np.ceil(duration_seconds / expected_seconds)) - 1)




def _ensure_flag_columns(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Ensure provenance / gap flag columns exist with safe defaults."""
    out = df.copy()
    if "timestamp_was_missing" not in out.columns:
        out["timestamp_was_missing"] = False
    if "_original_observation" not in out.columns:
        out["_original_observation"] = ~out["timestamp_was_missing"]
    for col in sensors:
        flag = f"{col}_was_missing"
        if flag not in out.columns:
            out[flag] = out[col].isna()
        interp = f"{col}_interpolated"
        if interp not in out.columns:
            out[interp] = False
    for col in (
        "gap_category",
        "in_short_gap",
        "in_medium_gap",
        "in_long_gap",
        "sequence_unsafe",
    ):
        if col not in out.columns:
            if col == "gap_category":
                out[col] = "none"
            else:
                out[col] = False
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load clean data
# ─────────────────────────────────────────────────────────────────────────────

def load_clean_data(
    path: str | Path = DEFAULT_CLEAN_PATH,
) -> pd.DataFrame:
    """
    Load the cleaned Agroconnect parquet and validate schema / dtypes.

    Does not overwrite the source file. Extra columns are preserved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {path}")

    df = pd.read_parquet(path)
    df = _normalize_timestamp_name(df)

    missing_sensors = [c for c in SENSOR_COLUMNS if c not in df.columns]
    if missing_sensors:
        raise ValueError(
            f"Missing required sensor columns: {missing_sensors}. "
            f"Available: {list(df.columns)}"
        )

    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    n_bad = int(df["FECHA"].isna().sum())
    if n_bad:
        raise ValueError(f"{n_bad} timestamps could not be converted to datetime.")

    for col in SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("FECHA").reset_index(drop=True)

    if not df["FECHA"].is_monotonic_increasing:
        raise ValueError("Timestamps are not chronologically sortable.")

    extras = [c for c in df.columns if c not in ("FECHA", *SENSOR_COLUMNS)]
    logger.info(
        "Loaded clean data: %s rows × %s cols | %s → %s | extra cols: %s",
        f"{len(df):,}",
        df.shape[1],
        df["FECHA"].min(),
        df["FECHA"].max(),
        extras or "none",
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Timestamp / gap analysis (on ORIGINAL timestamps)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_time_gaps(
    df: pd.DataFrame,
    expected_seconds: float = 30.0,
    normal_interval_min: float = 29.0,
    normal_interval_max: float = 32.0,
    short_gap_minutes: float = 5.5,
    medium_gap_minutes: float = 10.0,
    cfg: Optional[PreprocessConfig] = None,
) -> dict[str, Any]:
    """
    Analyse original inter-arrival times and build a gap report.

    Distinguishes normal 30–31 s jitter from genuine missing-record gaps
    using configurable thresholds (no hard-coded 42 s "error").

    Gap report columns:
      gap_start, gap_end, duration_seconds, estimated_missing_samples, gap_category
    """
    if cfg is None:
        cfg = PreprocessConfig(
            expected_seconds=expected_seconds,
            normal_interval_min=normal_interval_min,
            normal_interval_max=normal_interval_max,
            short_gap_minutes=short_gap_minutes,
            medium_gap_minutes=medium_gap_minutes,
        )

    df = _normalize_timestamp_name(df.copy())
    df = df.sort_values("FECHA").reset_index(drop=True)

    deltas = df["FECHA"].diff().dt.total_seconds()
    deltas_valid = deltas.iloc[1:]

    percentiles = {
        f"p{p}": float(np.nanpercentile(deltas_valid, p))
        for p in (1, 5, 25, 50, 75, 95, 99)
    }

    # Interval distribution (rounded to nearest second for readability)
    rounded = deltas_valid.round(0)
    interval_distribution = (
        rounded.value_counts()
        .sort_index()
        .rename_axis("interval_seconds")
        .reset_index(name="count")
    )
    interval_distribution["percentage"] = (
        interval_distribution["count"] / interval_distribution["count"].sum() * 100
    ).round(4)

    in_30_31 = ((deltas_valid >= 30.0) & (deltas_valid <= 31.0)).sum()
    pct_30_31 = float(in_30_31 / len(deltas_valid) * 100) if len(deltas_valid) else 0.0
    in_normal = (
        (deltas_valid >= cfg.normal_interval_min)
        & (deltas_valid <= cfg.normal_interval_max)
    ).sum()
    pct_normal_band = (
        float(in_normal / len(deltas_valid) * 100) if len(deltas_valid) else 0.0
    )

    # Gap candidates: intervals strictly above the normal jitter band
    gap_mask = deltas > cfg.normal_interval_max
    gap_rows: list[dict[str, Any]] = []
    for idx in df.index[gap_mask]:
        prev_idx = idx - 1
        if prev_idx < 0:
            continue
        duration = float(deltas.loc[idx])
        gap_start = df.loc[prev_idx, "FECHA"]
        gap_end = df.loc[idx, "FECHA"]
        category = _classify_gap_duration(duration, cfg)
        gap_rows.append(
            {
                "gap_start": gap_start,
                "gap_end": gap_end,
                "duration_seconds": duration,
                "duration_minutes": duration / 60.0,
                "estimated_missing_samples": _estimate_missing_samples(
                    duration, cfg.expected_seconds
                ),
                "gap_category": category,
                "prev_idx": int(prev_idx),
                "next_idx": int(idx),
            }
        )

    gap_report = pd.DataFrame(gap_rows)
    if gap_report.empty:
        gap_report = pd.DataFrame(
            columns=[
                "gap_start",
                "gap_end",
                "duration_seconds",
                "duration_minutes",
                "estimated_missing_samples",
                "gap_category",
                "prev_idx",
                "next_idx",
            ]
        )

    category_counts = (
        gap_report["gap_category"].value_counts().to_dict()
        if len(gap_report)
        else {}
    )

    report: dict[str, Any] = {
        "n_intervals": int(len(deltas_valid)),
        "min_interval": float(deltas_valid.min()),
        "max_interval": float(deltas_valid.max()),
        "mean_interval": float(deltas_valid.mean()),
        "median_interval": float(deltas_valid.median()),
        "std_interval": float(deltas_valid.std()),
        "percentiles": percentiles,
        "normal_interval_min": cfg.normal_interval_min,
        "normal_interval_max": cfg.normal_interval_max,
        "expected_seconds": cfg.expected_seconds,
        "pct_intervals_30_31": pct_30_31,
        "pct_intervals_in_normal_band": pct_normal_band,
        "n_gap_candidates": int(len(gap_report)),
        "gap_category_counts": category_counts,
        "interval_distribution": interval_distribution,
        "gap_report": gap_report,
    }

    logger.info(
        "Time-gap analysis: median=%.2fs mean=%.2fs | "
        "30–31s=%.2f%% | normal band [%g,%g]=%.2f%% | "
        "gap candidates=%d | categories=%s",
        report["median_interval"],
        report["mean_interval"],
        pct_30_31,
        cfg.normal_interval_min,
        cfg.normal_interval_max,
        pct_normal_band,
        report["n_gap_candidates"],
        category_counts,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reconstruct 30 s grid ONLY inside detected gaps
# ─────────────────────────────────────────────────────────────────────────────

def create_gap_grid(
    df: pd.DataFrame,
    gap_report: pd.DataFrame,
    expected_seconds: float = 30.0,
    categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Insert reconstructed timestamps (NaN sensors) only for selected gaps.

    Original observations are preserved unchanged. New rows have:
      - timestamp_was_missing = True
      - sensor values = NaN
      - ``{sensor}_was_missing`` = True

    By default reconstructs short / medium / long gaps (not normal_jitter).
    """
    categories = categories or ["short_gap", "medium_gap", "long_gap"]
    df = _normalize_timestamp_name(df.copy())
    df = df.sort_values("FECHA").reset_index(drop=True)
    sensors = _sensor_columns_present(df)

    out = _ensure_flag_columns(df, sensors)
    out["_original_observation"] = True
    out["timestamp_was_missing"] = False
    for col in sensors:
        # Situation A: NaN on an existing timestamp
        out[f"{col}_was_missing"] = out[col].isna()

    if gap_report is None or gap_report.empty:
        logger.info("create_gap_grid: no gaps to reconstruct.")
        return out.reset_index(drop=True)

    selected = gap_report[gap_report["gap_category"].isin(categories)]
    if selected.empty:
        logger.info(
            "create_gap_grid: no gaps in categories %s.", categories
        )
        return out.reset_index(drop=True)

    step = pd.Timedelta(seconds=expected_seconds)
    new_rows: list[dict[str, Any]] = []

    for _, g in selected.iterrows():
        gap_start = pd.Timestamp(g["gap_start"])
        gap_end = pd.Timestamp(g["gap_end"])
        duration = float(g["duration_seconds"]) if "duration_seconds" in g else (
            (gap_end - gap_start).total_seconds()
        )
        n_missing = int(g["estimated_missing_samples"]) if (
            "estimated_missing_samples" in g
            and pd.notna(g["estimated_missing_samples"])
        ) else _estimate_missing_samples(duration, expected_seconds)

        # Mild delays (e.g. 33–42 s) → 0 missing samples: report only,
        # do not invent a timestamp that would sit ~10 s from the next obs.
        if n_missing <= 0:
            continue

        # Place n_missing points at +expected, +2*expected, ... from gap_start,
        # always strictly before gap_end.
        for k in range(1, n_missing + 1):
            t = gap_start + k * step
            if t >= gap_end - pd.Timedelta(milliseconds=1):
                break
            row: dict[str, Any] = {
                "FECHA": t,
                "timestamp_was_missing": True,
                "_original_observation": False,
                "gap_category": g["gap_category"],
                "in_short_gap": g["gap_category"] == "short_gap",
                "in_medium_gap": g["gap_category"] == "medium_gap",
                "in_long_gap": g["gap_category"] == "long_gap",
                "sequence_unsafe": g["gap_category"] in ("medium_gap", "long_gap"),
            }
            for col in sensors:
                row[col] = np.nan
                row[f"{col}_was_missing"] = True
                row[f"{col}_interpolated"] = False
            new_rows.append(row)

    if not new_rows:
        logger.info("create_gap_grid: estimated missing slots produced 0 new rows.")
        return out.reset_index(drop=True)

    new_df = pd.DataFrame(new_rows)
    # Align columns
    for col in out.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan if col in sensors else False
    for col in new_df.columns:
        if col not in out.columns:
            out[col] = False if col.endswith(
                ("_was_missing", "_interpolated", "_observation")
            ) or col.startswith("in_") or col in (
                "timestamp_was_missing",
                "sequence_unsafe",
            ) else ("none" if col == "gap_category" else np.nan)

    combined = pd.concat([out, new_df[out.columns]], ignore_index=True)
    combined = combined.sort_values("FECHA").reset_index(drop=True)

    # Annotate original rows that bound gaps (optional clarity)
    # Mark reconstructed rows' neighbours already set via gap_category on new rows.

    logger.info(
        "create_gap_grid: inserted %d reconstructed timestamps "
        "across %d gap episode(s); originals preserved=%d.",
        len(new_rows),
        len(selected),
        int(combined["_original_observation"].sum()),
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Optional experimental full-series regularization (NOT default)
# ─────────────────────────────────────────────────────────────────────────────

def experimental_resample_mean(
    df: pd.DataFrame,
    freq: str = "30s",
) -> pd.DataFrame:
    """
    OPTIONAL / EXPERIMENTAL — professor reference: ``resample("30s").mean()``.

    Do NOT use blindly on the full Agroconnect series: normal 30–31 s jitter would
    be distorted and many artificial NaNs created. Kept for comparison only.
    """
    df = _normalize_timestamp_name(df.copy())
    sensors = _sensor_columns_present(df)
    timed = df.set_index("FECHA").sort_index()
    resampled = timed[sensors].resample(freq).mean()
    logger.warning(
        "experimental_resample_mean(%s): %d → %d rows. "
        "Not the default Agroconnect data pipeline.",
        freq,
        len(df),
        len(resampled),
    )
    return resampled.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Short-gap interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_short_gaps(
    df: pd.DataFrame,
    gap_report: pd.DataFrame,
    sensor_columns: Optional[list[str]] = None,
    cfg: Optional[PreprocessConfig] = None,
) -> pd.DataFrame:
    """
    Interpolate sensor values only inside reconstructed short gaps.

    Original timestamps and original sensor measurements are never modified.

    Short-gap rows are identified using:
        gap_start < FECHA < gap_end
        AND timestamp_was_missing == True

    Interpolation is performed separately for each gap and each sensor,
    using the two original observations surrounding the gap.
    """

    out = _normalize_timestamp_name(df.copy())
    out["FECHA"] = pd.to_datetime(out["FECHA"])
    out = out.sort_values("FECHA").reset_index(drop=True)

    sensors = sensor_columns or _sensor_columns_present(out)
    out = _ensure_flag_columns(out, sensors)

    if gap_report is None or gap_report.empty:
        return out

    short_gaps = gap_report[
        gap_report["gap_category"] == "short_gap"
    ]

    for _, gap in short_gaps.iterrows():

        gap_start = pd.Timestamp(gap["gap_start"])
        gap_end = pd.Timestamp(gap["gap_end"])

        # Only reconstructed timestamps INSIDE this gap
        gap_mask = (
            (out["FECHA"] > gap_start)
            & (out["FECHA"] < gap_end)
            & out["timestamp_was_missing"].fillna(False)
        )

        if not gap_mask.any():
            continue

        # Find the original observations immediately surrounding the gap
        before_mask = (
            (out["FECHA"] <= gap_start)
            & (~out["timestamp_was_missing"].fillna(False))
        )

        after_mask = (
            (out["FECHA"] >= gap_end)
            & (~out["timestamp_was_missing"].fillna(False))
        )

        before_candidates = out.loc[before_mask]
        after_candidates = out.loc[after_mask]

        if before_candidates.empty or after_candidates.empty:
            logger.warning(
                "Cannot interpolate gap %s → %s: "
                "missing boundary observation.",
                gap_start,
                gap_end,
            )
            continue

        before_idx = before_candidates.index[-1]
        after_idx = after_candidates.index[0]

        t0 = out.loc[before_idx, "FECHA"]
        t1 = out.loc[after_idx, "FECHA"]

        total_seconds = (t1 - t0).total_seconds()

        if total_seconds <= 0:
            continue

        # Interpolate each sensor independently
        for col in sensors:

            y0 = out.loc[before_idx, col]
            y1 = out.loc[after_idx, col]

            # If either boundary value is missing, don't invent a value.
            if pd.isna(y0) or pd.isna(y1):
                continue

            target_indices = out.index[gap_mask]

            for idx in target_indices:

                t = out.loc[idx, "FECHA"]
                fraction = (t - t0).total_seconds() / total_seconds

                value = y0 + fraction * (y1 - y0)

                # Only fill reconstructed missing values
                if pd.isna(out.loc[idx, col]):
                    out.loc[idx, col] = value

                    out.loc[idx, f"{col}_was_missing"] = True
                    out.loc[idx, f"{col}_interpolated"] = True

        # Mark the reconstructed rows
        out.loc[gap_mask, "gap_category"] = "short_gap"
        out.loc[gap_mask, "in_short_gap"] = True

    return out

# ─────────────────────────────────────────────────────────────────────────────
# 5. Medium / long gaps
# ─────────────────────────────────────────────────────────────────────────────

def handle_medium_gaps(
    df: pd.DataFrame,
    gap_report: pd.DataFrame,
    sensor_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Medium gaps (> short threshold, <= medium threshold):
    - process ONLY "medium_gap"
    - locate gaps using gap_start, gap_end, and timestamp_was_missing
    - medium gaps remain NaN and are NOT interpolated
    - mark them with in_medium_gap=True and sequence_unsafe=True
    """
    out = _normalize_timestamp_name(df.copy())
    sensors = sensor_columns or _sensor_columns_present(out)
    out = _ensure_flag_columns(out, sensors)

    if gap_report is None or gap_report.empty:
        logger.info("handle_medium_gaps: no medium gaps.")
        return out

    medium = gap_report[gap_report["gap_category"] == "medium_gap"]

    if medium.empty:
        logger.info("handle_medium_gaps: no medium gaps.")
        return out

    for _, g in medium.iterrows():
        gap_start = pd.Timestamp(g["gap_start"])
        gap_end = pd.Timestamp(g["gap_end"])

        gap_mask = (
            (out["FECHA"] > gap_start)
            & (out["FECHA"] < gap_end)
            & out["timestamp_was_missing"].fillna(False)
        )

        out.loc[gap_mask, sensors] = np.nan
        out.loc[gap_mask, "in_medium_gap"] = True
        out.loc[gap_mask, "gap_category"] = "medium_gap"
        out.loc[gap_mask, "sequence_unsafe"] = True

    logger.info(
        "handle_medium_gaps: preserved %d medium gap episode(s) without interpolation.",
        len(medium),
    )
    return out


def handle_long_gaps(
    df: pd.DataFrame,
    gap_report: pd.DataFrame,
    sensor_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Long gaps (> medium threshold):
    - process ONLY "long_gap"
    - locate gaps using gap_start, gap_end, and timestamp_was_missing
    - long gaps remain NaN and are NOT interpolated
    - mark them with in_long_gap=True and sequence_unsafe=True
    """
    out = _normalize_timestamp_name(df.copy())
    sensors = sensor_columns or _sensor_columns_present(out)
    out = _ensure_flag_columns(out, sensors)

    if gap_report is None or gap_report.empty:
        logger.info("handle_long_gaps: no long gaps.")
        return out

    long_gaps = gap_report[gap_report["gap_category"] == "long_gap"]

    if long_gaps.empty:
        logger.info("handle_long_gaps: no long gaps.")
        return out

    for _, g in long_gaps.iterrows():
        gap_start = pd.Timestamp(g["gap_start"])
        gap_end = pd.Timestamp(g["gap_end"])

        gap_mask = (
            (out["FECHA"] > gap_start)
            & (out["FECHA"] < gap_end)
            & out["timestamp_was_missing"].fillna(False)
        )

        out.loc[gap_mask, sensors] = np.nan
        out.loc[gap_mask, "in_long_gap"] = True
        out.loc[gap_mask, "gap_category"] = "long_gap"
        out.loc[gap_mask, "sequence_unsafe"] = True

    logger.info(
        "handle_long_gaps: preserved %d long gap episode(s) without interpolation.",
        len(long_gaps),
    )
    return out


def update_sequence_safety(
    df: pd.DataFrame,
    sensor_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Finalise ``sequence_unsafe`` for a later sequence builder.

    Unsafe if the row is in a medium/long gap OR any sensor is still NaN.
    """
    out = df.copy()
    sensors = sensor_cols or _sensor_columns_present(out)
    unresolved = out[sensors].isna().any(axis=1)
    medium = out["in_medium_gap"] if "in_medium_gap" in out.columns else False
    long = out["in_long_gap"] if "in_long_gap" in out.columns else False
    out["sequence_unsafe"] = medium | long | unresolved
    out["has_unresolved_nan"] = unresolved
    logger.info(
        "Sequence-unsafe rows: %d / %d (%.3f%%)",
        int(out["sequence_unsafe"].sum()),
        len(out),
        100.0 * float(out["sequence_unsafe"].mean()),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Missing-value analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distinguish:

    1. Sensor NaN on an existing (original) timestamp.
    2. Timestamp was missing and had to be reconstructed.

    Returns per-sensor counts and percentages.
    """
    df = _normalize_timestamp_name(df.copy())
    sensors = _sensor_columns_present(df)
    n = len(df)

    ts_missing = (
        df["timestamp_was_missing"].astype(bool)
        if "timestamp_was_missing" in df.columns
        else pd.Series(False, index=df.index)
    )

    rows = []
    for col in sensors:
        nan_mask = df[col].isna()
        flag = f"{col}_was_missing"
        was_missing = (
            df[flag].astype(bool) if flag in df.columns else nan_mask
        )
        # Situation A: original timestamp present, value missing
        sit_a = (~ts_missing) & was_missing
        # Situation B: reconstructed timestamp (inherently missing at insert)
        sit_b = ts_missing & was_missing
        rows.append(
            {
                "sensor": col,
                "null_count": int(nan_mask.sum()),
                "null_pct": round(float(nan_mask.mean() * 100), 4),
                "nan_on_existing_timestamp": int(sit_a.sum()),
                "nan_on_reconstructed_timestamp": int(sit_b.sum()),
                "was_missing_flag_count": int(was_missing.sum()),
                "still_nan_after_processing": int(nan_mask.sum()),
                "n_rows": n,
            }
        )

    report = pd.DataFrame(rows)
    logger.info(
        "Missing-value analysis: total reconstructed timestamps=%d | "
        "sensors with any NaN=%d",
        int(ts_missing.sum()),
        int((report["null_count"] > 0).sum()),
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 7. Physical ranges (flag only — never delete / repair)
# ─────────────────────────────────────────────────────────────────────────────

def check_physical_ranges(
    df: pd.DataFrame,
    physical_ranges: Optional[dict[str, tuple[float, float]]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag out-of-range values using known Agroconnect sensor limits.

    Returns ``(df_with_flags, summary_report)``.
    Does NOT delete or repair values (anomaly-detection project).

    If ``physical_ranges`` is None, reuses ``COLUMN_PHYSICAL_RANGES`` from
    ``data_loader`` (already defined from instrument datasheets). Does not
    invent new limits.
    """
    physical_ranges = physical_ranges or COLUMN_PHYSICAL_RANGES
    out = df.copy()
    rows = []

    for col, (lo, hi) in physical_ranges.items():
        if col not in out.columns:
            continue
        s = out[col]
        below = s < lo
        above = s > hi
        oor = below | above
        flag = f"{col}_out_of_range"
        out[flag] = oor.fillna(False)

        rows.append(
            {
                "sensor": col,
                "min_allowed": lo,
                "max_allowed": hi,
                "data_min": float(s.min()) if s.notna().any() else np.nan,
                "data_max": float(s.max()) if s.notna().any() else np.nan,
                "below_range": int(below.fillna(False).sum()),
                "above_range": int(above.fillna(False).sum()),
                "out_of_range": int(oor.fillna(False).sum()),
                "out_of_range_pct": round(
                    float(oor.fillna(False).mean() * 100), 4
                ),
            }
        )

    summary = pd.DataFrame(rows)
    total = int(summary["out_of_range"].sum()) if len(summary) else 0
    logger.info(
        "Physical-range check: %d out-of-range values across %d sensors "
        "(flagged only, not repaired).",
        total,
        len(summary),
    )
    return out, summary


# ─────────────────────────────────────────────────────────────────────────────
# 8. Duplicate timestamps
# ─────────────────────────────────────────────────────────────────────────────

def analyze_duplicate_timestamps(
    df: pd.DataFrame,
    remove_identical: bool = False,
) -> dict[str, Any]:
    """
    Analyse duplicate timestamps.

    * identical duplicates (same timestamp + same sensor values) → optional remove
    * conflicting duplicates (same timestamp, different values) → report only;
      never silently pick one.
    """
    df = _normalize_timestamp_name(df.copy())
    sensors = _sensor_columns_present(df)

    dup_mask = df["FECHA"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_timestamps = int(df.loc[dup_mask, "FECHA"].nunique()) if n_dup_rows else 0

    identical_groups = 0
    conflicting_groups = 0
    conflict_records: list[dict[str, Any]] = []

    if n_dup_rows:
        for ts, group in df.loc[dup_mask].groupby("FECHA", sort=False):
            subset = group[sensors]
            # Identical if all rows equal (NaN-aware via fill sentinel)
            if len(subset.drop_duplicates()) == 1:
                identical_groups += 1
            else:
                conflicting_groups += 1
                conflict_records.append(
                    {
                        "FECHA": ts,
                        "n_rows": len(group),
                        "n_unique_value_rows": int(len(subset.drop_duplicates())),
                    }
                )

    cleaned = df
    n_removed = 0
    if remove_identical and identical_groups:
        # Drop exact full-row duplicates; leave conflicting groups untouched
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(
            subset=["FECHA", *sensors], keep="first"
        ).reset_index(drop=True)
        # After dropping identical full duplicates, remaining FECHA dups are conflicts
        n_removed = before - len(cleaned)

    conflicts_df = pd.DataFrame(conflict_records)
    result = {
        "n_duplicate_rows": n_dup_rows,
        "n_duplicate_timestamps": n_dup_timestamps,
        "n_identical_groups": identical_groups,
        "n_conflicting_groups": conflicting_groups,
        "conflicts": conflicts_df,
        "n_identical_rows_removed": n_removed,
        "df": cleaned if remove_identical else df,
    }
    logger.info(
        "Duplicate timestamps: rows=%d timestamps=%d identical_groups=%d "
        "conflicting_groups=%d removed_identical_rows=%d",
        n_dup_rows,
        n_dup_timestamps,
        identical_groups,
        conflicting_groups,
        n_removed,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 9. Chronological split
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_days: int = 60,
    validation_days: int = 10,
    test_days: int = 10,
) -> dict[str, Any]:
    """
    Chronological split — no shuffle, no temporal overlap.

    Default: 60 d train / 10 d validation / 10 d final test.
    The test period must remain untouched until final evaluation.
    """
    df = _normalize_timestamp_name(df.copy())
    df = df.sort_values("FECHA").reset_index(drop=True)

    t0 = df["FECHA"].min()
    train_end = t0 + pd.Timedelta(days=train_days)
    val_end = train_end + pd.Timedelta(days=validation_days)
    test_end = val_end + pd.Timedelta(days=test_days)

    train = df[df["FECHA"] < train_end].copy()
    validation = df[(df["FECHA"] >= train_end) & (df["FECHA"] < val_end)].copy()
    test = df[(df["FECHA"] >= val_end) & (df["FECHA"] < test_end)].copy()

    remainder = df[df["FECHA"] >= test_end]
    if len(remainder):
        logger.warning(
            "Found %d rows after planned test_end (%s); appending to test.",
            len(remainder),
            test_end,
        )
        test = pd.concat([test, remainder], ignore_index=True)

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            f"Empty split detected. train={len(train)} val={len(validation)} "
            f"test={len(test)}. Check day counts vs data span."
        )

    train_start, train_end_ts = train["FECHA"].min(), train["FECHA"].max()
    val_start, val_end_ts = validation["FECHA"].min(), validation["FECHA"].max()
    test_start, test_end_ts = test["FECHA"].min(), test["FECHA"].max()

    assert train_end_ts < val_start, "Train/validation overlap or wrong order."
    assert val_end_ts < test_start, "Validation/test overlap or wrong order."

    meta = {
        "train_start": train_start,
        "train_end": train_end_ts,
        "validation_start": val_start,
        "validation_end": val_end_ts,
        "test_start": test_start,
        "test_end": test_end_ts,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_days": train_days,
        "validation_days": validation_days,
        "test_days": test_days,
    }
    logger.info(
        "Chronological split:\n"
        "  TRAIN      %s → %s  (%d rows)\n"
        "  VALIDATION %s → %s  (%d rows)\n"
        "  TEST       %s → %s  (%d rows)",
        train_start,
        train_end_ts,
        len(train),
        val_start,
        val_end_ts,
        len(validation),
        test_start,
        test_end_ts,
        len(test),
    )
    return {"train": train, "validation": validation, "test": test, "meta": meta}


# ─────────────────────────────────────────────────────────────────────────────
# 10. Leakage-safe scaling / optional imputer (fit on TRAIN only)
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(
    train_df: pd.DataFrame,
    sensor_cols: Optional[list[str]] = None,
) -> StandardScaler:
    """Fit StandardScaler on TRAIN sensor values only."""
    sensors = sensor_cols or _sensor_columns_present(train_df)
    scaler = StandardScaler()
    complete = train_df[sensors].dropna()
    if complete.empty:
        raise ValueError("Cannot fit scaler: no complete train rows.")
    scaler.fit(complete)
    logger.info(
        "Scaler fitted on %d complete train rows × %d sensors.",
        len(complete),
        len(sensors),
    )
    return scaler


def transform_with_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    sensor_cols: Optional[list[str]] = None,
    suffix: str = "_scaled",
) -> pd.DataFrame:
    """Apply a train-fitted scaler (transform only)."""
    if sensor_cols is not None:
        sensors = sensor_cols
    elif hasattr(scaler, "feature_names_in_"):
        sensors = list(scaler.feature_names_in_)
    else:
        sensors = _sensor_columns_present(df)

    out = df.copy()
    values = out[sensors]
    scaled = np.full((len(out), len(sensors)), np.nan, dtype=float)
    complete_mask = values.notna().all(axis=1).to_numpy()
    if complete_mask.any():
        scaled[complete_mask] = scaler.transform(values.loc[complete_mask])
    for i, col in enumerate(sensors):
        out[f"{col}{suffix}"] = scaled[:, i]
    return out


def fit_iterative_imputer_experimental(
    train_df: pd.DataFrame,
    sensor_cols: Optional[list[str]] = None,
    max_iter: int = 10,
    random_state: int = 42,
):
    """
    OPTIONAL — IterativeImputer fitted on TRAIN only (no leakage).

    Not part of the default pipeline. Call ``transform`` separately on
    validation / test.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    sensors = sensor_cols or _sensor_columns_present(train_df)
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputer.fit(train_df[sensors])
    logger.warning(
        "EXPERIMENTAL IterativeImputer fitted on train (%d rows). "
        "Apply transform only to val/test.",
        len(train_df),
    )
    return imputer, sensors


# ─────────────────────────────────────────────────────────────────────────────
# 11. Quality report
# ─────────────────────────────────────────────────────────────────────────────

def generate_quality_report(
    df: pd.DataFrame,
    time_analysis: dict[str, Any],
    gap_report: pd.DataFrame,
    missing_report: pd.DataFrame,
    duplicate_report: dict[str, Any],
    physical_summary: pd.DataFrame,
    split_meta: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Flatten key quality metrics into a single tabular report
    suitable for ``data/quality/data_quality_report.csv``.
    """
    rows: list[dict[str, Any]] = [
        {"section": "dataset", "metric": "n_rows", "value": len(df)},
        {
            "section": "dataset",
            "metric": "n_original_observations",
            "value": int(df["_original_observation"].sum())
            if "_original_observation" in df.columns
            else len(df),
        },
        {
            "section": "dataset",
            "metric": "n_reconstructed_timestamps",
            "value": int(df["timestamp_was_missing"].sum())
            if "timestamp_was_missing" in df.columns
            else 0,
        },
        {
            "section": "intervals",
            "metric": "min_interval_s",
            "value": time_analysis.get("min_interval"),
        },
        {
            "section": "intervals",
            "metric": "median_interval_s",
            "value": time_analysis.get("median_interval"),
        },
        {
            "section": "intervals",
            "metric": "mean_interval_s",
            "value": time_analysis.get("mean_interval"),
        },
        {
            "section": "intervals",
            "metric": "p95_interval_s",
            "value": time_analysis.get("percentiles", {}).get("p95"),
        },
        {
            "section": "intervals",
            "metric": "p99_interval_s",
            "value": time_analysis.get("percentiles", {}).get("p99"),
        },
        {
            "section": "intervals",
            "metric": "max_interval_s",
            "value": time_analysis.get("max_interval"),
        },
        {
            "section": "intervals",
            "metric": "pct_30_31_s",
            "value": time_analysis.get("pct_intervals_30_31"),
        },
        {
            "section": "intervals",
            "metric": "pct_normal_jitter_band",
            "value": time_analysis.get("pct_intervals_in_normal_band"),
        },
        {
            "section": "gaps",
            "metric": "n_gap_candidates",
            "value": time_analysis.get("n_gap_candidates"),
        },
    ]

    for cat, n in (time_analysis.get("gap_category_counts") or {}).items():
        rows.append(
            {"section": "gaps", "metric": f"n_{cat}", "value": n}
        )

    if gap_report is not None and len(gap_report):
        for _, g in gap_report.iterrows():
            rows.append(
                {
                    "section": "gap_episode",
                    "metric": f"{g['gap_category']}_{pd.Timestamp(g['gap_start']).isoformat()}",
                    "value": round(float(g["duration_seconds"]), 3),
                }
            )

    for _, r in missing_report.iterrows():
        rows.append(
            {
                "section": "missing",
                "metric": f"{r['sensor']}_null_count",
                "value": r["null_count"],
            }
        )
        rows.append(
            {
                "section": "missing",
                "metric": f"{r['sensor']}_null_pct",
                "value": r["null_pct"],
            }
        )

    rows.extend(
        [
            {
                "section": "duplicates",
                "metric": "n_duplicate_timestamps",
                "value": duplicate_report.get("n_duplicate_timestamps", 0),
            },
            {
                "section": "duplicates",
                "metric": "n_identical_groups",
                "value": duplicate_report.get("n_identical_groups", 0),
            },
            {
                "section": "duplicates",
                "metric": "n_conflicting_groups",
                "value": duplicate_report.get("n_conflicting_groups", 0),
            },
        ]
    )

    if physical_summary is not None and len(physical_summary):
        for _, r in physical_summary.iterrows():
            rows.append(
                {
                    "section": "physical_range",
                    "metric": f"{r['sensor']}_out_of_range",
                    "value": r["out_of_range"],
                }
            )

    if split_meta:
        for k, v in split_meta.items():
            rows.append(
                {
                    "section": "split",
                    "metric": k,
                    "value": v.isoformat() if isinstance(v, pd.Timestamp) else v,
                }
            )

    report = pd.DataFrame(rows)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 12. Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessed(
    df: pd.DataFrame,
    gap_report: pd.DataFrame,
    quality_report: pd.DataFrame,
    time_analysis: dict[str, Any],
    split_meta: Optional[dict[str, Any]] = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    quality_dir: str | Path = DEFAULT_QUALITY_DIR,
    scaler: Optional[StandardScaler] = None,
) -> dict[str, Path]:
    """
    Persist artefacts. Never overwrites ``clean_data.parquet``.

    Defaults:
      - data/processed/preprocessed_data.parquet
      - data/quality/gap_report.csv
      - data/quality/data_quality_report.csv
    """
    output_dir = Path(output_dir)
    quality_dir = Path(quality_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "preprocessed": output_dir / "preprocessed_data.parquet",
        "gap_report": quality_dir / "gap_report.csv",
        "quality_report": quality_dir / "data_quality_report.csv",
        "time_analysis": quality_dir / "time_gap_analysis.json",
    }

    if paths["preprocessed"].resolve() == Path(DEFAULT_CLEAN_PATH).resolve():
        raise RuntimeError("Refusing to overwrite clean_data.parquet.")

    df.to_parquet(paths["preprocessed"], index=False)

    gap_out = gap_report.copy()
    if len(gap_out):
        for col in ("gap_start", "gap_end"):
            if col in gap_out.columns:
                gap_out[col] = pd.to_datetime(gap_out[col]).astype(str)
    gap_out.to_csv(paths["gap_report"], index=False)
    quality_report.to_csv(paths["quality_report"], index=False)

    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, np.datetime64):
            return pd.Timestamp(obj).isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [_jsonable(v) for v in obj.tolist()]
        if isinstance(obj, pd.Series):
            return [_jsonable(v) for v in obj.tolist()]
        if isinstance(obj, pd.DataFrame):
            rec = obj.copy()
            for c in rec.columns:
                if pd.api.types.is_datetime64_any_dtype(rec[c]):
                    rec[c] = rec[c].astype(str)
            return rec.to_dict(orient="records")
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        return obj

    time_json = {
        k: _jsonable(v)
        for k, v in time_analysis.items()
        if k not in ("interval_distribution", "gap_report")
    }
    if split_meta is not None:
        paths["split_meta"] = quality_dir / "split_meta.json"
        paths["split_meta"].write_text(json.dumps(_jsonable(split_meta), indent=2))
    paths["time_analysis"].write_text(json.dumps(time_json, indent=2))

    # Also store interval distribution for inspection
    if "interval_distribution" in time_analysis:
        dist_path = quality_dir / "interval_distribution.csv"
        time_analysis["interval_distribution"].to_csv(dist_path, index=False)
        paths["interval_distribution"] = dist_path

    if scaler is not None:
        import joblib

        paths["scaler"] = output_dir / "scaler.joblib"
        joblib.dump(scaler, paths["scaler"])

    logger.info("Saved preprocessing artefacts to %s and %s", output_dir, quality_dir)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 13. Orchestration (modular steps — not a monolithic black box)
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    cfg: Optional[PreprocessConfig] = None,
    use_experimental_medium_imputer: bool = False,
) -> dict[str, Any]:
    """
    Run the modular preprocessing steps in order.

    Gap-handling policy
    -------------------
    * normal jitter (≤ ``normal_interval_max``, default 32 s): keep as-is
    * short gaps (≤ ``short_gap_minutes``): reconstruct 30 s slots + interpolate
    * medium gaps: reconstruct slots, keep NaN, report for investigation
    * long gaps: reconstruct slots, keep NaN, mark sequence_unsafe
    """
    cfg = cfg or PreprocessConfig()

    # 1. Load (never overwrite clean source)
    df = load_clean_data(cfg.clean_path)

    # 2. Duplicates on originals
    dup = analyze_duplicate_timestamps(df, remove_identical=True)
    df = dup["df"]

    # 3. Interval / gap analysis on ORIGINAL timestamps
    time_analysis = analyze_time_gaps(df, cfg=cfg)
    gap_report = time_analysis["gap_report"].copy()

    # 4. Reconstruct missing timestamps only inside detected gaps
    df = create_gap_grid(
        df,
        gap_report,
        expected_seconds=cfg.expected_seconds,
        categories=["short_gap", "medium_gap", "long_gap"],
    )

    # 5. Short-gap interpolation only
    df = interpolate_short_gaps(df, gap_report=gap_report, cfg=cfg)

    # 6. Medium / long: preserve NaNs, mark unsafe
    df = handle_medium_gaps(df, gap_report)
    df = handle_long_gaps(df, gap_report)
    df = update_sequence_safety(df)

    # 7. Missing-value report (Situation A vs B)
    missing_report = analyze_missing_values(df)

    # 8. Physical ranges — flag only
    df, physical_summary = check_physical_ranges(df)

    # 9. Quality report + persist (Scaling and splitting performed later in 05_data_splitting.ipynb)
    quality_report = generate_quality_report(
        df=df,
        time_analysis=time_analysis,
        gap_report=gap_report,
        missing_report=missing_report,
        duplicate_report=dup,
        physical_summary=physical_summary,
    )
    paths = save_preprocessed(
        df=df,
        gap_report=gap_report,
        quality_report=quality_report,
        time_analysis=time_analysis,
        output_dir=cfg.output_dir,
        quality_dir=cfg.quality_dir,
    )

    return {
        "df": df,
        "gap_report": gap_report,
        "missing_report": missing_report,
        "physical_summary": physical_summary,
        "duplicate_report": dup,
        "quality_report": quality_report,
        "time_analysis": time_analysis,
        "paths": paths,
        "config": cfg,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_preprocessing_pipeline()
    print("\n=== Interval summary ===")
    ta = result["time_analysis"]
    print(
        f"median={ta['median_interval']:.2f}s  mean={ta['mean_interval']:.2f}s  "
        f"p95={ta['percentiles']['p95']:.2f}s  p99={ta['percentiles']['p99']:.2f}s  "
        f"max={ta['max_interval']:.2f}s"
    )
    print(f"% intervals in 30–31 s: {ta['pct_intervals_30_31']:.3f}%")
    print(f"Gap category counts: {ta['gap_category_counts']}")
    print("\n=== Gap report ===")
    print(result["gap_report"].to_string(index=False))
    print("\n=== Outputs ===")
    for k, p in result["paths"].items():
        print(f"  {k}: {p}")
