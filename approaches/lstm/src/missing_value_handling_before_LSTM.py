# ============================================================
# MISSING VALUE HANDLING
# CIESOL SENSOR ANOMALY DETECTION PROJECT
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SENSOR_COLUMNS = [
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

DERIVED_COLUMNS = [
    "delta_temp",
    "delta_hum",
    "co2_gradient",
]

MODEL_FEATURES = [
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
    "hour_sin",
    "hour_cos",
    "is_night",
    "delta_temp",
    "delta_hum",
    "co2_gradient",
]


# ============================================================
# VALIDATION
# ============================================================

def validate_columns(
    df: pd.DataFrame,
    require_anomaly_columns: bool = True,
) -> None:
    """
    Verify that all columns required by the missing-value
    handling pipeline are present.

    Training:
        require_anomaly_columns=True

    Validation/Test:
        require_anomaly_columns=False
    """

    required_columns = (
        SENSOR_COLUMNS
        + ["FECHA"]
    )

    if require_anomaly_columns:
        required_columns += [
            "tipo_anomalia",
            "fuente_anomalia",
        ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            f"{missing_columns}"
        )


# ============================================================
# TIMESTAMP VALIDATION
# ============================================================

def validate_timestamps(df: pd.DataFrame) -> None:
    """
    Validate timestamp integrity before interpolation.

    Duplicate timestamps are not allowed because time-based
    interpolation relies on a unique time index.
    """

    timestamps = pd.to_datetime(
        df["FECHA"],
        errors="raise",
    )

    if timestamps.isna().any():
        raise ValueError(
            "FECHA contains invalid or missing timestamps."
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "FECHA is not chronologically ordered. "
            "Sort the DataFrame before missing-value handling."
        )

    duplicate_count = int(
        timestamps.duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"FECHA contains {duplicate_count:,} "
            "duplicate timestamps. "
            "Resolve duplicates before missing-value handling."
        )

# ============================================================
# IDENTIFY PROFESSOR MISSING-VALUE ANOMALIES
# ============================================================

def get_missing_anomaly_mask(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Identify rows affected by the professor's
    missing-value anomaly injection method.

    If anomaly-source information is not present
    (validation/test), return an all-False mask.
    """

    if "fuente_anomalia" not in df.columns:
        return pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

    return (
        df["fuente_anomalia"]
        .fillna("")
        .eq("professor_method_missing")
    )


# ============================================================
# CREATE / PRESERVE MISSINGNESS FLAGS
# ============================================================

def create_missingness_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a <sensor>_was_missing flag for every original
    sensor.

    The flag records whether the sensor was NaN before
    imputation.

    Existing flags are preserved using OR logic.
    """

    df = df.copy()

    for sensor in SENSOR_COLUMNS:

        flag_column = f"{sensor}_was_missing"

        current_missing = df[sensor].isna()

        if flag_column in df.columns:

            existing_flag = (
                df[flag_column]
                .fillna(False)
                .astype(bool)
            )

            df[flag_column] = (
                existing_flag
                | current_missing
            )

        else:

            df[flag_column] = current_missing

    return df

def create_any_sensor_missing_flag(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a row-level missingness flag.

    any_sensor_was_missing = 1
        if at least one sensor was missing at that row.

    The flag is calculated from the individual
    <sensor>_was_missing flags, which are created
    before any imputation.
    """

    df = df.copy()

    missing_flag_columns = [
        f"{sensor}_was_missing"
        for sensor in SENSOR_COLUMNS
    ]

    df["any_sensor_was_missing"] = (
        df[missing_flag_columns]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
        .astype(np.int8)
    )

    return df

# ============================================================
# IDENTIFY GAP BOUNDARIES
# ============================================================

def _get_gap_seconds(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate timestamp differences in seconds.
    """

    timestamps = pd.to_datetime(df["FECHA"])

    return timestamps.diff().dt.total_seconds()


def _create_interpolation_groups(
    df: pd.DataFrame,
    max_gap_seconds: float = 60.0,
) -> pd.Series:
    """
    Create groups that prevent interpolation across large
    timestamp gaps.

    A new group starts whenever the distance between two
    consecutive observations is greater than max_gap_seconds.

    This prevents interpolation from crossing known long
    recording gaps.
    """

    gap_seconds = _get_gap_seconds(df)

    new_group = (
        gap_seconds.isna()
        | (gap_seconds > max_gap_seconds)
    )

    return new_group.cumsum()


# ============================================================
# IMPUTE ONE SENSOR
# ============================================================
def _impute_one_sensor(
    df: pd.DataFrame,
    sensor: str,
    missing_anomaly_mask: pd.Series,
    max_interpolation_gap_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Impute professor-injected NaNs for one sensor.

    Rules:
    - Only professor_method_missing NaNs are modified.
    - Interpolation is performed independently inside timestamp groups.
    - Other injected anomaly values are NOT used as interpolation anchors.
    - Consecutive missing blocks are interpolated when valid anchors
      exist on both sides.
    - If a missing value is at a group boundary and therefore cannot
      be time-interpolated, the nearest valid observation within the
      same group is used as a fallback.
    """

    df = df.copy()

    # ------------------------------------------------------------
    # 1. Identify target NaNs
    # ------------------------------------------------------------

    target_mask = (
        missing_anomaly_mask
        & df[sensor].isna()
    )

    if not target_mask.any():
        return df

    # ------------------------------------------------------------
    # 2. Create interpolation groups
    # ------------------------------------------------------------

    groups = _create_interpolation_groups(
        df,
        max_gap_seconds=max_interpolation_gap_seconds,
    )

    # ------------------------------------------------------------
    # 3. Identify rows that must NOT be used as anchors
    # ------------------------------------------------------------

    # Any professor-injected anomaly should not be used as a
    # clean interpolation anchor.
    if "fuente_anomalia" in df.columns:
        anomaly_mask = (
            df["fuente_anomalia"]
            .fillna("")
            .ne("")
        )
    else:
        anomaly_mask = pd.Series(
            False,
            index=df.index,
        )

    # ------------------------------------------------------------
    # 4. Process each interpolation group independently
    # ------------------------------------------------------------

    for group_id in groups.dropna().unique():

        group_mask = groups.eq(group_id)

        group_target_mask = (
            group_mask
            & target_mask
        )

        if not group_target_mask.any():
            continue

        # --------------------------------------------------------
        # Build group
        # --------------------------------------------------------

        group_columns = [
            "FECHA",
            sensor,
        ]

        if "fuente_anomalia" in df.columns:
            group_columns.append("fuente_anomalia")

        group = df.loc[
            group_mask,
            group_columns,
        ].copy()

        group["FECHA"] = pd.to_datetime(
            group["FECHA"]
        )

        group = group.sort_values("FECHA")

        # --------------------------------------------------------
        # Create clean anchor series
        # --------------------------------------------------------

        # Keep original values only if:
        #   - value is not NaN
        #   - row is not an injected anomaly
        #
        # This prevents values such as PHEXT=110 at index 27017
        # from being used as interpolation anchors.
        anchor_values = group[sensor].copy()

        if "fuente_anomalia" in group.columns:
            invalid_anchor_mask = (
                group["fuente_anomalia"]
                .fillna("")
                .ne("")
            )

            anchor_values.loc[invalid_anchor_mask] = np.nan

        anchor_values = anchor_values.astype(float)

        group_indexed = group.set_index("FECHA")

        anchor_series = anchor_values.copy()
        anchor_series.index = group_indexed.index

        # --------------------------------------------------------
        # Time interpolation
        # --------------------------------------------------------

        interpolated = anchor_series.interpolate(
            method="time",
            limit_area="inside",
        )

        # --------------------------------------------------------
        # Write interpolated values back
        # --------------------------------------------------------

        target_indices = df.index[
            group_target_mask
        ]

        target_timestamps = pd.to_datetime(
            df.loc[
                target_indices,
                "FECHA",
            ]
        )

        values = (
            interpolated
            .reindex(target_timestamps)
            .to_numpy()
        )

        valid = ~np.isnan(values)

        valid_indices = target_indices[valid]

        df.loc[
            valid_indices,
            sensor,
        ] = values[valid]

        # --------------------------------------------------------
        # 5. Boundary fallback
        # --------------------------------------------------------
        #
        # If interpolation could not fill a target because it is
        # at the beginning/end of the interpolation group, use the
        # nearest clean anchor from the SAME group.
        #
        # This handles cases such as index 28799.

        remaining_indices = target_indices[
            ~valid
        ]

        if len(remaining_indices) == 0:
            continue

        clean_anchor_mask = (
            anchor_series.notna()
        )

        clean_anchor_times = (
            anchor_series.index[
                clean_anchor_mask
            ]
        )

        if len(clean_anchor_times) == 0:
            continue

        clean_anchor_values = (
            anchor_series.loc[
                clean_anchor_times
            ]
        )

        for target_idx in remaining_indices:

            target_time = pd.to_datetime(
                df.loc[target_idx, "FECHA"]
            )

            distances = np.abs(
                clean_anchor_times - target_time
            )

            nearest_position = distances.argmin()

            nearest_time = (
                clean_anchor_times[
                    nearest_position
                ]
            )

            nearest_value = (
                clean_anchor_values.iloc[
                    nearest_position
                ]
            )

            df.loc[
                target_idx,
                sensor,
            ] = nearest_value

    return df


# ============================================================
# IMPUTE ALL ORIGINAL SENSORS
# ============================================================

def impute_sensor_nans(
    df: pd.DataFrame,
    max_interpolation_gap_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Impute only sensor NaNs generated by the professor's
    missing-value anomaly method.

    Unexpected NaNs are NOT silently imputed.
    """

    df = df.copy()

    missing_anomaly_mask = get_missing_anomaly_mask(
        df
    )

    for sensor in SENSOR_COLUMNS:

        df = _impute_one_sensor(
            df=df,
            sensor=sensor,
            missing_anomaly_mask=missing_anomaly_mask,
            max_interpolation_gap_seconds=(
                max_interpolation_gap_seconds
            ),
        )

    return df


# ============================================================
# RECALCULATE DERIVED FEATURES
# ============================================================

def recompute_derived_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recalculate the three dependent engineered features.

    Exact project formulas:

        delta_temp   = XTINV - PTEXT
        delta_hum    = XHINV - PHEXT
        co2_gradient = XCO2I - PCO2EXT
    """

    df = df.copy()

    # --------------------------------------------------------
    # Internal vs external temperature
    # --------------------------------------------------------

    df["delta_temp"] = (
        df["XTINV"]
        - df["PTEXT"]
    )

    # --------------------------------------------------------
    # Internal vs external humidity
    # --------------------------------------------------------

    df["delta_hum"] = (
        df["XHINV"]
        - df["PHEXT"]
    )

    # --------------------------------------------------------
    # Internal vs external CO2
    # --------------------------------------------------------

    df["co2_gradient"] = (
        df["XCO2I"]
        - df["PCO2EXT"]
    )

    return df


# ============================================================
# FIND UNEXPECTED SENSOR NaNs
# ============================================================
def find_unfilled_injected_nans(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find professor-injected NaNs that remain after the
    interpolation step.
    """

    missing_anomaly_mask = get_missing_anomaly_mask(df)

    rows = []

    for sensor in SENSOR_COLUMNS:

        unfilled_mask = (
            missing_anomaly_mask
            & df[sensor].isna()
        )

        count = int(
            unfilled_mask.sum()
        )

        if count > 0:
            rows.append(
                {
                    "sensor": sensor,
                    "unfilled_injected_nan_rows": count,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "sensor",
                "unfilled_injected_nan_rows",
            ]
        )

    return pd.DataFrame(rows)

def find_non_injected_sensor_nans(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find sensor NaNs that are NOT associated with the
    professor's missing-value anomaly injection.

    For validation/test, where anomaly-source information
    does not exist, all sensor NaNs are considered
    non-injected.
    """

    missing_anomaly_mask = get_missing_anomaly_mask(df)

    rows = []

    for sensor in SENSOR_COLUMNS:

        nan_mask = df[sensor].isna()

        non_injected_mask = (
            nan_mask
            & ~missing_anomaly_mask
        )

        count = int(
            non_injected_mask.sum()
        )

        if count > 0:
            rows.append(
                {
                    "sensor": sensor,
                    "non_injected_nan_rows": count,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "sensor",
                "non_injected_nan_rows",
            ]
        )

    return pd.DataFrame(rows)


# ============================================================
# FINAL NaN VALIDATION
# ============================================================

def validate_final_nan_state(
    df: pd.DataFrame,
) -> dict:
    """
    Validate the final state after missing-value handling.
    """

    report = {}

    # --------------------------------------------------------
    # Sensor NaNs
    # --------------------------------------------------------

    sensor_nan_counts = (
        df[SENSOR_COLUMNS]
        .isna()
        .sum()
    )

    report["sensor_nan_counts"] = (
        sensor_nan_counts[
            sensor_nan_counts > 0
        ].to_dict()
    )

    report["sensor_nan_cells"] = int(
        sensor_nan_counts.sum()
    )

    # --------------------------------------------------------
    # Derived feature NaNs
    # --------------------------------------------------------

    derived_nan_counts = (
        df[DERIVED_COLUMNS]
        .isna()
        .sum()
    )

    report["derived_nan_counts"] = (
        derived_nan_counts[
            derived_nan_counts > 0
        ].to_dict()
    )

    report["derived_nan_cells"] = int(
        derived_nan_counts.sum()
    )

    # --------------------------------------------------------
    # Model feature NaNs
    # --------------------------------------------------------

    model_nan_counts = (
        df[MODEL_FEATURES]
        .isna()
        .sum()
    )

    report["model_nan_counts"] = (
        model_nan_counts[
            model_nan_counts > 0
        ].to_dict()
    )

    report["model_nan_cells"] = int(
        model_nan_counts.sum()
    )

    # --------------------------------------------------------
    # Missing-value anomaly rows
    # --------------------------------------------------------

    report["missing_anomaly_rows"] = int(
        get_missing_anomaly_mask(df).sum()
    )

    return report


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def handle_missing_values(
    df: pd.DataFrame,
    max_interpolation_gap_seconds: float = 60.0,
    raise_on_unexpected_nan: bool = True,
    injected_anomalies: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Complete missing-value handling pipeline.

    Training:
        injected_anomalies=True
        - Professor missing-value anomalies are present.
        - Injected sensor NaNs are imputed.
        - Anomaly labels/sources are preserved.

    Validation/Test:
        injected_anomalies=False
        - No professor anomaly injection.
        - No anomaly labels/sources are required.
        - No artificial imputation is performed.

    In all datasets:
        - Per-sensor missingness flags are created.
        - any_sensor_was_missing is created.
        - Derived features are recalculated.
        - Final NaN state is validated.
    """

    # --------------------------------------------------------
    # 1. Validate columns
    # --------------------------------------------------------

    validate_columns(
        df,
        require_anomaly_columns=injected_anomalies,
    )

    # --------------------------------------------------------
    # 2. Validate timestamps
    # --------------------------------------------------------

    validate_timestamps(df)

    # --------------------------------------------------------
    # Keep original state
    # --------------------------------------------------------

    df_before = df.copy()

    # --------------------------------------------------------
    # 3. Create / preserve missingness flags
    # --------------------------------------------------------

    df_clean = create_missingness_flags(
        df
    )

    # --------------------------------------------------------
    # 4. Create row-level missingness feature
    # --------------------------------------------------------

    df_clean = create_any_sensor_missing_flag(
        df_clean
    )

    # --------------------------------------------------------
    # 5. Handle professor-injected missing-value anomalies
    # --------------------------------------------------------

    if injected_anomalies:

        df_clean = impute_sensor_nans(
            df_clean,
            max_interpolation_gap_seconds=(
                max_interpolation_gap_seconds
            ),
        )

    # --------------------------------------------------------
    # 6. Recalculate dependent features
    # --------------------------------------------------------

    df_clean = recompute_derived_features(
        df_clean
    )

    # --------------------------------------------------------
    # 7. Validate final state
    # --------------------------------------------------------

    report = validate_final_nan_state(
        df_clean
    )

    # --------------------------------------------------------
    # 8. Basic information
    # --------------------------------------------------------

    report["rows_before"] = len(df_before)
    report["rows_after"] = len(df_clean)

    # --------------------------------------------------------
    # 9. Preserve anomaly labels/sources
    # --------------------------------------------------------

    if injected_anomalies:

        report["labels_preserved"] = bool(
            df_before["tipo_anomalia"].equals(
                df_clean["tipo_anomalia"]
            )
        )

        report["sources_preserved"] = bool(
            df_before["fuente_anomalia"].equals(
                df_clean["fuente_anomalia"]
            )
        )

    else:

        report["labels_preserved"] = None
        report["sources_preserved"] = None

    # --------------------------------------------------------
    # 10. Professor-injected anomaly information
    # --------------------------------------------------------

    if injected_anomalies:

        report["injected_missing_rows_before"] = int(
            get_missing_anomaly_mask(
                df_before
            ).sum()
        )

        report["injected_missing_rows_after"] = int(
            get_missing_anomaly_mask(
                df_clean
            ).sum()
        )

        # ----------------------------------------------------
        # Check injected NaNs that could not be imputed
        # ----------------------------------------------------

        unfilled_injected_nans = (
            find_unfilled_injected_nans(
                df_clean
            )
        )

        report["unfilled_injected_nans"] = (
            unfilled_injected_nans
        )

    else:

        report["injected_missing_rows_before"] = 0
        report["injected_missing_rows_after"] = 0

        report["unfilled_injected_nans"] = (
            pd.DataFrame(
                columns=[
                    "sensor",
                    "unfilled_injected_nan_rows",
                ]
            )
        )

    # --------------------------------------------------------
    # 11. Check non-injected sensor NaNs
    # --------------------------------------------------------

    non_injected_sensor_nans = (
        find_non_injected_sensor_nans(
            df_clean
        )
    )

    report["non_injected_sensor_nans"] = (
        non_injected_sensor_nans
    )

    # --------------------------------------------------------
    # 12. Stop if model features still contain NaNs
    # --------------------------------------------------------

    if (
        raise_on_unexpected_nan
        and report["model_nan_cells"] > 0
    ):

        print("\n" + "=" * 70)
        print(
            "ERROR: NaNs remain after missing-value handling"
        )
        print("=" * 70)

        print(
            "\nRemaining NaNs in model features:"
        )

        print(
            pd.Series(
                report["model_nan_counts"]
            )
        )

        print(
            "\nNon-injected sensor NaNs:"
        )

        if len(non_injected_sensor_nans) > 0:
            print(
                non_injected_sensor_nans
            )
        else:
            print("None")

        if injected_anomalies:

            print(
                "\nUnfilled injected NaNs:"
            )

            if len(unfilled_injected_nans) > 0:
                print(
                    unfilled_injected_nans
                )
            else:
                print("None")

        raise ValueError(
            "\nNaNs remain in the 18 model features "
            "after missing-value handling. "
            "Investigate the reported columns "
            "before scaling."
        )

    return df_clean, report

def check_sampling_interval(df, name):

    timestamps = pd.to_datetime(
        df["FECHA"]
    ).sort_values()

    delta_seconds = (
        timestamps
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    print("\n" + "=" * 70)
    print(f"{name} — SAMPLING INTERVAL CHECK")
    print("=" * 70)

    print(
        f"\nMedian interval: "
        f"{delta_seconds.median():.2f} s"
    )

    print(
        f"Mean interval:   "
        f"{delta_seconds.mean():.2f} s"
    )

    print(
        f"Minimum:         "
        f"{delta_seconds.min():.2f} s"
    )

    print(
        f"Maximum:         "
        f"{delta_seconds.max():.2f} s"
    )

    print("\nInterval distribution:")

    print(
        delta_seconds
        .round(1)
        .value_counts()
        .sort_index()
        .head(20)
    )
