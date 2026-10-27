"""
LSTM sequence creation for the CIESOL sensor anomaly detection project.

The module converts scaled, feature-engineered sensor data into
sequence-to-one LSTM samples.

Important design decisions:
- Uses the 18 model features.
- Expects scaling to have already been performed.
- Does NOT fit or modify a scaler.
- Does NOT delete rows because of sequence_unsafe.
- Prevents sequences from crossing unsafe boundaries.
- Allows normal 30-31 second sampling jitter.
- Does not treat every timestamp difference from exactly 30 seconds
  as a sequence break.
- Preserves the anomaly label of the target timestep.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ============================================================
# MODEL FEATURES
# ============================================================

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

MISSINGNESS_FEATURE = "any_sensor_was_missing"

SCALED_FEATURES = [
    f"{feature}_scaled"
    for feature in MODEL_FEATURES
]

# Final LSTM input = 18 scaled features + 1 binary feature
LSTM_FEATURES = (
    SCALED_FEATURES
    + [MISSINGNESS_FEATURE]
)


# ============================================================
# VALIDATION
# ============================================================

# ============================================================
# VALIDATION
# ============================================================

def validate_sequence_input(
    df: pd.DataFrame,
    scaled_features: list[str] | None = None,
    require_anomaly_labels: bool = False,
) -> None:
    """
    Validate a dataframe before LSTM sequence creation.

    Parameters
    ----------
    df:
        Scaled dataframe.

    scaled_features:
        Scaled continuous/engineered model features.

    require_anomaly_labels:
        True for training data, where anomaly injection has
        been applied and tipo_anomalia is available.

        False for validation/test data, where anomaly injection
        has not been applied.
    """

    if scaled_features is None:
        scaled_features = SCALED_FEATURES

    # --------------------------------------------------------
    # Required columns common to ALL datasets
    # --------------------------------------------------------

    required_columns = [
        "FECHA",
        *scaled_features,
        MISSINGNESS_FEATURE,
    ]

    # --------------------------------------------------------
    # Anomaly labels are required ONLY for training
    # --------------------------------------------------------

    if require_anomaly_labels:
        required_columns.append("tipo_anomalia")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns for sequence creation: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        df["FECHA"],
        errors="raise",
    )

    if timestamps.isna().any():
        raise ValueError(
            "FECHA contains missing timestamps."
        )

    if timestamps.duplicated().any():
        duplicate_count = int(
            timestamps.duplicated().sum()
        )

        raise ValueError(
            f"FECHA contains {duplicate_count:,} "
            "duplicate timestamps."
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "FECHA is not chronologically ordered."
        )

    # --------------------------------------------------------
    # LSTM input features
    # --------------------------------------------------------

    input_features = (
        list(scaled_features)
        + [MISSINGNESS_FEATURE]
    )

    # --------------------------------------------------------
    # LSTM input must not contain NaNs
    # --------------------------------------------------------

    nan_counts = (
        df[input_features]
        .isna()
        .sum()
    )

    nan_counts = nan_counts[
        nan_counts > 0
    ]

    if len(nan_counts) > 0:
        raise ValueError(
            "NaNs remain in LSTM input features.\n"
            f"{nan_counts.to_dict()}\n\n"
            "Run missing-value handling before "
            "sequence creation."
        )

    # --------------------------------------------------------
    # Validate binary missingness feature
    # --------------------------------------------------------

    unique_values = set(
        df[MISSINGNESS_FEATURE]
        .unique()
    )

    allowed_values = {0, 1}

    if not unique_values.issubset(
        allowed_values
    ):
        raise ValueError(
            f"'{MISSINGNESS_FEATURE}' must contain "
            f"only 0/1 values. "
            f"Found: {sorted(unique_values)}"
        )

# ============================================================
# BINARY TARGET
# ============================================================

def create_binary_target(
    df: pd.DataFrame,
    label_column: str = "tipo_anomalia",
) -> pd.Series:
    """
    Create a binary anomaly target.

    0 = normal
    1 = anomaly

    The project uses NaN in tipo_anomalia for normal observations.
    Any non-empty/non-null anomaly label is considered anomalous.
    """

    labels = df[label_column]

    y_binary = (
        labels.notna()
        & labels.astype(str).str.strip().ne("")
        & labels.astype(str).str.lower().ne("none")
    ).astype(np.int8)

    return y_binary


# ============================================================
# SAFE BOUNDARIES
# ============================================================

# ============================================================
# SAFE BOUNDARIES
# ============================================================

def create_safe_boundary_mask(
    df: pd.DataFrame,
    max_allowed_gap_seconds: float = 60.0,
) -> pd.Series:
    """
    Identify rows that can be used as sequence boundaries.

    A sequence must not cross:

    1. sequence_unsafe regions
    2. timestamp gaps greater than max_allowed_gap_seconds

    Important:
    - 30-31 second jitter is allowed.
    - 32-60 second timing jitter is also allowed.
    - >60 seconds is treated as a temporal boundary.

    If sequence_unsafe is not present, all rows are assumed
    to be sequence-safe.

    Returns
    -------
    pd.Series
        True where the row is unsafe as part of a sequence.
    """

    timestamps = pd.to_datetime(
        df["FECHA"]
    )

    # --------------------------------------------------------
    # Timestamp differences
    # --------------------------------------------------------

    time_diff = (
        timestamps.diff()
        .dt.total_seconds()
    )

    # --------------------------------------------------------
    # Sequence-unsafe metadata
    # --------------------------------------------------------

    if "sequence_unsafe" in df.columns:
        unsafe = (
            df["sequence_unsafe"]
            .fillna(False)
            .astype(bool)
        )
    else:
        # Validation/test may not have this metadata.
        unsafe = pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

    # --------------------------------------------------------
    # Large timestamp gaps
    # --------------------------------------------------------

    large_timestamp_gap = (
        time_diff > max_allowed_gap_seconds
    )

    # --------------------------------------------------------
    # Combine boundaries
    # --------------------------------------------------------

    boundary_mask = (
        unsafe
        | large_timestamp_gap
    )

    # --------------------------------------------------------
    # First row has no previous observation.
    # It is safe to use as the beginning of a sequence.
    # --------------------------------------------------------

    boundary_mask.iloc[0] = False

    return boundary_mask


# ============================================================
# SEQUENCE CREATION
# ============================================================

# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_lstm_sequences(
    df: pd.DataFrame,
    sequence_length: int = 20,
    scaled_features: list[str] | None = None,
    label_column: str = "tipo_anomalia",
    max_allowed_gap_seconds: float = 60.0,
    create_targets: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame, dict]:
    """
    Create sequence-to-one LSTM sequences.

    Parameters
    ----------
    df:
        Scaled dataframe.

    sequence_length:
        Number of timesteps in each sequence.

    scaled_features:
        Scaled continuous/engineered model features.

    label_column:
        Anomaly label column. Required only when
        create_targets=True.

    max_allowed_gap_seconds:
        Maximum timestamp gap allowed inside a sequence.

    create_targets:
        True for training data, where anomaly labels exist.

        False for validation/test data, where anomaly
        injection has not been applied.

    Returns
    -------
    X:
        LSTM input sequences with shape:

        (n_sequences, sequence_length, 19)

        19 = 18 scaled features + any_sensor_was_missing

    y:
        Binary anomaly targets when create_targets=True.
        None otherwise.

    metadata:
        Metadata for every generated sequence.

    report:
        Sequence creation statistics.
    """

    # ========================================================
    # Feature configuration
    # ========================================================

    if scaled_features is None:
        scaled_features = SCALED_FEATURES

    input_features = (
        list(scaled_features)
        + [MISSINGNESS_FEATURE]
    )

    # ========================================================
    # Validate sequence length
    # ========================================================

    if sequence_length < 2:
        raise ValueError(
            "sequence_length must be at least 2."
        )

    # ========================================================
    # Validate input
    # ========================================================

    validate_sequence_input(
        df,
        scaled_features=scaled_features,
        require_anomaly_labels=create_targets,
    )

    # ========================================================
    # Work on a copy
    # ========================================================

    df = df.copy()

    df["FECHA"] = pd.to_datetime(
        df["FECHA"]
    )

    # ========================================================
    # Create binary targets ONLY for training
    # ========================================================

    if create_targets:

        df["_target"] = create_binary_target(
            df,
            label_column=label_column,
        )

    # ========================================================
    # Sequence boundary detection
    # ========================================================

    boundary_mask = create_safe_boundary_mask(
        df,
        max_allowed_gap_seconds=max_allowed_gap_seconds,
    )

    df["_sequence_boundary"] = boundary_mask

    # ========================================================
    # Convert ALL 19 LSTM features to numpy
    # ========================================================

    feature_values = (
        df[input_features]
        .to_numpy(dtype=np.float32)
    )

    timestamps = (
        df["FECHA"]
        .to_numpy()
    )

    # ========================================================
    # Containers
    # ========================================================

    X_sequences = []
    y_sequences = []
    metadata_rows = []

    rejected_boundary_windows = 0

    # ========================================================
    # Total possible windows
    # ========================================================

    total_possible_windows = max(
        0,
        len(df) - sequence_length + 1,
    )

    # ========================================================
    # Create sliding windows
    # ========================================================

    for end_idx in range(
        sequence_length - 1,
        len(df),
    ):

        start_idx = (
            end_idx
            - sequence_length
            + 1
        )

        # ----------------------------------------------------
        # Do not allow sequence across unsafe boundary
        # ----------------------------------------------------

        window_boundaries = (
            boundary_mask.iloc[
                start_idx + 1 : end_idx + 1
            ]
        )

        if window_boundaries.any():

            rejected_boundary_windows += 1

            continue

        # ----------------------------------------------------
        # Create X
        # ----------------------------------------------------

        X = feature_values[
            start_idx : end_idx + 1
        ]

        X_sequences.append(X)

        # ----------------------------------------------------
        # Create y ONLY when labels exist
        # ----------------------------------------------------

        if create_targets:

            y_sequences.append(
                df["_target"].iloc[end_idx]
            )

        # ----------------------------------------------------
        # Sequence metadata
        # ----------------------------------------------------

        metadata_row = {
            "sequence_start": timestamps[start_idx],
            "sequence_end": timestamps[end_idx],
            "target_timestamp": timestamps[end_idx],
            "target_row_index": df.index[end_idx],
        }

        # ----------------------------------------------------
        # Training-only anomaly information
        # ----------------------------------------------------

        if create_targets:

            metadata_row[
                "target_anomaly_label"
            ] = df.iloc[end_idx][label_column]

            metadata_row[
                "target_is_anomaly"
            ] = int(
                df["_target"].iloc[end_idx]
            )

            if "fuente_anomalia" in df.columns:

                metadata_row[
                    "target_anomaly_source"
                ] = df.iloc[
                    end_idx
                ]["fuente_anomalia"]

        # ----------------------------------------------------
        # Quality / gap metadata
        # ----------------------------------------------------

        if "gap_category" in df.columns:

            metadata_row[
                "target_gap_category"
            ] = df.iloc[
                end_idx
            ]["gap_category"]

        if "in_short_gap" in df.columns:

            metadata_row[
                "target_in_short_gap"
            ] = df.iloc[
                end_idx
            ]["in_short_gap"]

        if "in_medium_gap" in df.columns:

            metadata_row[
                "target_in_medium_gap"
            ] = df.iloc[
                end_idx
            ]["in_medium_gap"]

        if "in_long_gap" in df.columns:

            metadata_row[
                "target_in_long_gap"
            ] = df.iloc[
                end_idx
            ]["in_long_gap"]

        if "sequence_unsafe" in df.columns:

            metadata_row[
                "target_sequence_unsafe"
            ] = df.iloc[
                end_idx
            ]["sequence_unsafe"]

        metadata_rows.append(
            metadata_row
        )

    # ========================================================
    # Convert X
    # ========================================================

    if X_sequences:

        X_array = np.stack(
            X_sequences
        ).astype(np.float32)

    else:

        X_array = np.empty(
            (
                0,
                sequence_length,
                len(input_features),
            ),
            dtype=np.float32,
        )

    # ========================================================
    # Convert y
    # ========================================================

    if create_targets:

        y_array = np.asarray(
            y_sequences,
            dtype=np.int8,
        )

    else:

        y_array = None

    # ========================================================
    # Metadata
    # ========================================================

    metadata = pd.DataFrame(
        metadata_rows
    )

    # ========================================================
    # Report
    # ========================================================

    if create_targets and len(y_array) > 0:

        anomaly_count = int(
            np.sum(y_array == 1)
        )

        normal_count = int(
            np.sum(y_array == 0)
        )

    else:

        anomaly_count = None
        normal_count = None

    report = {
        "input_rows": len(df),
        "sequence_length": sequence_length,
        "number_features": len(input_features),
        "total_possible_windows": total_possible_windows,
        "created_sequences": len(X_array),
        "rejected_boundary_windows": (
            rejected_boundary_windows
        ),
        "safe_sequence_percentage": (
            100.0 * len(X_array)
            / total_possible_windows
            if total_possible_windows > 0
            else 0.0
        ),
        "anomaly_target_sequences": anomaly_count,
        "normal_target_sequences": normal_count,
        "has_targets": create_targets,
    }

    return (
        X_array,
        y_array,
        metadata,
        report,
    )


# ============================================================
# SEQUENCE INSPECTION
# ============================================================

def print_sequence_report(
    report: dict,
    dataset_name: str = "DATASET",
) -> None:
    """
    Print a readable sequence-creation report.

    Works for:
    - TRAIN: sequences with anomaly targets
    - VALIDATION/TEST: sequences without anomaly targets
    """

    print("=" * 80)
    print(f"{dataset_name} — LSTM SEQUENCE REPORT")
    print("=" * 80)

    print(
        f"\nInput rows: "
        f"{report['input_rows']:,}"
    )

    print(
        f"Sequence length: "
        f"{report['sequence_length']}"
    )

    print(
        f"Number of features: "
        f"{report['number_features']}"
    )

    print(
        f"\nTotal possible windows: "
        f"{report['total_possible_windows']:,}"
    )

    print(
        f"Created sequences: "
        f"{report['created_sequences']:,}"
    )

    print(
        f"Rejected boundary windows: "
        f"{report['rejected_boundary_windows']:,}"
    )

    print(
        f"Usable windows: "
        f"{report['safe_sequence_percentage']:.2f}%"
    )

    # --------------------------------------------------------
    # Target information
    # --------------------------------------------------------

    if report.get("has_targets", False):

        print(
            f"\nNormal target sequences: "
            f"{report['normal_target_sequences']:,}"
        )

        print(
            f"Anomaly target sequences: "
            f"{report['anomaly_target_sequences']:,}"
        )

        print(
            "\nTargets: available "
            "(anomaly injection / labels present)"
        )

    else:

        print(
            "\nTargets: not created "
            "(no anomaly injection in this dataset)"
        )

    print("=" * 80)

# ============================================================
# SAVE / LOAD
# ============================================================

# ============================================================
# SAVE / LOAD
# ============================================================

def save_sequences(
    X: np.ndarray,
    y: np.ndarray | None,
    metadata: pd.DataFrame,
    output_prefix: str,
) -> None:
    """
    Save LSTM arrays and sequence metadata.

    Training:
        Saves X, y, and metadata.

    Validation/Test:
        Saves X and metadata.
        No y file is created because these datasets
        do not contain injected anomaly labels.

    Examples
    --------
    Training:
        save_sequences(
            X_train,
            y_train,
            train_metadata,
            "data/processed/train_sequences",
        )

    Validation:
        save_sequences(
            X_validation,
            None,
            validation_metadata,
            "data/processed/validation_sequences",
        )
    """

    # --------------------------------------------------------
    # Save X
    # --------------------------------------------------------

    np.save(
        f"{output_prefix}_X.npy",
        X,
    )

    # --------------------------------------------------------
    # Save y only when available
    # --------------------------------------------------------

    if y is not None:

        np.save(
            f"{output_prefix}_y.npy",
            y,
        )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata.to_parquet(
        f"{output_prefix}_metadata.parquet",
        index=False,
    )


def load_sequences(
    input_prefix: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load previously saved LSTM sequences.
    """

    X = np.load(
        f"{input_prefix}_X.npy"
    )

    y = np.load(
        f"{input_prefix}_y.npy",
        allow_pickle=True,
    )

    metadata = pd.read_parquet(
        f"{input_prefix}_metadata.parquet"
    )

    return X, y, metadata