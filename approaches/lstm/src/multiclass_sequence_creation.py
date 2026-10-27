
"""
Multiclass LSTM sequence creation for CIESOL anomaly classification.

Purpose
-------
Create 20-step sequences for Model 2 (multiclass anomaly classifier).

Model 2 predicts the anomaly type of the FINAL observation in each
20-step sequence.

Target classes
--------------
0 = Contextual
1 = Ruido
2 = Desviacion de Correlacion
3 = Datos Faltantes
4 = Valores Fuera de Rango
5 = Sensor Atascado

Important
---------
- The complete time series is used to create sequences.
- We DO NOT filter anomaly rows before sequence creation.
- Normal observations are kept as temporal context.
- A sequence is kept only when its FINAL observation is anomalous.
- Previous anomalies inside the 20-step window are allowed.
- Anomaly metadata is used only to create the target.
- Anomaly metadata is NEVER included in X.
- X contains exactly 19 model features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

SEQUENCE_LENGTH = 20


FEATURE_COLUMNS = [
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
    "any_sensor_was_missing",
]


CLASS_MAPPING = {
    "Contextual": 0,
    "Ruido": 1,
    "Desviacion de Correlacion": 2,
    "Datos Faltantes": 3,
    "Valores Fuera de Rango": 4,
    "Sensor Atascado": 5,
}


REVERSE_CLASS_MAPPING = {
    value: key
    for key, value in CLASS_MAPPING.items()
}


# These columns are required for target generation / validation.
TARGET_METADATA_COLUMNS = [
    "tipo_anomalia",
    "anomalia_id",
]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_input_dataframe(
    df: pd.DataFrame,
    split_name: str = "data",
) -> None:
    """
    Validate that the DataFrame contains everything required to create
    multiclass sequences.

    Parameters
    ----------
    df:
        Processed/scaled DataFrame.

    split_name:
        Name used in error messages, e.g. "TRAIN".
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{split_name}: expected pandas DataFrame, "
            f"got {type(df).__name__}."
        )

    required_columns = FEATURE_COLUMNS + TARGET_METADATA_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{split_name}: missing required columns:\n"
            f"{missing_columns}"
        )

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"{split_name}: only {len(df)} rows available, "
            f"but SEQUENCE_LENGTH={SEQUENCE_LENGTH}."
        )

    # Check feature dtypes / convertibility.
    for column in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"{split_name}: feature '{column}' is not numeric."
            )

    # Check for NaNs in model inputs.
    feature_nan_count = int(
        df[FEATURE_COLUMNS].isna().sum().sum()
    )

    if feature_nan_count > 0:
        nan_by_column = (
            df[FEATURE_COLUMNS]
            .isna()
            .sum()
            .loc[lambda x: x > 0]
            .to_dict()
        )

        raise ValueError(
            f"{split_name}: model features contain "
            f"{feature_nan_count} NaN values.\n"
            f"NaNs by column:\n{nan_by_column}\n\n"
            f"Missing-value handling must be completed before "
            f"multiclass sequence creation."
        )


def validate_class_mapping() -> None:
    """
    Validate that the class mapping is exactly the expected six-class setup.
    """

    expected_classes = {
        "Contextual": 0,
        "Ruido": 1,
        "Desviacion de Correlacion": 2,
        "Datos Faltantes": 3,
        "Valores Fuera de Rango": 4,
        "Sensor Atascado": 5,
    }

    if CLASS_MAPPING != expected_classes:
        raise ValueError(
            "CLASS_MAPPING does not match the expected six-class mapping.\n"
            f"Expected: {expected_classes}\n"
            f"Received: {CLASS_MAPPING}"
        )

    if set(REVERSE_CLASS_MAPPING.keys()) != set(range(6)):
        raise ValueError(
            "Reverse class mapping does not contain labels 0-5."
        )


# =============================================================================
# SEQUENCE CREATION
# =============================================================================

def create_multiclass_sequences(
    df: pd.DataFrame,
    split_name: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create multiclass LSTM sequences.

    A sequence contains 20 consecutive observations.

    The target is determined ONLY from the final observation:

        t-19 ... t-1 | t
                     ↑
                   target

    If the final observation is normal:
        - sequence is discarded.

    If the final observation is anomalous:
        - sequence is kept.
        - target = final observation's anomaly type.

    Therefore, previous anomalies inside the sequence are allowed and
    remain as temporal context.

    Parameters
    ----------
    df:
        Complete processed/scaled DataFrame.

    split_name:
        Human-readable split name for reports.

    Returns
    -------
    X:
        Float32 array of shape
        (number_of_anomalous_target_sequences, 20, 19)

    y:
        Int64 array of shape
        (number_of_anomalous_target_sequences,)
    """

    validate_class_mapping()
    validate_input_dataframe(df, split_name)

    # -------------------------------------------------------------------------
    # Reset index
    # -------------------------------------------------------------------------

    df = df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Extract model features
    # -------------------------------------------------------------------------

    X_all = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    # Final safety check.
    if not np.isfinite(X_all).all():
        bad_positions = np.argwhere(~np.isfinite(X_all))

        raise ValueError(
            f"{split_name}: X contains non-finite values "
            f"(NaN or infinity).\n"
            f"Number of bad values: {len(bad_positions)}"
        )

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    X_sequences = []
    y_sequences = []

    # -------------------------------------------------------------------------
    # Create windows
    # -------------------------------------------------------------------------

    total_possible_sequences = (
        len(df) - SEQUENCE_LENGTH + 1
    )

    discarded_normal_sequences = 0

    anomaly_rows_without_type = 0
    anomaly_rows_without_id = 0
    unknown_anomaly_types = set()

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    for end_idx in range(
        SEQUENCE_LENGTH - 1,
        len(df),
    ):

        start_idx = (
            end_idx - SEQUENCE_LENGTH + 1
        )

        # The target is ALWAYS the final observation.
        target_type = df.iloc[end_idx]["tipo_anomalia"]

        # ---------------------------------------------------------------------
        # Normal target → discard sequence
        # ---------------------------------------------------------------------

        if pd.isna(target_type):

            discarded_normal_sequences += 1
            continue

        # ---------------------------------------------------------------------
        # Anomalous target → validate metadata
        # ---------------------------------------------------------------------

        anomaly_id = df.iloc[end_idx]["anomalia_id"]

        if pd.isna(anomaly_id):
            anomaly_rows_without_id += 1

        # Convert target to string for robust comparison.
        target_type = str(target_type).strip()

        # ---------------------------------------------------------------------
        # Validate anomaly type
        # ---------------------------------------------------------------------

        if target_type not in CLASS_MAPPING:

            unknown_anomaly_types.add(target_type)

            continue

        # ---------------------------------------------------------------------
        # Extract 20-step sequence
        # ---------------------------------------------------------------------

        sequence = X_all[
            start_idx:end_idx + 1
        ]

        # Safety check.
        if sequence.shape != (
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS),
        ):
            raise RuntimeError(
                f"{split_name}: unexpected sequence shape "
                f"{sequence.shape} at end_idx={end_idx}."
            )

        # ---------------------------------------------------------------------
        # Convert anomaly type → integer class
        # ---------------------------------------------------------------------

        class_id = CLASS_MAPPING[target_type]

        X_sequences.append(sequence)
        y_sequences.append(class_id)

    # -------------------------------------------------------------------------
    # Unknown classes are a hard error
    # -------------------------------------------------------------------------

    if unknown_anomaly_types:

        raise ValueError(
            f"{split_name}: unknown anomaly types found:\n"
            f"{sorted(unknown_anomaly_types)}\n\n"
            f"Expected only:\n"
            f"{list(CLASS_MAPPING.keys())}"
        )

    # -------------------------------------------------------------------------
    # Convert lists to NumPy arrays
    # -------------------------------------------------------------------------

    if X_sequences:

        X = np.asarray(
            X_sequences,
            dtype=np.float32,
        )

        y = np.asarray(
            y_sequences,
            dtype=np.int64,
        )

    else:

        X = np.empty(
            (
                0,
                SEQUENCE_LENGTH,
                len(FEATURE_COLUMNS),
            ),
            dtype=np.float32,
        )

        y = np.empty(
            (0,),
            dtype=np.int64,
        )

    # -------------------------------------------------------------------------
    # Validate final arrays
    # -------------------------------------------------------------------------

    expected_shape = (
        len(y),
        SEQUENCE_LENGTH,
        len(FEATURE_COLUMNS),
    )

    if X.shape != expected_shape:

        raise RuntimeError(
            f"{split_name}: unexpected final X shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {X.shape}"
        )

    if y.shape != (len(y),):

        raise RuntimeError(
            f"{split_name}: unexpected y shape: {y.shape}"
        )

    if len(X) != len(y):

        raise RuntimeError(
            f"{split_name}: X/y length mismatch.\n"
            f"X: {len(X)}\n"
            f"y: {len(y)}"
        )

    if len(y) > 0:

        if not np.isfinite(X).all():

            raise ValueError(
                f"{split_name}: final X contains "
                f"non-finite values."
            )

        if not np.isin(
            y,
            list(CLASS_MAPPING.values()),
        ).all():

            raise ValueError(
                f"{split_name}: y contains labels outside "
                f"the valid range 0-5."
            )

    # -------------------------------------------------------------------------
    # Store report information
    # -------------------------------------------------------------------------

    anomaly_rows_without_type = 0

    # The only rows that become targets are the rows with non-null
    # tipo_anomalia. Since unknown types already raise an error, the
    # final number of anomaly targets is simply len(y).

    print()
    print("=" * 72)
    print(f"{split_name.upper()} — MULTICLASS SEQUENCE CREATION")
    print("=" * 72)

    print(f"Input rows:                 {len(df):,}")
    print(f"Sequence length:            {SEQUENCE_LENGTH}")
    print(f"Number of features:         {len(FEATURE_COLUMNS)}")
    print(f"Total possible sequences:   {total_possible_sequences:,}")
    print(f"Created sequences:          {len(X):,}")
    print(f"Discarded normal targets:   {discarded_normal_sequences:,}")
    print(f"Output X shape:             {X.shape}")
    print(f"Output y shape:             {y.shape}")

    if anomaly_rows_without_id > 0:
        print(
            f"WARNING: anomaly targets without ID: "
            f"{anomaly_rows_without_id:,}"
        )
    else:
        print("Anomaly targets without ID: 0")

    print()
    print("Class distribution:")

    if len(y) == 0:

        print("  No anomalous target sequences found.")

    else:

        counts = np.bincount(
            y,
            minlength=len(CLASS_MAPPING),
        )

        for class_id, class_name in REVERSE_CLASS_MAPPING.items():

            count = int(counts[class_id])
            percentage = (
                100.0 * count / len(y)
            )

            print(
                f"  {class_id}: "
                f"{class_name:<28} "
                f"{count:>7,} "
                f"({percentage:6.2f}%)"
            )

    print("=" * 72)

    return X, y


# =============================================================================
# DATASET REPORTING
# =============================================================================

def print_class_distribution(
    y: np.ndarray,
    split_name: str = "data",
) -> None:
    """
    Print class counts and percentages.
    """

    if y.ndim != 1:
        raise ValueError(
            f"{split_name}: expected 1D y array, got {y.shape}."
        )

    print()
    print(f"{split_name.upper()} — CLASS DISTRIBUTION")
    print("-" * 72)

    total = len(y)

    if total == 0:
        print("No samples.")
        return

    counts = np.bincount(
        y,
        minlength=len(CLASS_MAPPING),
    )

    for class_id, class_name in REVERSE_CLASS_MAPPING.items():

        count = int(counts[class_id])
        percentage = 100.0 * count / total

        print(
            f"{class_id}: "
            f"{class_name:<28} "
            f"{count:>7,} "
            f"({percentage:6.2f}%)"
        )

    print("-" * 72)
    print(f"Total: {total:,}")


def validate_sequences(
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "data",
) -> None:
    """
    Perform final validation of generated multiclass sequences.
    """

    expected_features = len(FEATURE_COLUMNS)

    # -------------------------------------------------------------------------
    # Shape
    # -------------------------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(
            f"{split_name}: X must be 3D, got {X.ndim}D."
        )

    if X.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(
            f"{split_name}: expected sequence length "
            f"{SEQUENCE_LENGTH}, got {X.shape[1]}."
        )

    if X.shape[2] != expected_features:

        raise ValueError(
            f"{split_name}: expected "
            f"{expected_features} features, got {X.shape[2]}."
        )

    if y.ndim != 1:

        raise ValueError(
            f"{split_name}: y must be 1D, got {y.ndim}D."
        )

    if len(X) != len(y):

        raise ValueError(
            f"{split_name}: X/y length mismatch: "
            f"{len(X)} vs {len(y)}."
        )

    # -------------------------------------------------------------------------
    # Numeric validity
    # -------------------------------------------------------------------------

    if not np.isfinite(X).all():

        raise ValueError(
            f"{split_name}: X contains NaN or infinite values."
        )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    valid_labels = np.arange(
        len(CLASS_MAPPING)
    )

    if len(y) > 0:

        if not np.isin(y, valid_labels).all():

            invalid_labels = np.unique(
                y[
                    ~np.isin(
                        y,
                        valid_labels,
                    )
                ]
            )

            raise ValueError(
                f"{split_name}: invalid labels: "
                f"{invalid_labels.tolist()}"
            )

    # -------------------------------------------------------------------------
    # Dtypes
    # -------------------------------------------------------------------------

    if X.dtype != np.float32:

        raise ValueError(
            f"{split_name}: X must be float32, "
            f"got {X.dtype}."
        )

    if y.dtype != np.int64:

        raise ValueError(
            f"{split_name}: y must be int64, "
            f"got {y.dtype}."
        )

    print()
    print(
        f"✓ {split_name}: sequence validation passed."
    )
    print(
        f"  X: {X.shape}, dtype={X.dtype}"
    )
    print(
        f"  y: {y.shape}, dtype={y.dtype}"
    )


# =============================================================================
# SAVING
# =============================================================================

def save_multiclass_sequences(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str | Path,
    split_name: str,
) -> Tuple[Path, Path]:
    """
    Save multiclass X/y arrays.

    Files
    -----
    {split_name}_multiclass_sequences_X.npy
    {split_name}_multiclass_sequences_y.npy
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_name = split_name.lower()

    X_path = (
        output_dir
        / f"{split_name}_multiclass_sequences_X.npy"
    )

    y_path = (
        output_dir
        / f"{split_name}_multiclass_sequences_y.npy"
    )

    np.save(X_path, X)
    np.save(y_path, y)

    print()
    print(f"Saved {split_name.upper()}:")
    print(f"  X → {X_path}")
    print(f"  y → {y_path}")

    return X_path, y_path


# =============================================================================
# LOAD HELPER
# =============================================================================

def load_multiclass_sequences(
    output_dir: str | Path,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load previously saved multiclass sequences.
    """

    output_dir = Path(output_dir)
    split_name = split_name.lower()

    X_path = (
        output_dir
        / f"{split_name}_multiclass_sequences_X.npy"
    )

    y_path = (
        output_dir
        / f"{split_name}_multiclass_sequences_y.npy"
    )

    if not X_path.exists():

        raise FileNotFoundError(
            f"X file not found:\n{X_path}"
        )

    if not y_path.exists():

        raise FileNotFoundError(
            f"y file not found:\n{y_path}"
        )

    X = np.load(X_path)
    y = np.load(y_path)

    validate_sequences(
        X,
        y,
        split_name=split_name,
    )

    return X, y


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(
    y_train: np.ndarray,
) -> Dict[int, float]:
    """
    Calculate balanced class weights using TRAIN labels only.

    Formula
    -------
    weight_i = N / (number_of_classes * count_i)

    Validation and test labels must NOT be used to calculate weights.
    """

    if y_train.ndim != 1:

        raise ValueError(
            "y_train must be 1-dimensional."
        )

    if len(y_train) == 0:

        raise ValueError(
            "y_train is empty."
        )

    num_classes = len(CLASS_MAPPING)

    counts = np.bincount(
        y_train,
        minlength=num_classes,
    )

    missing_classes = [
        class_id
        for class_id, count in enumerate(counts)
        if count == 0
    ]

    if missing_classes:

        missing_names = [
            REVERSE_CLASS_MAPPING[class_id]
            for class_id in missing_classes
        ]

        raise ValueError(
            "The training data contains no samples for "
            f"these classes: {missing_names}"
        )

    total = len(y_train)

    class_weights = {
        class_id: total / (
            num_classes * counts[class_id]
        )
        for class_id in range(num_classes)
    }

    print()
    print("=" * 72)
    print("TRAIN — CLASS WEIGHTS")
    print("=" * 72)

    for class_id in range(num_classes):

        print(
            f"{class_id}: "
            f"{REVERSE_CLASS_MAPPING[class_id]:<28} "
            f"count={counts[class_id]:>7,} "
            f"weight={class_weights[class_id]:.6f}"
        )

    print("=" * 72)

    return class_weights


# =============================================================================
# OPTIONAL DATASET SUMMARY
# =============================================================================

def print_feature_summary() -> None:
    """
    Print the final feature configuration used by Model 2.
    """

    print()
    print("=" * 72)
    print("MODEL 2 — FEATURE CONFIGURATION")
    print("=" * 72)

    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")

    print()
    print("Features:")

    for index, feature in enumerate(FEATURE_COLUMNS):
        print(
            f"  {index:>2}: {feature}"
        )

    print()
    print("Target classes:")

    for class_id, class_name in REVERSE_CLASS_MAPPING.items():
        print(
            f"  {class_id}: {class_name}"
        )

    print("=" * 72)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SEQUENCE_LENGTH",
    "FEATURE_COLUMNS",
    "CLASS_MAPPING",
    "REVERSE_CLASS_MAPPING",
    "create_multiclass_sequences",
    "validate_sequences",
    "print_class_distribution",
    "save_multiclass_sequences",
    "load_multiclass_sequences",
    "calculate_class_weights",
    "print_feature_summary",
]

