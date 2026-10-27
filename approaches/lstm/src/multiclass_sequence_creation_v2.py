
"""
Model 2 — Multiclass LSTM sequence creation.

Creates sequences from scaled Model 2 features.

Target:
    The anomaly type of the FINAL observation in each sequence.

Normal final observations are discarded.

Previous anomalies inside a sequence are allowed because the full
chronological time series is used as temporal context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20

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


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_multiclass_sequences(
    X_scaled: np.ndarray,
    tipo_anomalia: pd.Series,
    feature_names: list[str],
    split_name: str = "data",
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create multiclass LSTM sequences for Model 2.

    Parameters
    ----------
    X_scaled:
        2D scaled feature matrix with shape
        (n_rows, n_features).

    tipo_anomalia:
        Anomaly-type labels aligned row-by-row with X_scaled.
        Normal observations should contain NaN.

    feature_names:
        List of Model 2 feature names.

    split_name:
        Name used in validation/error messages.

    sequence_length:
        Number of timesteps per sequence.

    Returns
    -------
    X_sequences:
        Array with shape
        (n_sequences, sequence_length, n_features).

    y_sequences:
        Integer class labels with shape
        (n_sequences,).
    """

    # --------------------------------------------------------
    # Validate X
    # --------------------------------------------------------

    X_scaled = np.asarray(X_scaled, dtype=np.float32)

    if X_scaled.ndim != 2:
        raise ValueError(
            f"{split_name}: X_scaled must be 2D. "
            f"Received shape: {X_scaled.shape}"
        )

    n_rows, n_features = X_scaled.shape

    if n_features != len(feature_names):
        raise ValueError(
            f"{split_name}: feature mismatch. "
            f"X contains {n_features} features, "
            f"but feature_names contains {len(feature_names)}."
        )

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            f"{split_name}: X_scaled contains NaN or Inf."
        )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    tipo_anomalia = pd.Series(tipo_anomalia).reset_index(drop=True)

    if len(tipo_anomalia) != n_rows:
        raise ValueError(
            f"{split_name}: X and tipo_anomalia have different "
            f"number of rows: {n_rows} vs {len(tipo_anomalia)}."
        )

    if n_rows < sequence_length:
        raise ValueError(
            f"{split_name}: not enough rows to create "
            f"{sequence_length}-step sequences."
        )

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    X_sequences = []
    y_sequences = []

    discarded_normal = 0
    unknown_classes = set()

    for end_idx in range(
        sequence_length - 1,
        n_rows,
    ):
        start_idx = end_idx - sequence_length + 1

        # Target = anomaly type of FINAL observation
        target_type = tipo_anomalia.iloc[end_idx]

        # Normal final observation → no Model 2 target
        if pd.isna(target_type):
            discarded_normal += 1
            continue

        target_type = str(target_type).strip()

        # Unknown anomaly type
        if target_type not in CLASS_MAPPING:
            unknown_classes.add(target_type)
            continue

        # Full temporal context
        sequence = X_scaled[
            start_idx:end_idx + 1
        ]

        X_sequences.append(sequence)
        y_sequences.append(
            CLASS_MAPPING[target_type]
        )

    # --------------------------------------------------------
    # Validate anomaly classes
    # --------------------------------------------------------

    if unknown_classes:
        raise ValueError(
            f"{split_name}: unknown anomaly types found: "
            f"{sorted(unknown_classes)}"
        )

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    X_sequences = np.asarray(
        X_sequences,
        dtype=np.float32,
    )

    y_sequences = np.asarray(
        y_sequences,
        dtype=np.int64,
    )

    # Handle case where no anomaly sequences exist
    if len(X_sequences) == 0:
        X_sequences = np.empty(
            (
                0,
                sequence_length,
                n_features,
            ),
            dtype=np.float32,
        )

        y_sequences = np.empty(
            (0,),
            dtype=np.int64,
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_shape = (
        len(y_sequences),
        sequence_length,
        n_features,
    )

    if X_sequences.shape != expected_shape:
        raise RuntimeError(
            f"{split_name}: unexpected X shape. "
            f"Expected {expected_shape}, "
            f"received {X_sequences.shape}."
        )

    if len(X_sequences) != len(y_sequences):
        raise RuntimeError(
            f"{split_name}: X/y length mismatch."
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(f"{split_name.upper()} — MODEL 2 SEQUENCE CREATION")
    print("=" * 80)

    print(f"Input rows:               {n_rows:,}")
    print(f"Number of features:       {n_features}")
    print(f"Sequence length:          {sequence_length}")
    print(
        f"Possible sequences:       "
        f"{n_rows - sequence_length + 1:,}"
    )
    print(f"Created sequences:        {len(X_sequences):,}")
    print(f"Discarded normal targets: {discarded_normal:,}")
    print(f"X shape:                  {X_sequences.shape}")
    print(f"y shape:                  {y_sequences.shape}")

    print("\nClass distribution:")

    if len(y_sequences) > 0:
        counts = np.bincount(
            y_sequences,
            minlength=len(CLASS_MAPPING),
        )

        for class_id, class_name in REVERSE_CLASS_MAPPING.items():
            count = counts[class_id]
            percentage = (
                100 * count / len(y_sequences)
            )

            print(
                f"  {class_id}: "
                f"{class_name:<28} "
                f"{count:>8,} "
                f"({percentage:6.2f}%)"
            )

    print("=" * 80)

    return X_sequences, y_sequences


# ============================================================
# VALIDATION
# ============================================================

def validate_multiclass_sequences(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict:
    """
    Validate Model 2 sequence arrays.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    expected_features = len(feature_names)

    checks = {
        "X_is_3d": X.ndim == 3,
        "y_is_1d": y.ndim == 1,
        "correct_sequence_length": (
            X.ndim == 3
            and X.shape[1] == sequence_length
        ),
        "correct_feature_count": (
            X.ndim == 3
            and X.shape[2] == expected_features
        ),
        "same_number_of_samples": (
            len(X) == len(y)
        ),
        "X_is_finite": (
            np.isfinite(X).all()
        ),
        "valid_class_labels": (
            len(y) == 0
            or np.isin(
                y,
                list(REVERSE_CLASS_MAPPING.keys()),
            ).all()
        ),
    }

    checks["all_checks_passed"] = all(
        checks.values()
    )

    return checks


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    y_train: np.ndarray,
) -> dict[int, float]:
    """
    Calculate balanced class weights using TRAIN labels only.
    """

    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

