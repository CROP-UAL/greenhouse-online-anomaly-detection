# ============================================================
# src/scaling.py
# ============================================================

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# MODEL FEATURES
# ============================================================

# These 18 continuous / engineered features are standardized.
MODEL_FEATURES = [
    # --------------------------------------------------------
    # Original sensor measurements
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Feature-engineered variables
    # --------------------------------------------------------
    "hour_sin",
    "hour_cos",
    "is_night",
    "delta_temp",
    "delta_hum",
    "co2_gradient",
]


# ============================================================
# MISSINGNESS FEATURE
# ============================================================

# This feature is intentionally NOT standardized.
#
# 0 = no sensor was originally missing in this row
# 1 = at least one sensor was originally missing in this row
#
# It is kept as a binary feature and will be passed directly
# to the LSTM together with the 18 scaled features.
MISSINGNESS_FEATURE = "any_sensor_was_missing"


# ============================================================
# LSTM INPUT FEATURES
# ============================================================

# These are the actual columns that will later be used by the
# LSTM sequence-creation step.
#
# The first 18 are the scaled versions of MODEL_FEATURES.
# The final feature remains binary 0/1.
SCALED_MODEL_FEATURES = [
    f"{feature}_scaled"
    for feature in MODEL_FEATURES
]

LSTM_FEATURES = (
    SCALED_MODEL_FEATURES
    + [MISSINGNESS_FEATURE]
)


# ============================================================
# CHECK MODEL FEATURES
# ============================================================

def validate_model_features(
    df: pd.DataFrame,
    model_features: List[str] = MODEL_FEATURES,
) -> None:
    """
    Check that all continuous/engineered model features
    exist in the dataframe.
    """

    missing = [
        col
        for col in model_features
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "The following model features are missing "
            f"from the dataframe: {missing}"
        )


# ============================================================
# CHECK MISSINGNESS FEATURE
# ============================================================

def validate_missingness_feature(
    df: pd.DataFrame,
) -> None:
    """
    Check that the row-level missingness indicator exists.
    """

    if MISSINGNESS_FEATURE not in df.columns:
        raise ValueError(
            "The required missingness feature "
            f"'{MISSINGNESS_FEATURE}' is missing "
            "from the dataframe."
        )


# ============================================================
# CHECK LSTM FEATURES
# ============================================================

def validate_lstm_features(
    df: pd.DataFrame,
) -> None:
    """
    Check that all features required by the LSTM exist.
    """

    missing = [
        col
        for col in LSTM_FEATURES
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "The following LSTM features are missing "
            f"from the dataframe: {missing}"
        )


# ============================================================
# FIT SCALER
# ============================================================

def fit_scaler(
    df_train: pd.DataFrame,
    model_features: List[str] = MODEL_FEATURES,
) -> StandardScaler:
    """
    Fit StandardScaler using TRAINING DATA ONLY.

    Only the 18 continuous/engineered features are
    standardized.

    The binary missingness feature is NOT included in
    the scaler.

    NaN values are not imputed here. Missing-value handling
    must be completed before calling this function.
    """

    # --------------------------------------------------------
    # Validate required continuous features
    # --------------------------------------------------------

    validate_model_features(
        df_train,
        model_features,
    )

    # --------------------------------------------------------
    # Validate missingness feature
    # --------------------------------------------------------

    validate_missingness_feature(
        df_train
    )

    # --------------------------------------------------------
    # Create scaler
    # --------------------------------------------------------

    scaler = StandardScaler()

    # --------------------------------------------------------
    # IMPORTANT:
    # Fit ONLY using TRAINING DATA.
    #
    # The missingness indicator is intentionally excluded.
    # --------------------------------------------------------

    scaler.fit(
        df_train[model_features]
    )

    return scaler


# ============================================================
# TRANSFORM DATA
# ============================================================

def transform_with_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    model_features: List[str] = MODEL_FEATURES,
) -> pd.DataFrame:
    """
    Transform model features using an already-fitted scaler.

    The original features are preserved.

    New columns are created with the suffix '_scaled'.

    The missingness indicator remains binary 0/1 and is
    NOT standardized.

    Parameters
    ----------
    df:
        DataFrame to transform.

    scaler:
        StandardScaler fitted on TRAINING DATA ONLY.

    model_features:
        Continuous/engineered features to standardize.

    Returns
    -------
    pd.DataFrame
        Original dataframe plus scaled features and the
        binary missingness feature.
    """

    # --------------------------------------------------------
    # Validate continuous features
    # --------------------------------------------------------

    validate_model_features(
        df,
        model_features,
    )

    # --------------------------------------------------------
    # Validate missingness feature
    # --------------------------------------------------------

    validate_missingness_feature(
        df
    )

    # --------------------------------------------------------
    # Copy original dataframe
    # --------------------------------------------------------

    result = df.copy()

    # --------------------------------------------------------
    # Standardize the 18 continuous/engineered features
    # --------------------------------------------------------

    scaled_values = scaler.transform(
        result[model_features]
    )

    # --------------------------------------------------------
    # Store scaled features
    # --------------------------------------------------------

    for i, feature in enumerate(model_features):

        result[f"{feature}_scaled"] = (
            scaled_values[:, i]
        )

    # --------------------------------------------------------
    # Keep missingness indicator as binary 0/1
    # --------------------------------------------------------

    result[MISSINGNESS_FEATURE] = (
        result[MISSINGNESS_FEATURE]
        .fillna(0)
        .astype("int8")
    )

    return result


# ============================================================
# SCALE TRAIN / VALIDATION / TEST
# ============================================================

def scale_train_validation_test(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_features: List[str] = MODEL_FEATURES,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
]:
    """
    Complete scaling pipeline.

    1. Fit scaler ONLY on TRAIN.
    2. Transform TRAIN.
    3. Transform VALIDATION using TRAIN scaler.
    4. Transform TEST using TRAIN scaler.

    The 18 continuous/engineered features are standardized.

    The binary 'any_sensor_was_missing' feature is preserved
    as 0/1 and is not standardized.

    Returns
    -------
    train_scaled
    validation_scaled
    test_scaled
    fitted_scaler
    """

    # ========================================================
    # 1. FIT ONLY ON TRAIN
    # ========================================================

    scaler = fit_scaler(
        train_df,
        model_features,
    )

    # ========================================================
    # 2. TRANSFORM TRAIN
    # ========================================================

    train_scaled = transform_with_scaler(
        train_df,
        scaler,
        model_features,
    )

    # ========================================================
    # 3. TRANSFORM VALIDATION
    # ========================================================

    validation_scaled = transform_with_scaler(
        validation_df,
        scaler,
        model_features,
    )

    # ========================================================
    # 4. TRANSFORM TEST
    # ========================================================

    test_scaled = transform_with_scaler(
        test_df,
        scaler,
        model_features,
    )

    # ========================================================
    # 5. RETURN RESULTS
    # ========================================================

    return (
        train_scaled,
        validation_scaled,
        test_scaled,
        scaler,
    )


# ============================================================
# CHECK SCALED FEATURES
# ============================================================

def validate_scaled_features(
    df: pd.DataFrame,
) -> None:
    """
    Validate that all 19 features required by the LSTM
    are present.
    """

    validate_lstm_features(df)

    # --------------------------------------------------------
    # Check for NaNs
    # --------------------------------------------------------

    nan_counts = (
        df[LSTM_FEATURES]
        .isna()
        .sum()
    )

    remaining_nans = (
        nan_counts[nan_counts > 0]
    )

    if len(remaining_nans) > 0:

        raise ValueError(
            "NaNs remain in the LSTM input features:\n"
            f"{remaining_nans}"
        )

    # --------------------------------------------------------
    # Check binary missingness feature
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
            f"only 0/1 values. Found: "
            f"{sorted(unique_values)}"
        )


# ============================================================
# PRINT SCALING SUMMARY
# ============================================================

def print_scaling_summary(
    df: pd.DataFrame,
    dataset_name: str = "DATASET",
) -> None:
    """
    Print a summary of the features that will be supplied
    to the LSTM.
    """

    print("=" * 80)
    print(f"{dataset_name} — SCALING SUMMARY")
    print("=" * 80)

    print(
        f"\nStandardized features: "
        f"{len(MODEL_FEATURES)}"
    )

    for feature in MODEL_FEATURES:
        print(f"  - {feature}_scaled")

    print(
        f"\nBinary feature: "
        f"{MISSINGNESS_FEATURE}"
    )

    print(
        "\nTotal LSTM input features: "
        f"{len(LSTM_FEATURES)}"
    )

    print(
        f"\n{MISSINGNESS_FEATURE} distribution:"
    )

    print(
        df[MISSINGNESS_FEATURE]
        .value_counts()
        .sort_index()
    )

    print("=" * 80)


# ============================================================
# SAVE SCALER
# ============================================================

def save_scaler(
    scaler: StandardScaler,
    path: str | Path,
) -> None:
    """
    Save fitted scaler for reproducibility.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        path,
    )


# ============================================================
# LOAD SCALER
# ============================================================

def load_scaler(
    path: str | Path,
) -> StandardScaler:
    """
    Load a previously fitted scaler.
    """

    return joblib.load(path)