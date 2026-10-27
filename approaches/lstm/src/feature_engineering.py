"""
Feature Engineering for the CIESOL Sensor Anomaly Detection Project.

This module creates contextual and physical relationship features
from the original sensor measurements.

The function `engineer_features()` is designed to be applied
consistently to:

    - injected training data
    - validation data
    - test data

Important:
    - No anomaly injection is performed here.
    - No scaling is performed here.
    - Existing columns are preserved.
    - The input DataFrame is never modified in place.
    - Anomaly metadata columns are NOT used to create features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Original Agroconnect sensor columns used as model inputs.
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


# Engineered features created by this module.
ENGINEERED_FEATURES = [
    "hour_sin",
    "hour_cos",
    "is_night",
    "delta_temp",
    "delta_hum",
    "co2_gradient",
]


# ------------------------------------------------------------
# Night-time definition
# ------------------------------------------------------------
#
# This is deliberately defined using clock time rather than
# PRAD values.
#
# Reason:
# PRAD may itself contain an injected anomaly in the training
# dataset. We do not want an anomaly in PRAD to determine
# whether a timestamp is classified as night or day.
#
# These limits can be adjusted later if the EDA justifies it.
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 7


# ------------------------------------------------------------
# Radiation ratio protection
# ------------------------------------------------------------
#
# When PRAD is zero or extremely close to zero, the ratio
#
#     PRGINT / PRAD
#
# becomes unstable.
#
# Therefore, values with PRAD <= this threshold are assigned
# NaN. This can later be handled consistently during the
# modeling/preprocessing stage.

# ============================================================
# VALIDATION
# ============================================================

def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Validate that all required columns are present.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    required_columns : list[str]
        Columns required by the feature-engineering process.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns for feature engineering: "
            f"{missing_columns}"
        )


# ============================================================
# TIME FEATURES
# ============================================================

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features and a night/day indicator.

    Creates:
        - hour_sin
        - hour_cos
        - is_night

    The original FECHA column is preserved.
    """

    # Make sure FECHA is datetime.
    if not pd.api.types.is_datetime64_any_dtype(df["FECHA"]):
        df["FECHA"] = pd.to_datetime(
            df["FECHA"],
            errors="coerce",
        )

    # Extract decimal hour.
    #
    # Including minutes/seconds gives a smoother representation
    # than using only the integer hour.
    decimal_hour = (
        df["FECHA"].dt.hour
        + df["FECHA"].dt.minute / 60.0
        + df["FECHA"].dt.second / 3600.0
    )

    # Cyclical encoding.
    df["hour_sin"] = np.sin(
        2.0 * np.pi * decimal_hour / 24.0
    )

    df["hour_cos"] = np.cos(
        2.0 * np.pi * decimal_hour / 24.0
    )

    # Explicit night/day context.
    #
    # Example with the default configuration:
    #
    #   00:00 -> night
    #   06:30 -> night
    #   07:00 -> day
    #   14:00 -> day
    #   20:00 -> night
    #
    # The interval crosses midnight.
    df["is_night"] = (
        (decimal_hour >= NIGHT_START_HOUR)
        | (decimal_hour < NIGHT_END_HOUR)
    ).astype("int8")

    return df


# ============================================================
# PHYSICAL RELATIONSHIP FEATURES
# ============================================================

def _add_physical_relationship_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add features describing relationships between sensors.

    Creates:
        - delta_temp
        - delta_hum
        - radiation_ratio
        - co2_gradient
    """

    # --------------------------------------------------------
    # Internal vs external temperature
    # --------------------------------------------------------
    #
    # Positive values mean:
    #
    #     internal temperature > external temperature
    #
    df["delta_temp"] = (
        df["XTINV"] - df["PTEXT"]
    )

    # --------------------------------------------------------
    # Internal vs external humidity
    # --------------------------------------------------------
    df["delta_hum"] = (
        df["XHINV"] - df["PHEXT"]
    )

    # --------------------------------------------------------
    # Internal vs external CO2
    # --------------------------------------------------------
    df["co2_gradient"] = (
        df["XCO2I"] - df["PCO2EXT"]
    )

    return df


# ============================================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================================

def engineer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the project's engineered features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input Agroconnect dataframe.

        This can be:
            - injected training data
            - clean validation data
            - clean test data

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe with engineered features
        added.

    Notes
    -----
    The following seven features are currently created:

        1. hour_sin
        2. hour_cos
        3. is_night
        4. delta_temp
        5. delta_hum
        6. radiation_ratio
        7. co2_gradient

    Existing columns are preserved.
    The original dataframe is not modified.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    # Required columns.
    required_columns = [
        "FECHA",
        *SENSOR_COLUMNS,
    ]

    _validate_required_columns(
        df,
        required_columns,
    )

    # Never modify the original dataframe.
    df_features = df.copy()

    # Add temporal features.
    df_features = _add_time_features(
        df_features
    )

    # Add physical relationship features.
    df_features = _add_physical_relationship_features(
        df_features
    )

    return df_features


# ============================================================
# MODEL FEATURE SELECTION
# ============================================================

def get_model_features() -> list[str]:
    """
    Return the columns that should initially be used as
    LSTM input features.

    Returns
    -------
    list[str]
        19 model features:

            12 original sensors
            +
            7 engineered features
    """

    return SENSOR_COLUMNS + ENGINEERED_FEATURES


# ============================================================
# FEATURE SUMMARY
# ============================================================

def get_feature_groups() -> dict[str, list[str]]:
    """
    Return the project's feature groups.

    Useful for documentation, debugging, and notebook
    inspection.
    """

    return {
        "raw_sensors": SENSOR_COLUMNS.copy(),
        "temporal_features": [
            "hour_sin",
            "hour_cos",
            "is_night",
        ],
        "physical_relationship_features": [
            "delta_temp",
            "delta_hum",
            "co2_gradient",
        ],
    }


# ============================================================
# SIMPLE VALIDATION
# ============================================================

def validate_engineered_features(
    df: pd.DataFrame,
) -> dict[str, object]:
    """
    Perform basic checks on the engineered dataframe.

    Returns a dictionary that can be printed in the notebook.

    This function does not modify the dataframe.
    """

    required_engineered = ENGINEERED_FEATURES

    missing = [
        column
        for column in required_engineered
        if column not in df.columns
    ]

    if missing:
        return {
            "valid": False,
            "missing_features": missing,
            "n_features": 0,
        }

    # Check cyclical features.
    hour_sin_valid = df["hour_sin"].between(
        -1.0,
        1.0,
        inclusive="both",
    ).all()

    hour_cos_valid = df["hour_cos"].between(
        -1.0,
        1.0,
        inclusive="both",
    ).all()

    # is_night should only contain 0 and 1.
    night_values = set(
        df["is_night"]
        .dropna()
        .unique()
        .tolist()
    )

    is_night_valid = night_values.issubset(
        {0, 1}
    )

    # Count missing values in engineered features.
    missing_counts = (
        df[ENGINEERED_FEATURES]
        .isna()
        .sum()
        .to_dict()
    )

    return {
        "valid": (
            hour_sin_valid
            and hour_cos_valid
            and is_night_valid
        ),
        "missing_features": missing,
        "n_features": len(ENGINEERED_FEATURES),
        "engineered_missing_values": missing_counts,
        "hour_sin_valid": hour_sin_valid,
        "hour_cos_valid": hour_cos_valid,
        "is_night_valid": is_night_valid,
    }
