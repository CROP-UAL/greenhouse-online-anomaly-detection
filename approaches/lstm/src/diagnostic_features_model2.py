"""
Diagnostic feature engineering for Model 2.

Model 2 classifies the anomaly type after Model 1 detects an anomaly.

Classes:
    - Contextual
    - Datos Faltantes
    - Desviacion de Correlacion
    - Ruido
    - Sensor Atascado
    - Valores Fuera de Rango

This module must run:
    Existing feature engineering
        ↓
    Missing-value handling
        ↓
    Diagnostic features  <-- this module
        ↓
    Scaling
        ↓
    Sequence creation
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# SENSOR DEFINITIONS
# ============================================================================

TARGET_SENSORS = [
    "PCO2EXT",
    "PHEXT",
    "PRAD",
    "PRGINT",
    "PTEXT",
    "PVV",
    "XCO2I",
    "XHINV",
    "XTINV",
]

CONTEXT_SENSORS = [
    "UVENT_cen",
    "UVENT_lN",
]

ALL_SENSORS = TARGET_SENSORS + CONTEXT_SENSORS


# Physical ranges used by the anomaly injection methodology.
PHYSICAL_RANGES = {
    "PCO2EXT": (300.0, 800.0),
    "PHEXT": (0.0, 100.0),
    "PRAD": (0.0, 1500.0),
    "PRGINT": (0.0, 1200.0),
    "PTEXT": (-20.0, 60.0),
    "PVV": (0.0, 50.0),
    "XCO2I": (200.0, 3000.0),
    "XHINV": (0.0, 100.0),
    "XTINV": (-5.0, 60.0),
}


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _validate_sensors(
    df: pd.DataFrame,
    sensors: list[str],
) -> None:
    """Validate that all requested sensor columns exist."""

    missing = [
        sensor
        for sensor in sensors
        if sensor not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required sensor columns: {missing}"
        )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Safe element-wise division."""

    denominator = denominator.replace(0, np.nan)

    result = numerator / denominator

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ============================================================================
# 1. TEMPORAL DIAGNOSTIC FEATURES
# ============================================================================

def add_temporal_diagnostic_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """
    Add local temporal-dynamics features.

    Features per sensor:

        <sensor>_delta_prev
        <sensor>_delta_next
        <sensor>_abs_delta_prev
        <sensor>_abs_delta_next
        <sensor>_rolling_std
        <sensor>_rolling_range
    """

    if sensors is None:
        sensors = TARGET_SENSORS

    if rolling_window < 2:
        raise ValueError(
            "rolling_window must be >= 2"
        )

    result = df.copy()

    _validate_sensors(
        result,
        sensors,
    )

    for sensor in sensors:

        series = pd.to_numeric(
            result[sensor],
            errors="coerce",
        )

        # Difference from previous observation.
        delta_prev = series.diff()

        result[
            f"{sensor}_delta_prev"
        ] = delta_prev

        # Difference to next observation.
        delta_next = (
            series.shift(-1) - series
        )

        result[
            f"{sensor}_delta_next"
        ] = delta_next

        # Absolute changes.
        result[
            f"{sensor}_abs_delta_prev"
        ] = delta_prev.abs()

        result[
            f"{sensor}_abs_delta_next"
        ] = delta_next.abs()

        # Local rolling standard deviation.
        result[
            f"{sensor}_rolling_std"
        ] = (
            series
            .rolling(
                window=rolling_window,
                center=True,
                min_periods=2,
            )
            .std()
        )

        # Local rolling range.
        rolling_max = (
            series
            .rolling(
                window=rolling_window,
                center=True,
                min_periods=2,
            )
            .max()
        )

        rolling_min = (
            series
            .rolling(
                window=rolling_window,
                center=True,
                min_periods=2,
            )
            .min()
        )

        result[
            f"{sensor}_rolling_range"
        ] = rolling_max - rolling_min

    return result


# ============================================================================
# 2. PHYSICAL-RANGE DIAGNOSTIC FEATURES
# ============================================================================

def add_physical_diagnostic_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add physical-range diagnostic features.

    Features per sensor:

        <sensor>_is_out_of_physical_range
        <sensor>_distance_to_min
        <sensor>_distance_to_max
        <sensor>_normalized_position
    """

    if sensors is None:
        sensors = TARGET_SENSORS

    result = df.copy()

    _validate_sensors(
        result,
        sensors,
    )

    for sensor in sensors:

        if sensor not in PHYSICAL_RANGES:
            raise ValueError(
                f"No physical range defined for '{sensor}'."
            )

        lower, upper = PHYSICAL_RANGES[sensor]

        series = pd.to_numeric(
            result[sensor],
            errors="coerce",
        )

        # Out-of-range indicator.
        out_of_range = (
            (series < lower)
            | (series > upper)
        ).astype(np.float32)

        # Missing values are not physical violations.
        out_of_range.loc[
            series.isna()
        ] = np.nan

        result[
            f"{sensor}_is_out_of_physical_range"
        ] = out_of_range

        # Distance to physical minimum.
        result[
            f"{sensor}_distance_to_min"
        ] = series - lower

        # Distance to physical maximum.
        result[
            f"{sensor}_distance_to_max"
        ] = upper - series

        # Normalized position inside physical range.
        result[
            f"{sensor}_normalized_position"
        ] = (
            (series - lower)
            / (upper - lower)
        )

    return result


# ============================================================================
# 3. CROSS-SENSOR / CONTEXTUAL FEATURES
# ============================================================================

def add_relationship_diagnostic_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cross-sensor and contextual relationship features.

    Main relationships:

        PRAD ↔ PRGINT
        XTINV ↔ PTEXT
        XHINV ↔ PHEXT
        XCO2I ↔ ventilation
    """

    result = df.copy()

    required = [
        "PRAD",
        "PRGINT",
        "XTINV",
        "PTEXT",
        "XHINV",
        "PHEXT",
        "XCO2I",
        "UVENT_cen",
        "UVENT_lN",
    ]

    _validate_sensors(
        result,
        required,
    )

    # ------------------------------------------------------------------------
    # Solar radiation relationship
    # ------------------------------------------------------------------------

    result["PRGINT_PRAD_ratio"] = _safe_divide(
        result["PRGINT"],
        result["PRAD"],
    )

    result["PRAD_minus_PRGINT"] = (
        result["PRAD"]
        - result["PRGINT"]
    )

    result["PRGINT_minus_PRAD"] = (
        result["PRGINT"]
        - result["PRAD"]
    )

    result[
        "PRGINT_PRAD_relative_difference"
    ] = _safe_divide(
        result["PRGINT"] - result["PRAD"],
        result["PRAD"].abs(),
    )

    # ------------------------------------------------------------------------
    # Temperature relationship
    # ------------------------------------------------------------------------

    if "delta_temp" not in result.columns:
        result["delta_temp"] = (
            result["XTINV"]
            - result["PTEXT"]
        )

    result["temp_abs_difference"] = (
        result["XTINV"]
        - result["PTEXT"]
    ).abs()

    result["temp_ratio"] = _safe_divide(
        result["XTINV"],
        result["PTEXT"],
    )

    result[
        "interior_temp_below_exterior"
    ] = (
        result["XTINV"]
        < result["PTEXT"]
    ).astype(np.float32)

    # ------------------------------------------------------------------------
    # Humidity relationship
    # ------------------------------------------------------------------------

    if "delta_hum" not in result.columns:
        result["delta_hum"] = (
            result["XHINV"]
            - result["PHEXT"]
        )

    result["humidity_abs_difference"] = (
        result["XHINV"]
        - result["PHEXT"]
    ).abs()

    result["humidity_ratio"] = _safe_divide(
        result["XHINV"],
        result["PHEXT"],
    )

    # ------------------------------------------------------------------------
    # CO2 relationship
    # ------------------------------------------------------------------------

    if "co2_gradient" not in result.columns:
        result["co2_gradient"] = (
            result["XCO2I"]
            - result["PCO2EXT"]
        )

    result["co2_abs_gradient"] = (
        result["XCO2I"]
        - result["PCO2EXT"]
    ).abs()

    result["co2_ratio"] = _safe_divide(
        result["XCO2I"],
        result["PCO2EXT"],
    )

    # ------------------------------------------------------------------------
    # Ventilation relationship
    # ------------------------------------------------------------------------

    result["ventilation_mean"] = (
        result["UVENT_cen"]
        + result["UVENT_lN"]
    ) / 2.0

    result["ventilation_max"] = (
        result[
            ["UVENT_cen", "UVENT_lN"]
        ].max(axis=1)
    )

    result["ventilation_min"] = (
        result[
            ["UVENT_cen", "UVENT_lN"]
        ].min(axis=1)
    )

    result["ventilation_difference"] = (
        result["UVENT_cen"]
        - result["UVENT_lN"]
    )

    result[
        "ventilation_abs_difference"
    ] = (
        result["UVENT_cen"]
        - result["UVENT_lN"]
    ).abs()

    result["co2_per_ventilation"] = _safe_divide(
        result["XCO2I"],
        result["ventilation_mean"],
    )

    # ------------------------------------------------------------------------
    # CO2 + ventilation context
    # ------------------------------------------------------------------------

    result["high_co2"] = (
        result["XCO2I"] >= 700.0
    ).astype(np.float32)

    result["high_ventilation"] = (
        result["ventilation_mean"] >= 40.0
    ).astype(np.float32)

    result[
        "high_co2_high_ventilation"
    ] = (
        (result["XCO2I"] >= 700.0)
        & (result["ventilation_mean"] >= 40.0)
    ).astype(np.float32)

    # ------------------------------------------------------------------------
    # Radiation + temperature context
    # ------------------------------------------------------------------------

    result["high_radiation"] = (
        result["PRAD"] >= 400.0
    ).astype(np.float32)

    result[
        "high_external_temperature"
    ] = (
        result["PTEXT"] > 20.0
    ).astype(np.float32)

    result[
        "high_radiation_high_external_temp"
    ] = (
        (result["PRAD"] >= 400.0)
        & (result["PTEXT"] > 20.0)
    ).astype(np.float32)

    result[
        "temperature_inversion_context"
    ] = (
        (result["PRAD"] >= 400.0)
        & (result["PTEXT"] > 20.0)
        & (
            result["XTINV"]
            >= result["PTEXT"]
        )
    ).astype(np.float32)

    return result


# ============================================================================
# 4. EVENT-SHAPE FEATURES
# ============================================================================

def add_event_shape_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add local event-shape features.

    Designed mainly to distinguish:

        Ruido:
            normal → spike → normal

        Sensor Atascado:
            normal → constant → constant → constant
    """

    if sensors is None:
        sensors = TARGET_SENSORS

    result = df.copy()

    _validate_sensors(
        result,
        sensors,
    )

    for sensor in sensors:

        series = pd.to_numeric(
            result[sensor],
            errors="coerce",
        )

        prev_change = (
            series - series.shift(1)
        )

        next_change = (
            series.shift(-1) - series
        )

        abs_prev = prev_change.abs()
        abs_next = next_change.abs()

        # Difference between incoming and outgoing change.
        result[
            f"{sensor}_change_balance"
        ] = (
            abs_prev - abs_next
        ).abs()

        # Local variability.
        local_std = (
            series
            .rolling(
                window=5,
                center=True,
                min_periods=2,
            )
            .std()
        )

        denominator = local_std.replace(
            0,
            np.nan,
        )

        result[
            f"{sensor}_spike_score"
        ] = (
            (abs_prev + abs_next)
            / denominator
        )

        # Mean local change magnitude.
        result[
            f"{sensor}_local_change_mean"
        ] = (
            abs_prev + abs_next
        ) / 2.0

    return result


# ============================================================================
# 5. MISSINGNESS VALIDATION
# ============================================================================

def validate_missingness_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
) -> dict:
    """
    Validate missingness indicators created by the existing
    missing-value handler.

    These features are NOT created here.
    """

    if sensors is None:
        sensors = TARGET_SENSORS

    expected = [
        f"{sensor}_was_missing"
        for sensor in sensors
    ]

    expected.append(
        "any_sensor_was_missing"
    )

    present = [
        feature
        for feature in expected
        if feature in df.columns
    ]

    missing = [
        feature
        for feature in expected
        if feature not in df.columns
    ]

    return {
        "expected_features": len(expected),
        "present_features": len(present),
        "missing_features": missing,
        "all_features_present": len(missing) == 0,
    }


# ============================================================================
# 6. COMPLETE DIAGNOSTIC PIPELINE
# ============================================================================

def add_all_diagnostic_features(
    df: pd.DataFrame,
    temporal_rolling_window: int = 5,
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply all Model-2 diagnostic features.

    Input:
        DataFrame AFTER existing feature engineering
        and AFTER existing missing-value handling.

    Output:
        DataFrame ready for Model-2 scaling.
    """

    if sensors is None:
        sensors = TARGET_SENSORS

    result = df.copy()

    # 1. Temporal behavior.
    result = add_temporal_diagnostic_features(
        result,
        sensors=sensors,
        rolling_window=temporal_rolling_window,
    )

    # 2. Physical validity.
    result = add_physical_diagnostic_features(
        result,
        sensors=sensors,
    )

    # 3. Cross-sensor and contextual relationships.
    result = add_relationship_diagnostic_features(
        result,
    )

    # 4. Local event shape.
    result = add_event_shape_features(
        result,
        sensors=sensors,
    )

    return result


# ============================================================================
# 7. TEMPORAL FEATURE VALIDATION
# ============================================================================

def validate_temporal_diagnostic_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
) -> dict:
    """Validate the temporal diagnostic features."""

    if sensors is None:
        sensors = TARGET_SENSORS

    expected_features = []

    for sensor in sensors:
        expected_features.extend(
            [
                f"{sensor}_delta_prev",
                f"{sensor}_delta_next",
                f"{sensor}_abs_delta_prev",
                f"{sensor}_abs_delta_next",
                f"{sensor}_rolling_std",
                f"{sensor}_rolling_range",
            ]
        )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in df.columns
    ]

    created_features = [
        feature
        for feature in expected_features
        if feature in df.columns
    ]

    return {
        "expected_features": len(expected_features),
        "created_features": len(created_features),
        "missing_features": missing_features,
        "all_features_present": (
            len(missing_features) == 0
        ),
    }


# ============================================================================
# 8. COMPLETE FEATURE VALIDATION
# ============================================================================

def validate_diagnostic_features(
    df: pd.DataFrame,
    sensors: list[str] | None = None,
) -> dict:
    """
    Validate all diagnostic feature groups.
    """

    if sensors is None:
        sensors = TARGET_SENSORS

    # ------------------------------------------------------------------------
    # Temporal
    # ------------------------------------------------------------------------

    temporal = validate_temporal_diagnostic_features(
        df,
        sensors=sensors,
    )

    # ------------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------------

    missingness = validate_missingness_features(
        df,
        sensors=sensors,
    )

    # ------------------------------------------------------------------------
    # Physical
    # ------------------------------------------------------------------------

    physical_features = []

    for sensor in sensors:
        physical_features.extend(
            [
                f"{sensor}_is_out_of_physical_range",
                f"{sensor}_distance_to_min",
                f"{sensor}_distance_to_max",
                f"{sensor}_normalized_position",
            ]
        )

    # ------------------------------------------------------------------------
    # Relationship
    # ------------------------------------------------------------------------

    relationship_features = [
        "PRGINT_PRAD_ratio",
        "PRAD_minus_PRGINT",
        "PRGINT_minus_PRAD",
        "PRGINT_PRAD_relative_difference",
        "delta_temp",
        "temp_abs_difference",
        "temp_ratio",
        "interior_temp_below_exterior",
        "delta_hum",
        "humidity_abs_difference",
        "humidity_ratio",
        "co2_gradient",
        "co2_abs_gradient",
        "co2_ratio",
        "ventilation_mean",
        "ventilation_max",
        "ventilation_min",
        "ventilation_difference",
        "ventilation_abs_difference",
        "co2_per_ventilation",
        "high_co2",
        "high_ventilation",
        "high_co2_high_ventilation",
        "high_radiation",
        "high_external_temperature",
        "high_radiation_high_external_temp",
        "temperature_inversion_context",
    ]

    # ------------------------------------------------------------------------
    # Event shape
    # ------------------------------------------------------------------------

    event_shape_features = []

    for sensor in sensors:
        event_shape_features.extend(
            [
                f"{sensor}_change_balance",
                f"{sensor}_spike_score",
                f"{sensor}_local_change_mean",
            ]
        )

    def check_features(
        features: list[str],
    ) -> dict:

        missing = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        return {
            "expected": len(features),
            "present": (
                len(features) - len(missing)
            ),
            "missing": missing,
        }

    physical = check_features(
        physical_features
    )

    relationship = check_features(
        relationship_features
    )

    event_shape = check_features(
        event_shape_features
    )

    # ------------------------------------------------------------------------
    # Combined validation
    # ------------------------------------------------------------------------

    all_missing = (
        temporal["missing_features"]
        + missingness["missing_features"]
        + physical["missing"]
        + relationship["missing"]
        + event_shape["missing"]
    )

    return {
        "temporal": temporal,
        "missingness": missingness,
        "physical": physical,
        "relationship": relationship,
        "event_shape": event_shape,
        "total_missing_features": len(all_missing),
        "all_features_present": (
            len(all_missing) == 0
        ),
    }