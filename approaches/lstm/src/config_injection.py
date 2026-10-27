"""
config_injection.py - Adapted Anomaly Injection Configuration for CIESOL
=========================================================================
Project: CIESOL Sensor Anomaly Detection
Dataset: CIESOL Greenhouse Environmental Sensors (~2-month subset: Oct 2020 - Dec 2020)

Methodological Alignment & Workflow:
------------------------------------
1. Preprocessing and chronological split are already completed upstream:
   ~60 days Train / ~10 days Validation / ~10 days Test.
2. Ingestion for Anomaly Injection:
   - Input MUST be the raw physical/unscaled Train dataframe (`train_unscaled.parquet`).
   - Injection is performed on PHYSICAL engineering units (ppm, °C, W/m2, %, m/s).
3. Injection Target:
   - Synthetic anomalies are generated EXCLUSIVELY in TRAIN.
   - Validation and Test splits remain completely CLEAN and untouched.
4. Downstream Scaling:
   - clean physical Train -> anomaly injection -> injected physical Train -> scaling.
   - Scaler is fitted on CLEAN physical Train only, then transforms injected Train, Val, and Test.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# 1. PROJECT PATHS & DIRECTORIES
# =============================================================================

# Root directory of the project (/home/jetson/Desktop/ciesol-anomaly-detection)
_CURRENT_PATH = Path(__file__).resolve()
BASE_DIR = _CURRENT_PATH.parent if _CURRENT_PATH.parent.name != "src" else _CURRENT_PATH.parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_INJECTED_DIR = DATA_DIR / "injected"
DATA_QUALITY_DIR = DATA_DIR / "quality"

# Ensure output directory for injected artifacts exists
DATA_INJECTED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. DATASET INPUTS & INJECTION POLICIES
# =============================================================================

# Nominal sampling characteristics (same sensor system)
NOMINAL_SAMPLING_SECONDS = 30
SAMPLES_PER_MINUTE = 2

# Input preprocessed split paths:
# Injection MUST operate on the unscaled physical Train split.
TRAIN_UNSCALED_PATH = DATA_PROCESSED_DIR / "train_unscaled.parquet"
VAL_UNSCALED_PATH = DATA_PROCESSED_DIR / "validation_unscaled.parquet"
TEST_UNSCALED_PATH = DATA_PROCESSED_DIR / "test_unscaled.parquet"
PREPROCESSED_FULL_PATH = DATA_PROCESSED_DIR / "preprocessed_data.parquet"

# Explicit injection input pointer
INJECTION_INPUT_PATH = TRAIN_UNSCALED_PATH

# Synthetic anomalies are generated only in TRAIN.
# Validation and TEST remain clean to provide unbiased evaluation data.
INJECT_ONLY_INTO_TRAIN = True
INJECT_VALIDATION = False
INJECT_TEST = False

# Overlap policy:
# In the professor's methodology, synthetic anomaly injections target clean,
# non-overlapping rows/segments to avoid confounding multi-anomaly artifacts.
ALLOW_ANOMALY_OVERLAP = False


# =============================================================================
# 3. SENSOR SCOPE & ROLES
# =============================================================================

# Complete set of sensor columns present in the dataset (12 columns):
ALL_DATASET_SENSORS: List[str] = [
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

# Professor: Target injection sensors included 9 primary variables:
#   ['PCO2EXT', 'PHEXT', 'PRAD', 'PRGINT', 'PTEXT', 'PVV', 'XCO2I', 'XHINV', 'XTINV']
# Adapted: Preserved the exact same 9 target sensors.
# Note: XTS is present in the dataset but was excluded from feature/injection targets
# by the professor; UVENT_cen and UVENT_lN serve exclusively as contextual reference.
SENSORES_INYECCION: List[str] = [
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

# Actuator / Contextual reference variables (used to evaluate state, not direct targets)
COLUMNAS_VENTILACION: List[str] = ["UVENT_cen", "UVENT_lN"]

# Professor: PRAD was excluded from Stuck Sensor injection because flat diurnal peaks
# and nighttime zeros are normal physical solar behavior, causing false positives.
# Adapted: Preserved the same 8 eligible stuck-sensor targets.
SENSORES_STUCK: List[str] = [
    "PCO2EXT",
    "PHEXT",
    "PRGINT",
    "PTEXT",
    "PVV",
    "XCO2I",
    "XHINV",
    "XTINV",
]


# =============================================================================
# 4. PHYSICAL SENSOR RANGES
# =============================================================================

# Physical/operational ranges inherited from the professor's anomaly-injection
# configuration for the same underlying dataset.
# Values outside [min, max] represent physically invalid or hardware-fault readings.
# Note: Ranges are NOT narrowed for the 2-month subset; physical limits remain invariant.

RANGOS_FISICOS: Dict[str, Dict[str, Any]] = {
    # Exterior CO2 concentration (ppm)
    "PCO2EXT":   {"min": 300.0, "max": 800.0, "unit": "ppm"},
    # Exterior relative humidity (%)
    "PHEXT":     {"min": 0.0,   "max": 100.0, "unit": "%"},
    # Exterior global solar radiation (W/m2)
    "PRAD":      {"min": 0.0,   "max": 1500.0, "unit": "W/m2"},
    # Interior global solar radiation (W/m2)
    "PRGINT":    {"min": 0.0,   "max": 1200.0, "unit": "W/m2"},
    # Exterior air temperature (°C)
    "PTEXT":     {"min": -20.0, "max": 60.0, "unit": "°C"},
    # Exterior wind speed (m/s)
    "PVV":       {"min": 0.0,   "max": 50.0, "unit": "m/s"},
    # Interior CO2 concentration (ppm)
    "XCO2I":     {"min": 200.0, "max": 3000.0, "unit": "ppm"},
    # Interior relative humidity (%)
    "XHINV":     {"min": 0.0,   "max": 100.0, "unit": "%"},
    # Interior air temperature (°C)
    "XTINV":     {"min": -5.0,  "max": 60.0, "unit": "°C"},
    # Reference ranges for non-target dataset columns (for audit/checks):
    "UVENT_cen": {"min": 0.0,   "max": 100.0, "unit": "%"},
    "UVENT_lN":  {"min": 0.0,   "max": 100.0, "unit": "%"},
    "XTS":       {"min": -10.0, "max": 60.0, "unit": "°C"},
}


# =============================================================================
# 5. ANOMALY TYPES & LABELS
# =============================================================================

ETIQUETA_NORMAL = "normal"
ETIQUETA_ANOMALIA = "anomalia"

TIPOS_ANOMALIA: List[str] = [
    "Caida Sistema",
    "Fallo Parcial Sistema",
    "Datos Faltantes",
    "Sensor Atascado",
    "Ruido",
    "Valores Fuera de Rango",
    "Desviacion de Correlacion",
    "Contextual",
]


# =============================================================================
# 6. SYSTEM-LEVEL FAILURES (Deterministic / Infrastructure)
# =============================================================================

# System-level power/bus outage where all or most target variables drop simultaneously.
# Physical thresholds are preserved from the professor's configuration.

CONFIG_CAIDA_SISTEMA: Dict[str, Any] = {
    # Professor: 5.0 min minimum outage duration.
    # Adapted: 5.0 min (10 samples at 30s) preserved.
    # Reason: Physical definition of system failure is scale-invariant.
    "min_duracion_min": 5.0,
    # Professor: ~25 infrastructure outages over ~535 train days.
    # Adapted: 3 events for ~60 days (proportional scaling: 25 * 60 / 535 ≈ 2.8).
    # Reason: Provide sufficient training representation without dominating normal operational state.
    "num_eventos_train": 3,
}

CONFIG_FALLO_PARCIAL: Dict[str, Any] = {
    # Professor: >= 4 sensors simultaneous NaN or 0 for >= 5.0 min.
    # Adapted: Preserved identical criteria.
    # Reason: Multi-channel sensor bus failure definition is independent of time period.
    "min_sensores_nan": 4,
    "min_sensores_cero": 4,
    "min_duracion_min": 5.0,
    # Professor: ~30 partial system failures over ~535 train days.
    # Adapted: 4 events for ~60 days (proportional scaling: 30 * 60 / 535 ≈ 3.4).
    # Reason: Proportional scaling to ~60-day training duration.
    "num_eventos_train": 4,
}


# =============================================================================
# 7. MISSING DATA (DATOS FALTANTES) - PERCENTAGE-BASED
# =============================================================================

# Professor methodology: Row-percentage selection ('porcentaje_filas': 0.02).
# Mechanism: Pointwise row selection on active operational periods, avoiding naturally zero states.

CONFIG_DATOS_FALTANTES: Dict[str, Any] = {
    # Professor: 0.02 (2% of rows) with 1 sensor per affected row.
    # Adapted: 0.02 preserved (corresponds to ~3,457 injected missing cells across 172,873 train rows).
    # Reason: Relative percentage is scale-invariant and maintains identical missing rate.
    "porcentaje_filas": 0.02,
    "num_sensores_por_fila": 1,
    # Baseline constraints inherited from professor: avoid injecting NaN during natural zeros
    "valor_minimo_inyeccion": {
        "PVV": 3.0,     # Only inject NaN when real wind speed > 3 m/s (0 = natural calm)
        "PRGINT": 50.0, # Only inject NaN during daytime solar radiation (0 = night)
    },
}


# =============================================================================
# 8. STUCK SENSOR (SENSOR ATASCADO) - QUOTA-BASED SEQUENCES
# =============================================================================

# Professor methodology: Explicit per-sensor sequence quotas with sensor-specific
# duration windows derived from natural plateau analysis on CIESOL.
#
# Scaling rationale:
# - Professor train quota: 40 sequences per sensor over ~535 training days.
# - My training duration: ~60 days (~11.2% of professor's training days).
# - Proportional scaling: 40 * (60 / 535) = 4.49 ≈ 4 sequences per sensor.
# - Total across 8 eligible sensors: 8 * 4 = 32 sequences.

CONFIG_SENSOR_ATASCADO: Dict[str, Any] = {
    # Professor: 40 sequences/sensor in train split (total 320 across 8 sensors over ~535 days).
    # Adapted: 4 sequences/sensor in train split (total 32 across 8 eligible sensors over ~60 days).
    # Reason: Proportional temporal scaling (40 * 60 / 535 ≈ 4.5).
    "secuencias_por_sensor_train": {
        "PCO2EXT": 4,
        "PHEXT":   4,
        "PRGINT":  4,
        "PTEXT":   4,
        "PVV":     4,
        "XCO2I":   4,
        "XHINV":   4,
        "XTINV":   4,
    },
    "num_secuencias_total_train": 32,
    # Sensor-specific duration windows in samples (at 30s sampling rate):
    # Inherited directly from professor's natural plateau analysis on CIESOL
    "duracion_por_sensor": {
        "PCO2EXT": {"duracion_min": 20, "duracion_max": 40},  # 10 to 20 min
        "PTEXT":   {"duracion_min": 16, "duracion_max": 40},  # 8 to 20 min
        "XTINV":   {"duracion_min": 20, "duracion_max": 60},  # 10 to 30 min
        "XHINV":   {"duracion_min": 28, "duracion_max": 70},  # 14 to 35 min
        "PHEXT":   {"duracion_min": 32, "duracion_max": 80},  # 16 to 40 min
        "PVV":     {"duracion_min": 20, "duracion_max": 60},  # 10 to 30 min
        "XCO2I":   {"duracion_min": 20, "duracion_max": 60},  # 10 to 30 min
        "PRGINT":  {"duracion_min": 20, "duracion_max": 60},  # 10 to 30 min
    },
    # Minimum sigma separation between stuck value and injection window mean
    # Prevents degenerate plateau matching the real signal
    "min_sigma_separacion": 0.5,
    # Contextual activity filters before allowing stuck injection
    "filtros_contextuales": {
        "PRGINT": {"ref_col": "PRAD",  "ref_min": 5.0,  "ref_variabilidad_min": 2.0},
        "XTINV":  {"ref_col": "PTEXT", "ref_min": 5.0,  "ref_variabilidad_min": 0.1},
        "XHINV":  {"ref_col": "XTINV", "ref_min": 10.0, "ref_variabilidad_min": 0.5},
        "PTEXT":  {"ref_col": "PRAD",  "ref_min": 10.0, "ref_variabilidad_min": 0.0},
    },
    # Previous activity threshold requirements
    "actividad_previa": {
        "XTINV":  {"ventana": 10, "std_min": 0.20},
        "PTEXT":  {"ventana": 20, "std_min": 0.20},
        "XHINV":  {"ventana": 10, "std_min": 0.50},
        "PRGINT": {"ventana": 10, "std_min": 2.00},
    },
    "valor_minimo_inyeccion": {
        "PVV": 0.5,
        "PRGINT": 10.0,
    },
}


# =============================================================================
# 9. NOISE / SPIKES (RUIDO) - PERCENTAGE-BASED
# =============================================================================

# Professor methodology: Row-level percentage (0.03) with 1 sensor/row and std factor in [3, 6],
# combined with sensor-specific delta magnitudes and polarity constraints.

CONFIG_RUIDO: Dict[str, Any] = {
    # Professor: 0.03 (3% of rows) with factor_std in [3, 6] and 1 sensor per row.
    # Adapted: 0.03 preserved (~5,186 rows affected in 60-day train split).
    # Reason: Scale-invariant percentage maintains consistent noise prevalence without competing quotas.
    "porcentaje_filas": 0.03,
    "factor_std_min": 3.0,
    "factor_std_max": 6.0,
    "num_sensores_por_fila": 1,
    # Delta magnitude bounds per sensor (inherited from professor)
    "deltas_por_sensor": {
        "PTEXT":   {"min": 3.0,   "max": 8.0,   "signos": [-1, 1]},
        "XTINV":   {"min": 3.0,   "max": 8.0,   "signos": [-1, 1]},
        "PHEXT":   {"min": 10.0,  "max": 25.0,  "signos": [-1, 1]},
        "XHINV":   {"min": 10.0,  "max": 25.0,  "signos": [-1, 1]},
        "PCO2EXT": {"min": 25.0,  "max": 75.0,  "signos": [-1, 1]},
        "XCO2I":   {"min": 60.0,  "max": 160.0, "signos": [-1, 1]},
        "PVV":     {"min": 5.0,   "max": 9.0,   "signos": [1]},     # Only positive spikes (calm is natural)
        "PRAD":    {"min": 150.0, "max": 500.0, "signos": [1]},     # Only positive spikes (negative spike = cloud)
        "PRGINT":  {"min": 80.0,  "max": 250.0, "signos": [1]},     # Only positive spikes
    },
    # Polarities and physical minimum baselines inherited from professor
    "solo_positivo": ["PVV", "PRGINT", "PRAD"],
    "valor_minimo_inyeccion": {
        "PRGINT": 100.0,
        "PVV": 5.0,
        "PCO2EXT": 400.0,
        "PRAD": 50.0,
    },
}


# =============================================================================
# 10. OUT-OF-RANGE VALUES (VALORES FUERA DE RANGO) - PERCENTAGE-BASED
# =============================================================================

# Professor methodology: Row percentage (0.02) injecting values beyond physical limits
# using a margin error factor (0.10).

CONFIG_FUERA_DE_RANGO: Dict[str, Any] = {
    # Professor: 0.02 (2% of rows) with 1 sensor per row and margin factor 0.1.
    # Adapted: 0.02 preserved (~3,457 rows affected in 60-day train split).
    # Reason: Scale-invariant percentage methodology without competing quotas.
    "porcentaje_filas": 0.02,
    "num_sensores_por_fila": 1,
    # Injects value exceeding [min, max] by at least 10% of physical span
    "margen_error_factor": 0.10,
}


# =============================================================================
# 11. CORRELATION DEVIATION (DESVIACION DE CORRELACION) - PERCENTAGE-BASED
# =============================================================================

# Meaningful physical pairs identified in professor's methodology and supported by EDA:
#   ('PRAD',  'PRGINT') -> Exterior vs Interior Solar Radiation
#   ('XTINV', 'XHINV')  -> Interior Temperature vs Relative Humidity (psychrometric coupling)
#   ('PTEXT', 'XTINV')  -> Exterior vs Interior Temperature (thermal envelope)
#   ('PHEXT', 'XHINV')  -> Exterior vs Interior Relative Humidity (hygric envelope)
#
# Note: UVENT pairs were explicitly removed by the professor because windows are closed (0%)
# most of the time, causing degenerate percentile distributions (p25 = p75 = 0).

PARES_CORRELACION: List[Tuple[str, str]] = [
    ("PRAD",  "PRGINT"),
    ("XTINV", "XHINV"),
    ("PTEXT", "XTINV"),
    ("PHEXT", "XHINV"),
]

CONFIG_DESVIACION_CORRELACION: Dict[str, Any] = {
    # Professor: 0.03 (3% of daytime rows) between 07:00 and 18:00.
    # Adapted: 0.03 preserved for daytime active hours [7, 18].
    # Reason: Preserves temporal daytime filter and relative occurrence rate without competing quotas.
    "porcentaje_filas": 0.03,
    "filtro_hora_inicio": 7,
    "filtro_hora_fin": 18,
    "pares": PARES_CORRELACION,
}


# =============================================================================
# 12. CONTEXTUAL ANOMALIES (ANOMALIAS CONTEXTUALES)
# =============================================================================

# Three physically grounded contextual anomalies defined by the professor:
# 1. PRAD/PRGINT: Solar radiation high outside but anomalously low inside (dirty/blocked cover).
# 2. CO2/Ventilation: CO2 elevated inside while windows are open (actuator/sensor discrepancy).
# 3. Temperature Inversion: Interior cooler than exterior under high solar radiation.

CONFIG_CONTEXTUAL: Dict[str, Any] = {
    # --- Contextual Type 1: Solar Cover Transmittance Obstruction (Percentage-based) ---
    # Professor: porcentaje_filas: 0.020, prad_min: 200.0, ratio_normal_min: 0.40, ratio_anomalo_max: 0.08
    # Adapted: Preserved physical conditions and percentage.
    # Reason: Physical transmittance properties of greenhouse glazing are identical.
    "prad_prgint": {
        "porcentaje_filas": 0.020,
        "prad_min": 200.0,          # Clear daytime solar radiation
        "ratio_normal_min": 0.40,   # Baseline normal transmittance > 40%
        "ratio_anomalo_max": 0.08,  # Faulty injected transmittance = 2-8%
    },

    # --- Contextual Type 2: High CO2 with Open Ventilation (Sequence Quota-based) ---
    # Professor: 100 sequences in train split over ~535 days.
    # Adapted: 11 sequences in ~60-day train split.
    # Reason: Proportional temporal scaling (100 * 60 / 535 = 11.21 ≈ 11 sequences).
    "co2_vent": {
        # Professor: 100 sequences over ~535 train days.
        # Adapted: 11 sequences for ~60 train days.
        # Reason: Proportional scaling (100 * 60 / 535 ≈ 11.2).
        "num_secuencias_train": 11,
        "umbral_vent_abierta": 40.0, # Ventilation aperture > 40%
        "prad_min_diurno": 50.0,     # Daytime condition (>50 W/m2)
        "co2_anomalo_min": 700.0,    # Faulty elevated CO2 (700-1200 ppm)
        "co2_anomalo_max": 1200.0,
        "duracion_min": 3,          # 3 to 8 samples (1.5 to 4.0 min)
        "duracion_max": 8,
    },

    # --- Contextual Type 3: Inverted Temperature under High Insolation (Percentage-based) ---
    # Professor: porcentaje_filas: 0.016, prad_min: 400.0, ptext_min: 20.0, delta: 3-10 °C
    # Adapted: Preserved physical conditions and percentage.
    # Reason: Greenhouse thermodynamics under solar gain are identical.
    "temp_inv": {
        "porcentaje_filas": 0.016,
        "prad_min": 400.0,          # Strong insolation (>400 W/m2)
        "ptext_min": 20.0,          # Warm exterior (>20 °C)
        "delta_min": 3.0,           # XTINV injected as PTEXT - (3 to 10 °C)
        "delta_max": 10.0,
    },
}


# =============================================================================
# 13. REPRODUCIBILITY & RANDOM SEED
# =============================================================================

# Fixed seed for deterministic, fully reproducible synthetic anomaly injection
RANDOM_SEED: int = 42


# =============================================================================
# 14. OUTPUT PATHS FOR INJECTED ARTIFACTS
# =============================================================================

# Output files generated during the injection stage (under data/injected/)
PARQUET_TRAIN_INJECTED_PHYSICAL = DATA_INJECTED_DIR / "train_injected_physical.parquet"
CSV_INJECTION_SUMMARY = DATA_INJECTED_DIR / "injection_summary.csv"
CSV_INJECTION_EVENTS = DATA_INJECTED_DIR / "injection_events.csv"
CSV_INJECTION_AUDIT = DATA_INJECTED_DIR / "injection_audit_by_sensor.csv"


# =============================================================================
# 15. CONFIGURATION SUMMARY & INSPECTION
# =============================================================================

def get_config_summary() -> Dict[str, Any]:
    """Returns a consolidated summary of key injection parameters."""
    return {
        "base_dir": str(BASE_DIR),
        "injection_input_path": str(INJECTION_INPUT_PATH),
        "inject_only_train": INJECT_ONLY_INTO_TRAIN,
        "inject_validation": INJECT_VALIDATION,
        "inject_test": INJECT_TEST,
        "allow_anomaly_overlap": ALLOW_ANOMALY_OVERLAP,
        "random_seed": RANDOM_SEED,
        "n_target_sensors": len(SENSORES_INYECCION),
        "target_sensors": SENSORES_INYECCION,
        "contextual_sensors": COLUMNAS_VENTILACION,
        "stuck_sensors": SENSORES_STUCK,
        "stuck_sequences_train": CONFIG_SENSOR_ATASCADO["num_secuencias_total_train"],
        "co2_vent_sequences_train": CONFIG_CONTEXTUAL["co2_vent"]["num_secuencias_train"],
        "system_outage_events_train": CONFIG_CAIDA_SISTEMA["num_eventos_train"],
        "partial_outage_events_train": CONFIG_FALLO_PARCIAL["num_eventos_train"],
        "missing_data_row_pct": CONFIG_DATOS_FALTANTES["porcentaje_filas"],
        "noise_row_pct": CONFIG_RUIDO["porcentaje_filas"],
        "out_of_range_row_pct": CONFIG_FUERA_DE_RANGO["porcentaje_filas"],
        "correlation_pairs": PARES_CORRELACION,
    }


# =============================================================================
# 16. VALIDATION FUNCTION
# =============================================================================

def validate_config() -> bool:
    """
    Validates internal consistency and integrity of the configuration.
    Raises AssertionError if any check fails.
    """
    # 1. Project directories and unscaled input check
    assert BASE_DIR.exists(), f"BASE_DIR does not exist: {BASE_DIR}"
    assert DATA_DIR.exists(), f"DATA_DIR does not exist: {DATA_DIR}"
    assert DATA_PROCESSED_DIR.exists(), f"DATA_PROCESSED_DIR does not exist: {DATA_PROCESSED_DIR}"
    assert DATA_INJECTED_DIR.exists(), f"DATA_INJECTED_DIR does not exist: {DATA_INJECTED_DIR}"
    assert INJECTION_INPUT_PATH.exists(), f"Injection input file does not exist: {INJECTION_INPUT_PATH}"
    assert "unscaled" in str(INJECTION_INPUT_PATH), "Injection input MUST be unscaled physical data"

    # 2. Sensor names check
    for s in SENSORES_INYECCION:
        assert s in ALL_DATASET_SENSORS, f"Target sensor '{s}' not in ALL_DATASET_SENSORS"
    for s in SENSORES_STUCK:
        assert s in SENSORES_INYECCION, f"Stuck sensor '{s}' not in SENSORES_INYECCION"
    for s in COLUMNAS_VENTILACION:
        assert s in ALL_DATASET_SENSORS, f"Ventilation sensor '{s}' not in ALL_DATASET_SENSORS"
        assert s not in SENSORES_INYECCION, f"Ventilation sensor '{s}' should not be in SENSORES_INYECCION"

    # 3. Correlation pairs check
    for s1, s2 in PARES_CORRELACION:
        assert s1 in SENSORES_INYECCION, f"Pair sensor '{s1}' not in SENSORES_INYECCION"
        assert s2 in SENSORES_INYECCION, f"Pair sensor '{s2}' not in SENSORES_INYECCION"

    # 4. Physical ranges check
    for s in SENSORES_INYECCION:
        assert s in RANGOS_FISICOS, f"Sensor '{s}' missing from RANGOS_FISICOS"
        r = RANGOS_FISICOS[s]
        assert r["min"] < r["max"], f"Invalid physical range for '{s}': min {r['min']} >= max {r['max']}"

    # 5. Percentages check (must be strictly in (0, 1))
    pct_params = [
        ("datos_faltantes", CONFIG_DATOS_FALTANTES["porcentaje_filas"]),
        ("ruido", CONFIG_RUIDO["porcentaje_filas"]),
        ("fuera_rango", CONFIG_FUERA_DE_RANGO["porcentaje_filas"]),
        ("corr_dev", CONFIG_DESVIACION_CORRELACION["porcentaje_filas"]),
        ("ctx_prad_prgint", CONFIG_CONTEXTUAL["prad_prgint"]["porcentaje_filas"]),
        ("ctx_temp_inv", CONFIG_CONTEXTUAL["temp_inv"]["porcentaje_filas"]),
    ]
    for name, p in pct_params:
        assert 0.0 < p < 1.0, f"Percentage for {name} ({p}) must be in range (0, 1)"

    # 6. Durations check (must be positive)
    assert CONFIG_CAIDA_SISTEMA["min_duracion_min"] > 0
    assert CONFIG_FALLO_PARCIAL["min_duracion_min"] > 0
    for s, d in CONFIG_SENSOR_ATASCADO["duracion_por_sensor"].items():
        assert d["duracion_min"] > 0, f"duracion_min for {s} must be > 0"
        assert d["duracion_max"] >= d["duracion_min"], f"duracion_max < duracion_min for {s}"
    assert CONFIG_CONTEXTUAL["co2_vent"]["duracion_min"] > 0
    assert CONFIG_CONTEXTUAL["co2_vent"]["duracion_max"] >= CONFIG_CONTEXTUAL["co2_vent"]["duracion_min"]

    # 7. Train-only injection enforcement
    assert INJECT_ONLY_INTO_TRAIN is True, "INJECT_ONLY_INTO_TRAIN must be True"
    assert INJECT_VALIDATION is False, "INJECT_VALIDATION must be False"
    assert INJECT_TEST is False, "INJECT_TEST must be False"

    # 8. Overlap policy enforcement
    assert ALLOW_ANOMALY_OVERLAP is False, "ALLOW_ANOMALY_OVERLAP must be False"

    # 9. Reproducibility seed check
    assert isinstance(RANDOM_SEED, int) and RANDOM_SEED >= 0, "RANDOM_SEED must be non-negative integer"

    # 10. Stuck sensor quota consistency check
    stuck_sum = sum(CONFIG_SENSOR_ATASCADO["secuencias_por_sensor_train"].values())
    assert stuck_sum == CONFIG_SENSOR_ATASCADO["num_secuencias_total_train"], (
        f"Sum of stuck sequences ({stuck_sum}) != total ({CONFIG_SENSOR_ATASCADO['num_secuencias_total_train']})"
    )

    return True


if __name__ == "__main__":
    print("Validating configuration...")
    if validate_config():
        print("✓ All configuration validation checks passed successfully!\n")
    print("=== Configuration Summary ===")
    for k, v in get_config_summary().items():
        print(f"  {k}: {v}")
