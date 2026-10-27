"""Frozen public contract and paths for the six-anomaly experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATASET = PROJECT.parent / "Dataset" / "Datos2MesesLimpios" / "sensor_data_raw2.csv"
RUNS = ROOT / "data" / "runs"

CLASSES = (
    "Contextual",
    "Ruido",
    "Desviacion de Correlacion",
    "Datos Faltantes",
    "Valores Fuera de Rango",
    "Sensor Atascado",
)
ALIASES = {
    "Desviación de Correlación": "Desviacion de Correlacion",
    "Desviación de correlación": "Desviacion de Correlacion",
    "Valor Fuera de Rango": "Valores Fuera de Rango",
    "Caída Sistema": "Datos Faltantes",
    "Caida Sistema": "Datos Faltantes",
    "Fallo Parcial Sistema": "Datos Faltantes",
}
SCHEMA = "six-anomaly-two-stage-v1"
MASK_COLUMNS = tuple(f"ausente_{s}" for s in (
    "PCO2EXT", "PHEXT", "PRAD", "PRGINT", "PTEXT", "PVV", "XCO2I", "XHINV", "XTINV"))

# Reuse the compact causal v8 feature selection as a fixed feature schema.
# Both models are fitted anew; invalid causal features use a fixed zero fill.
BASE_FEATURES = (
    "PVV_abs_diff1", "XCO2I_abs_diff1", "PCO2EXT_abs_diff1",
    "diff_rel_XTINV_PTEXT", "PTEXT_isna", "PRGINT_isna", "XHINV_isna",
    "XTINV_isna", "PRAD_isna", "PHEXT_isna", "diff_rel_PRAD_PRGINT",
    "PCO2EXT_rstd_3h", "diff_abs_XTINV_PTEXT", "diff_abs_PRAD_PRGINT",
    "delta_temp", "XTINV_rolling_std_20", "PHEXT_abs_diff1",
    "diff_abs_XHINV_PHEXT", "PRAD_diurnal_residual", "cos_hour",
    "PRGINT_rstd_3h", "PRAD_rstd_3h", "PHEXT_rstd_3h",
    "PVV_rolling_range_20", "diff_abs_XCO2I_PCO2EXT",
    "XTINV_rstd_15m", "PVV_rmean_3h", "PCO2EXT_rolling_range_20",
    "PVV_rstd_30m", "XTINV_rstd_5m", "XCO2I_rolling_std_20",
    "XCO2I_rstd_3h", "PCO2EXT_rmean_3h", "PVV_rstd_15m",
    "PHEXT_rolling_range_20", "XCO2I_diurnal_residual",
    "PRAD_rolling_range_20", "PRGINT_rstd_30m", "PRAD_rstd_30m",
    "XHINV_rstd_3h", "PCO2EXT_rstd_30m", "PHEXT_rolling_std_20",
    "PHEXT_rstd_30m", "XHINV_zscore_local", "PTEXT_rstd_30m",
    "ratio_rad", "PCO2EXT_diurnal_residual", "delta_hum",
    "XHINV_rolling_std_20", "PHEXT_rstd_5m",
)

def canonical(label: str) -> str:
    value = ALIASES.get(str(label), str(label))
    if value != "Normal" and value not in CLASSES:
        raise ValueError(f"Unknown anomaly type: {label}")
    return value
