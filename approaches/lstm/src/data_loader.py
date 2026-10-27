import re
import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    # Temperatura interior y exterior — Campbell HC2A-S3 (PLC1, PLC6)
    "TEMP_INTERIOR": (-40.0, 60.0),
    "TEMP_EXTERIOR": (-40.0, 60.0),
    # Temperatura de suelo — Campbell 109 termistor NTC (PLC1)
    "TEMP_SUELO":    (-50.0, 70.0),
    # CO2 — E+E Elektronik EE820-HV1A6E1 (PLC1, PLC6)
    "CO2":           (0.0, 2000.0),
    # Humedad relativa — Campbell HC2A-S3 (PLC1, PLC6)
    "HR":            (0.0, 100.0),
    # Radiación global — Campbell SP-110-SS / SP-214-SS (PLC1, PLC6)
    "RADGLOBAL":     (0.0, 2000.0),
    # Radiación PAR — Campbell SQ-204X-SS (PLC1, PLC6)
    "RADPAR":        (0.0, 700.0),
    # Velocidad del viento — Wittich & Visser PA2 (PLC6): 0..165 km/h = 0..45.8 m/s
    "VELVIENTO":     (0.0, 45.8),
    # Dirección del viento — Wittich & Visser PRV (PLC6)
    "DIRVIENTO":     (0.0, 360.0),
    # Lluvia — cualquier valor negativo es anomalía
    "LLUVIA":        (0.0, 200.0),
    # Posición ventanas (encoder) — De Gier I-DE, lineal 4-20 mA = 0-100% (PLC1, PLC2)
    "UV_POS":        (0.0, 100.0),
}

# Mapeo columna del paper → rango físico real
# Usado directamente en el sistema de detección de anomalías
COLUMN_PHYSICAL_RANGES: dict[str, tuple[float, float]] = {
    "PCO2EXT":   PHYSICAL_RANGES["CO2"],
    "PHEXT":     PHYSICAL_RANGES["HR"],
    "PRAD":      PHYSICAL_RANGES["RADGLOBAL"],
    "PRGINT":    PHYSICAL_RANGES["RADGLOBAL"],
    "PTEXT":     PHYSICAL_RANGES["TEMP_EXTERIOR"],
    "PVV":       PHYSICAL_RANGES["VELVIENTO"],
    "UVENT_cen": PHYSICAL_RANGES["UV_POS"],
    "UVENT_lN":  PHYSICAL_RANGES["UV_POS"],
    "XCO2I":     PHYSICAL_RANGES["CO2"],
    "XHINV":     PHYSICAL_RANGES["HR"],
    "XTINV":     PHYSICAL_RANGES["TEMP_INTERIOR"],
    "XTS":       PHYSICAL_RANGES["TEMP_SUELO"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _clean_column_name(raw: str) -> str:
    """Elimina paréntesis, espacios y caracteres raros del nombre de columna."""
    name = str(raw).strip()
    name = re.sub(r"[\(\)]", "", name)  # quita paréntesis
    name = name.strip()
    return name


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza cambios de nombre de columna entre versiones del firmware.

    Periodo oct-feb 2023/24 (54-62 cols):
      - Los actuadores venían con prefijo 'AUX_' (p.ej. AUX_ACTERMICO_BOMBA_S1)
    Periodo mar 2024+ (101-106 cols):
      - El prefijo 'AUX_' desaparece (p.ej. ACTERMICO_BOMBA_S1)

    Al renombrar aquí, el concat posterior produce una sola columna unificada
    en vez de dos columnas paralelas con la mitad de NaN.
    """
    rename_map = {col: col[4:] for col in df.columns if col.startswith("AUX_")}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _find_header_row(df_raw: pd.DataFrame) -> int:
    """Devuelve el índice (0-based) de la fila que contiene 'Fecha'/'FECHA' en col 0."""
    for i in range(min(5, len(df_raw))):
        cell = str(df_raw.iloc[i, 0]).strip().upper()
        if cell == "FECHA":
            return i
    raise ValueError("No se encontró la fila de cabecera con 'FECHA' en las primeras 5 filas.")


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Convierte la columna FECHA a datetime (UTC cuando hay offset).
    Formatos observados:
      - '10/10/2020 00:00:03'           → dayfirst, sin offset
      - '2024-01-15 12:00:00 +1H'        → resta 1h → UTC
      - '2024-04-07 00:14:00 +2H, DST'   → resta 2h → UTC
    Si no hay sufijo, se parsea tal cual (sin asumir zona).
    """
    raw = series.astype(str)
    # Extraer offset en horas (1 o 2); si no hay sufijo → 0
    offsets = raw.str.extract(r"\+(\d+)H", expand=False).fillna("0").astype(int)
    # Limpiar sufijo y parsear (dayfirst para DD/MM/YYYY del paper dataset)
    cleaned = raw.str.replace(r"\s*\+\d+H.*$", "", regex=True).str.strip()
    dt = pd.to_datetime(cleaned, errors="coerce", dayfirst=True)
    # Restar el offset para convertir a UTC
    dt = dt - pd.to_timedelta(offsets, unit="h")
    return dt


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte todas las columnas (excepto FECHA) a float64.
    Maneja el separador decimal europeo (coma → punto).
    """
    for col in df.columns:
        if col == "FECHA":
            continue
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .replace({"nan": "NaN", "None": "NaN", "": "NaN"})
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Función principal: cargar un único fichero
# ─────────────────────────────────────────────────────────────────────────────

def load_single_file(path: str | Path, sep: str = ";") -> pd.DataFrame:
    path = Path(path)

    # Leer sin cabecera para detectarla manualmente
    df_raw = pd.read_csv(path, header=None, dtype=str, sep=sep)

    # Encontrar y extraer cabecera
    header_idx = _find_header_row(df_raw)
    columns = [_clean_column_name(c) for c in df_raw.iloc[header_idx].tolist()]

    # Quedarnos solo con las filas de datos
    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = columns
    df = df.reset_index(drop=True)

    # Unificar nombre de timestamp (Fecha / FECHA → FECHA)
    fecha_cols = [c for c in df.columns if str(c).strip().upper() == "FECHA"]
    if not fecha_cols:
        raise ValueError(f"Columna de fecha no encontrada en {path.name}. Columnas: {list(df.columns)}")
    if fecha_cols[0] != "FECHA":
        df = df.rename(columns={fecha_cols[0]: "FECHA"})

    # Eliminar filas completamente vacías
    df = df.dropna(how="all")

    # Parsear timestamp
    df["FECHA"] = _parse_timestamp(df["FECHA"])
    df = df.dropna(subset=["FECHA"])

    # Convertir columnas numéricas
    df = _convert_numeric_columns(df)

    # Ordenar ascendente (los ficheros vienen en orden descendente)
    df = df.sort_values("FECHA").reset_index(drop=True)

    # Normalizar nombres de columna entre versiones del firmware (AUX_ prefix)
    df = _normalize_column_names(df)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Función principal: cargar todos los ficheros del directorio
# ─────────────────────────────────────────────────────────────────────────────

def load_all_files(
    dataset_dir: str | Path = "data/raw",
    pattern: str = "sensor_data_raw2.csv",
    sep: str = ";",
    exclude_copies: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron ficheros con el patrón '{pattern}' en '{dataset_dir}'"
        )

    if exclude_copies:
        files = [f for f in files if " - copia" not in f.name.lower()]

    logger.info(f"Cargando {len(files)} ficheros desde '{dataset_dir}'...")

    dfs = []
    errors = []

    iterator = tqdm(files, desc="Cargando CSV", unit="fichero") if show_progress else files

    for f in iterator:
        try:
            df = load_single_file(f, sep=sep)
            dfs.append(df)
        except Exception as e:
            errors.append((f.name, str(e)))
            logger.warning(f"Error en {f.name}: {e}")

    if errors:
        logger.warning(f"\n{len(errors)} ficheros con errores: {[e[0] for e in errors]}")

    if not dfs:
        raise RuntimeError("No se pudo cargar ningún fichero correctamente.")

    # Concatenar con outer join → columnas nuevas rellenadas con NaN
    df_all = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    # Ordenar por timestamp
    df_all = df_all.sort_values("FECHA").reset_index(drop=True)

    # Eliminar duplicados exactos de timestamp (mismo instante, mismos valores)
    n_before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["FECHA"], keep="first")
    n_removed = n_before - len(df_all)
    if n_removed:
        logger.info(f"Eliminados {n_removed} timestamps duplicados.")

    logger.info(
        f"Dataset unificado: {len(df_all):,} filas × {len(df_all.columns)} columnas. "
        f"Rango: {df_all['FECHA'].min()} → {df_all['FECHA'].max()}"
    )

    return df_all

# ─────────────────────────────────────────────────────────────────────────────
# I/O de datos procesados
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, output_path: str | Path = "data/processed/clean_data.parquet") -> None:
    """Guarda el DataFrame en formato Parquet (rápido y comprimido)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Dataset guardado en '{output_path}' ({size_mb:.1f} MB)")


def load_processed(path: str | Path = "data/processed/clean_data.parquet") -> pd.DataFrame:
    """Carga el dataset procesado desde Parquet."""
    return pd.read_parquet(path, engine="pyarrow")



def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen estadístico del dataset: tipo, nulos, rango temporal, min/max/media.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary = df[numeric_cols].describe().T
    summary["null_count"] = df[numeric_cols].isnull().sum()
    summary["null_pct"] = (summary["null_count"] / len(df) * 100).round(2)
    return summary