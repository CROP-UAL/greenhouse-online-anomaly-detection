"""
features.py — Módulo de Carga, Split Temporal, Regularización y Feature Engineering (Pre-inyección)
====================================================================================================
Estructura las series temporales a un paso regular de 30s y calcula variables derivadas
(cíclicas sin/cos, diferencias temporales, estadísticas en ventana móvil, features de
estatismo para Sensor Atascado) ANTES de la inyección.

GARANTÍA CAUSAL: todas las operaciones solo usan la fila actual y el pasado;
las ventanas rolling usan únicamente muestras previas (closed='left' implícito en
rolling de índice entero).

Split temporal estricto (81 días completos):
  df_warmup : Día 1        — buffer de calentamiento sin NaN
  df_train  : Días 2–61   — entrenamiento
  df_val    : Días 62–71  — validación (early stopping y calibración umbral)
  df_test   : Días 72–81  — prueba final (evaluación no vista)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y REGULARIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def _safe_divide(numerator, denominator, *, epsilon: float = 1e-6):
    """Division policy for batch reference features: invalid denominator -> 0."""
    num = np.asarray(numerator, np.float64)
    den = np.asarray(denominator, np.float64)
    output = np.zeros(np.broadcast_shapes(num.shape, den.shape), np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > epsilon)
    np.divide(num, den, out=output, where=valid)
    return output


def cargar_y_regularizar_datos(filepath: str, sampling_seconds: int = 30) -> pd.DataFrame:
    """
    Carga el dataset CSV de 2 meses, parsea fechas y regulariza el índice a un paso fijo.
    El resampling usa la media y conserva los huecos; el extractor streaming\n    registra la ausencia antes de imputar causalmente con valores pasados.
    """
    print(f"Cargando dataset desde: {filepath}")
    df = pd.read_csv(filepath, sep=';')

    # Normalizar nombre de fecha y parsear
    col_fecha = 'Fecha' if 'Fecha' in df.columns else df.columns[0]
    df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True)
    df = df.sort_values(by=col_fecha).reset_index(drop=True)

    # Regularización temporal a intervalo de 30s
    df = df.set_index(col_fecha)
    freq_str = f"{sampling_seconds}s"
    # Causal: no backfill desde el futuro.
    # Preserve missing bins and raw NaNs. Never erase the missingness evidence
    # before the streaming extractor creates each sensor's *_isna feature.
    usable = None
    if 'usar_en_train' in df.columns:
        source = df.pop('usar_en_train')
        usable = source.astype(str).str.lower().isin(('true', '1'))
    df_resampled = df.resample(freq_str).mean().reset_index()
    if usable is not None:
        df_resampled['usar_en_train'] = (
            usable.resample(freq_str).last().reindex(
                pd.DatetimeIndex(df_resampled[col_fecha])).fillna(False).to_numpy(bool)
        )

    print(
        f"Dataset regularizado a {sampling_seconds}s: {df_resampled.shape[0]:,} filas | "
        f"Rango: {df_resampled[col_fecha].min()} → {df_resampled[col_fecha].max()}"
    )
    return df_resampled


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT TEMPORAL ESTRICTO — 81 DÍAS
# ─────────────────────────────────────────────────────────────────────────────

def split_temporal_81dias(
    df: pd.DataFrame,
    n_dias_warmup: int = 1,
    n_dias_train:  int = 60,
    n_dias_val:    int = 10,
    n_dias_test:   int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Particiona el DataFrame en cuatro splits temporales consecutivos sin solapamiento.

    Los cortes se realizan sobre días calendario completos (medianoche exacta)
    calculados a partir del **primer día** del dataset.

    Parámetros
    ----------
    df            : DataFrame con columna 'Fecha' (o primera columna como fecha).
    n_dias_warmup : Número de días de calentamiento (default 1).
    n_dias_train  : Número de días de entrenamiento (default 60).
    n_dias_val    : Número de días de validación (default 10).
    n_dias_test   : Número de días de prueba (default 10).

    Returns
    -------
    (df_warmup, df_train, df_val, df_test) — cuatro DataFrames sin solapamiento.
    """
    col_fecha = 'Fecha' if 'Fecha' in df.columns else df.columns[0]
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df = df.sort_values(col_fecha).reset_index(drop=True)

    # Día de inicio: medianoche del primer día del dataset
    dia_inicio = df[col_fecha].min().normalize()

    corte_warmup_fin = dia_inicio + pd.Timedelta(days=n_dias_warmup)
    corte_train_fin  = corte_warmup_fin + pd.Timedelta(days=n_dias_train)
    corte_val_fin    = corte_train_fin  + pd.Timedelta(days=n_dias_val)
    # El test va hasta el final del dataset (puede ser < n_dias_test si el dataset es corto)
    corte_test_fin   = corte_val_fin    + pd.Timedelta(days=n_dias_test)

    fecha = df[col_fecha]
    df_warmup = df[fecha < corte_warmup_fin].copy()
    df_train  = df[(fecha >= corte_warmup_fin) & (fecha < corte_train_fin)].copy()
    df_val    = df[(fecha >= corte_train_fin)  & (fecha < corte_val_fin)].copy()
    df_test   = df[(fecha >= corte_val_fin)    & (fecha < corte_test_fin)].copy()

    total_dias_dataset = (df[col_fecha].max().normalize() - dia_inicio).days + 1
    print(
        f"\n{'─'*65}\n"
        f"  SPLIT TEMPORAL — {total_dias_dataset} días en el dataset\n"
        f"{'─'*65}\n"
        f"  Warmup : {df_warmup[col_fecha].min().date()} → {df_warmup[col_fecha].max().date()} "
        f"({len(df_warmup):>7,} filas | {n_dias_warmup}d)\n"
        f"  Train  : {df_train[col_fecha].min().date()} → {df_train[col_fecha].max().date()} "
        f"({len(df_train):>7,} filas | {n_dias_train}d)\n"
        f"  Val    : {df_val[col_fecha].min().date()}   → {df_val[col_fecha].max().date()}   "
        f"({len(df_val):>7,} filas | {n_dias_val}d)\n"
        f"  Test   : {df_test[col_fecha].min().date()}  → {df_test[col_fecha].max().date()}  "
        f"({len(df_test):>7,} filas | {n_dias_test}d)\n"
        f"{'─'*65}"
    )
    return df_warmup, df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def generar_features_ciclicas_y_tabulares(
    df: pd.DataFrame,
    sensor_cols: List[str],
    # Ventanas de estatismo (muestras a 30s/muestra)
    w_stuck_5m:  int   = 10,    # 5 min
    w_stuck_15m: int   = 30,    # 15 min
    eps_stuck:   float = 1e-6,
    # Ventanas rolling principales
    w_30m: int = 60,            # 30 min
    w_3h:  int = 360,           # 3 horas
    w_30m_min_periods: int = 5,
    w_3h_min_periods:  int = 30,
    inter_sensor_pairs: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Genera features causales compartidas por entrenamiento y streaming.

    Todas las operaciones usan únicamente la fila actual y el pasado:
    - Rolling integer-indexed equivale a una ventana retrospectiva pura.
    - Diff(k) con k > 0 es siempre pasado.

    Optimización de rendimiento:
    Se acumulan todas las columnas generadas en un diccionario (`new_cols`) y se unen
    al DataFrame original de una sola vez mediante `pd.concat` para evitar la fragmentación
    de memoria y eliminar los avisos `PerformanceWarning`.
    """
    # Preserve the missingness signal before derived features fill NaNs.
    df = df.copy()
    for col in sensor_cols:
        if col in df.columns:
            df[f'{col}_isna'] = df[col].isna().astype(np.int8)

    col_fecha = 'Fecha' if 'Fecha' in df.columns else df.columns[0]
    new_cols = {}

    # ── 1. Características Cíclicas de Tiempo ───────────────────────────────
    segundos_dia = (
        df[col_fecha].dt.hour   * 3600 +
        df[col_fecha].dt.minute * 60   +
        df[col_fecha].dt.second
    )
    new_cols['sin_hour']  = np.sin(2 * np.pi * segundos_dia / 86400.0)
    new_cols['cos_hour']  = np.cos(2 * np.pi * segundos_dia / 86400.0)

    # Día de la semana (0=lunes, 6=domingo) — captura patrones semanales
    dow = df[col_fecha].dt.dayofweek
    new_cols['sin_dow']   = np.sin(2 * np.pi * dow / 7.0)
    new_cols['cos_dow']   = np.cos(2 * np.pi * dow / 7.0)

    # Mes del año (1–12) — captura estacionalidad
    mes = df[col_fecha].dt.month - 1   # 0–11
    new_cols['sin_month'] = np.sin(2 * np.pi * mes / 12.0)
    new_cols['cos_month'] = np.cos(2 * np.pi * mes / 12.0)

    # Variable Día/Noche (is_daytime): True (1) si radiación solar > 10 W/m² o horario diurno (7h-19h)
    prad = df['PRAD'] if 'PRAD' in df.columns else pd.Series(0.0, index=df.index)
    prgint = df['PRGINT'] if 'PRGINT' in df.columns else pd.Series(0.0, index=df.index)
    is_day_rad = (prad > 10.0) | (prgint > 10.0)
    is_day_hour = (df[col_fecha].dt.hour >= 7) & (df[col_fecha].dt.hour <= 19)
    new_cols['is_daytime'] = (is_day_rad | is_day_hour).astype(np.int8)

    # ── 2. Diferencias Temporales, Absolutas y Lags ──────────────────────────
    # diff(1)  = 30s    → captura spikes/ruido
    # diff(3)  = 1.5min → inicio de tendencia sostenida
    # diff(10) = 5min   → inicio de plateau (stuck)
    # diff(30) = 15min  → cambio de media a plazo medio
    for col in sensor_cols:
        if col not in df.columns:
            continue
        s = df[col]
        d1 = s.diff(1).fillna(0.0)
        new_cols[f'{col}_diff1']     = d1
        new_cols[f'{col}_abs_diff1'] = d1.abs()
        new_cols[f'{col}_diff3']     = s.diff(3).fillna(0.0)
        new_cols[f'{col}_diff10']    = s.diff(10).fillna(0.0)
        new_cols[f'{col}_diff30']    = s.diff(30).fillna(0.0)
        new_cols[f'{col}_lag1']      = s.shift(1).fillna(s.iloc[0])
        new_cols[f'{col}_lag30']     = s.shift(30).fillna(s.iloc[0])


    # ── 3. Features de Estatismo — Sensor Atascado ──────────────────────────
    # rolling_std a 5 min (10 muestras) y 15 min (30 muestras):
    #   - En un plateau genuino, std → 0 antes de que diff30 llegue a cero.
    #   - stuck_flag = 1 si la varianza en la ventana corta es prácticamente nula.
    for col in sensor_cols:
        if col not in df.columns:
            continue
        s = df[col]

        rstd_5m  = s.rolling(window=w_stuck_5m,  min_periods=2).std().fillna(0.0)
        rstd_15m = s.rolling(window=w_stuck_15m, min_periods=5).std().fillna(0.0)

        new_cols[f'{col}_rstd_5m']       = rstd_5m
        new_cols[f'{col}_rstd_15m']      = rstd_15m
        new_cols[f'{col}_stuck_flag_5m'] = (rstd_5m  < eps_stuck).astype(np.int8)
        new_cols[f'{col}_stuck_flag_15m']= (rstd_15m < eps_stuck).astype(np.int8)

    # ── 3b. Ventana Móvil de 20 Muestras (10 min @ 30s) — Garantía Causal (shift(1)) ──
    # Extract rolling_std_20, rolling_range_20, zero_diff_count_20 and diurnal_residual
    for col in sensor_cols:
        if col not in df.columns:
            continue
        s = df[col]
        s_shifted = s.shift(1)
        roll20 = s_shifted.rolling(window=20, min_periods=5)

        # rolling_std_20: Desviación estándar de las últimas 20 muestras previas
        new_cols[f'{col}_rolling_std_20']   = roll20.std().fillna(0.0)

        # rolling_range_20: Rango (max - min) en la ventana de 20 pasos previos
        new_cols[f'{col}_rolling_range_20'] = (roll20.max() - roll20.min()).fillna(0.0)

        # zero_diff_count_20: Conteo de diferencias absolutas cercanas a cero (<1e-4) en 20 pasos
        abs_d1_shifted = s.diff(1).abs().shift(1)
        is_zero_diff = (abs_d1_shifted < 1e-4).astype(float)
        new_cols[f'{col}_zero_diff_count_20'] = is_zero_diff.rolling(window=20, min_periods=5).sum().fillna(0.0)

        # diurnal_residual: Residuo entre el valor actual (t) y la media móvil 1h (120 muestras) pasada
        rmean_1h = s_shifted.rolling(window=120, min_periods=10).mean()
        new_cols[f'{col}_diurnal_residual'] = (s - rmean_1h).fillna(0.0)

    # ── 4. Estadísticas Rolling Principales (30 min y 3 horas) ──────────────
    for col in sensor_cols:
        if col not in df.columns:
            continue
        s = df[col]

        short = s.rolling(window=w_30m, min_periods=w_30m_min_periods)
        long  = s.rolling(window=w_3h,  min_periods=w_3h_min_periods)

        rmean_30m = short.mean()
        rstd_30m  = short.std().fillna(0.0)
        rmean_3h  = long.mean()
        rstd_3h   = long.std().fillna(0.0)

        new_cols[f'{col}_rmean_30m']    = rmean_30m
        new_cols[f'{col}_rstd_30m']     = rstd_30m
        new_cols[f'{col}_rmean_3h']     = rmean_3h
        new_cols[f'{col}_rstd_3h']      = rstd_3h

        # Z-score local: cuántas std se aleja el valor actual de su media reciente
        new_cols[f'{col}_zscore_local'] = _safe_divide(
            s-rmean_30m, rstd_30m+1e-6)

    # ── 5. Relaciones Cruzadas y Correlación Rodante Causal entre Sensores ──
    pairs_to_process = inter_sensor_pairs
    if pairs_to_process is None:
        try:
            from config import INTER_SENSOR_PAIRS
            pairs_to_process = INTER_SENSOR_PAIRS
        except ImportError:
            pairs_to_process = [
                ('PRAD', 'PRGINT'),
                ('XTINV', 'PTEXT'),
                ('XHINV', 'PHEXT'),
                ('XCO2I', 'PCO2EXT')
            ]

    for colA, colB in pairs_to_process:
        if colA in df.columns and colB in df.columns:
            sA = df[colA]
            sB = df[colB]
            
            # Diferencias Bivariadas Inter-Sensor (Absolutas y Relativas)
            diff_abs = (sA - sB).abs()
            diff_rel = _safe_divide(diff_abs, sA.abs()+sB.abs()+1.0)
            new_cols[f'diff_abs_{colA}_{colB}'] = diff_abs
            new_cols[f'diff_rel_{colA}_{colB}'] = diff_rel

            # Causal Rolling Covariance & Correlation (shift(1) estricto para evitar mirar al futuro)
            sA_shift = sA.shift(1)
            sB_shift = sB.shift(1)

            roll10_A = sA_shift.rolling(window=10, min_periods=3)
            cov10 = roll10_A.cov(sB_shift).fillna(0.0)
            corr10 = roll10_A.corr(sB_shift).fillna(0.0)
            new_cols[f'cov_roll10_{colA}_{colB}']  = cov10
            new_cols[f'corr_roll10_{colA}_{colB}'] = corr10

            roll20_A = sA_shift.rolling(window=20, min_periods=5)
            cov20 = roll20_A.cov(sB_shift).fillna(0.0)
            corr20 = roll20_A.corr(sB_shift).fillna(0.0)
            new_cols[f'cov_roll20_{colA}_{colB}']  = cov20
            new_cols[f'corr_roll20_{colA}_{colB}'] = corr20

    # Compatibilidad con nombres de columnas históricas
    if {'PRAD', 'PRGINT'}.issubset(df.columns):
        new_cols['ratio_rad']          = _safe_divide(df['PRGINT'], df['PRAD']+1.0)
        new_cols['rolling_corr_20']     = new_cols.get('corr_roll20_PRAD_PRGINT', pd.Series(0.0, index=df.index))
        new_cols['rolling_corr_rad_20'] = new_cols['rolling_corr_20']
    if {'XTINV', 'PTEXT'}.issubset(df.columns):
        new_cols['delta_temp']         = df['XTINV'] - df['PTEXT']
        new_cols['rolling_corr_temp_20'] = new_cols.get('corr_roll20_XTINV_PTEXT', pd.Series(0.0, index=df.index))
    if {'XHINV', 'PHEXT'}.issubset(df.columns):
        new_cols['delta_hum']          = df['XHINV'] - df['PHEXT']
        new_cols['rolling_corr_hum_20']  = new_cols.get('corr_roll20_XHINV_PHEXT', pd.Series(0.0, index=df.index))
    if {'XCO2I', 'PCO2EXT'}.issubset(df.columns):
        new_cols['gradiente_co2']      = df['XCO2I'] - df['PCO2EXT']
        new_cols['rolling_corr_co2_20']  = new_cols.get('corr_roll20_XCO2I_PCO2EXT', pd.Series(0.0, index=df.index))

    # Z-scores horarios batch excluidos: evitar fuga temporal.

    # Unir todas las columnas generadas de una sola vez
    df_gen = pd.DataFrame(new_cols, index=df.index)
    df_feat = pd.concat([df, df_gen], axis=1)

    n_features_nuevas = df_feat.shape[1] - df.shape[1]
    print(
        f"Feature engineering completado. "
        f"Total columnas: {df_feat.shape[1]} (+{n_features_nuevas} generadas)"
    )
    return df_feat
