"""
injection.py — Módulo de Inyección Sintética de Anomalías (Metodología Manolo)
================================================================================
Inyecta 6 tipos de anomalías sintéticas sobre la matriz de datos y sus características:
1. Datos Faltantes (NaNs)
2. Sensor Atascado (Secuencias constantes)
3. Ruido / Picos (Spikes)
4. Valores Fuera de Rango (Límites físicos)
5. Desviación de Correlación (Exterior vs Interior)
6. Anomalías Contextuales (Radiación nocturna)

Genera la columna binaria target_anomalia (0 = Normal, 1 = Anomalía)
y la etiqueta tipo_anomalia para diagnóstico.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def aplicar_inyecciones(
    df: pd.DataFrame, 
    sensor_cols: List[str], 
    rangos_fisicos: Optional[Dict[str, Dict[str, float]]] = None,
    config_inyeccion: Optional[Dict] = None,
    random_state: int = 42,
    config: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aplica secuencialmente los 6 tipos de inyecciones sintéticas.
    """
    if config_inyeccion is None:
        config_inyeccion = config or INYECCION_CONFIG
    if seed is not None:
        random_state = seed
    if rangos_fisicos is None:
        from config import RANGOS_FISICOS
        rangos_fisicos = RANGOS_FISICOS

    df_out = df.copy()

    rng = np.random.default_rng(random_state)
    
    # Inicializar etiquetas de control
    df_out['target_anomalia'] = 0
    df_out['tipo_anomalia'] = 'Normal'
    
    # Raw missing/range faults are binary positives before synthetic sampling.
    # This keeps preserved acquisition gaps out of the "Normal" truth class.
    cols_obj = [c for c in sensor_cols if c in df_out.columns]
    if len(cols_obj) == len(sensor_cols):
        from src.binary_postprocess import (
            configured_rules, rule_codes, RULE_LABELS,
        )
        raw_values = df_out[cols_obj].to_numpy(np.float32)
        missing_flags = np.isnan(raw_values).astype(np.uint8)
        times_ns = pd.to_datetime(df_out["Fecha"]).to_numpy(
            dtype="datetime64[ns]").astype(np.int64)
        low = np.array([rangos_fisicos[c]["min"] for c in cols_obj], np.float32)
        high = np.array([rangos_fisicos[c]["max"] for c in cols_obj], np.float32)
        codes, _ = rule_codes(raw_values, missing_flags, times_ns,
                              configured_rules(), low, high)
        for code, label in RULE_LABELS.items():
            mask = codes == code
            df_out.loc[mask, "target_anomalia"] = 1
            df_out.loc[mask, "tipo_anomalia"] = label
    
    # Precalculo de desviaciones estándar por columna
    stds = {c: df_out[c].std() for c in cols_obj}
    
    # ─── 1. Datos Faltantes (NaNs) ──────────────────────────────────────────────
    cfg = config_inyeccion['datos_faltantes']
    idx_normales = df_out.index[df_out['target_anomalia'] == 0].to_numpy()
    n_afectadas = int(len(idx_normales) * cfg['porcentaje_filas'])
    
    if n_afectadas > 0:
        chosen_idx = rng.choice(idx_normales, size=n_afectadas, replace=False)
        for idx in chosen_idx:
            c = rng.choice(cols_obj)
            df_out.loc[idx, c] = np.nan
            df_out.loc[idx, 'target_anomalia'] = 1
            df_out.loc[idx, 'tipo_anomalia'] = 'Datos Faltantes'

    # ─── 2. Sensor Atascado (Constant values) ───────────────────────────────────
    cfg = config_inyeccion['sensor_atascado']
    idx_normales = df_out.index[df_out['target_anomalia'] == 0].to_numpy()
    
    col_fecha = 'Fecha' if 'Fecha' in df_out.columns else df_out.columns[0]
    if pd.api.types.is_datetime64_any_dtype(df_out[col_fecha]) and len(df_out) > 1:
        dias_df = max(1.0, (df_out[col_fecha].iloc[-1] - df_out[col_fecha].iloc[0]).total_seconds() / 86400.0)
    else:
        dias_df = len(df_out) / 2880.0
    
    sec_por_dia = cfg.get('secuencias_por_dia', 3.33)
    num_sec = int(round(dias_df * sec_por_dia))
    
    for _ in range(num_sec):
        if len(idx_normales) == 0:
            break
        c = rng.choice(cols_obj)
        duracion_muestras = int(rng.integers(cfg['duracion_min'], cfg['duracion_max'] + 1) * 2) # 30s step
        
        if len(idx_normales) <= duracion_muestras + 1:
            break
        start_pos = rng.choice(len(idx_normales) - duracion_muestras - 1)
        seq_idx = idx_normales[start_pos : start_pos + duracion_muestras]
        
        val_atascado = df_out.loc[seq_idx[0], c]
        if pd.isna(val_atascado):
            val_atascado = df_out[c].mean()
            
        df_out.loc[seq_idx, c] = val_atascado
        df_out.loc[seq_idx, 'target_anomalia'] = 1
        df_out.loc[seq_idx, 'tipo_anomalia'] = 'Sensor Atascado'

    # ─── 3. Ruido / Picos ───────────────────────────────────────────────────────
    cfg = config_inyeccion['ruido']
    idx_normales = df_out.index[df_out['target_anomalia'] == 0].to_numpy()
    n_afectadas = int(len(idx_normales) * cfg['porcentaje_filas'])
    
    if n_afectadas > 0:
        chosen_idx = rng.choice(idx_normales, size=n_afectadas, replace=False)
        for idx in chosen_idx:
            c = rng.choice(cols_obj)
            mult = rng.uniform(cfg['factor_std_min'], cfg['factor_std_max'])
            signo = rng.choice([-1, 1])
            ruido = signo * mult * stds[c]
            
            curr_val = df_out.loc[idx, c]
            if not pd.isna(curr_val):
                df_out.loc[idx, c] = curr_val + ruido
                df_out.loc[idx, 'target_anomalia'] = 1
                df_out.loc[idx, 'tipo_anomalia'] = 'Ruido'

    # ─── 4. Valores Fuera de Rango ──────────────────────────────────────────────
    cfg = config_inyeccion['fuera_rango']
    idx_normales = df_out.index[df_out['target_anomalia'] == 0].to_numpy()
    n_afectadas = int(len(idx_normales) * cfg['porcentaje_filas'])
    
    if n_afectadas > 0:
        chosen_idx = rng.choice(idx_normales, size=n_afectadas, replace=False)
        for idx in chosen_idx:
            c = rng.choice(cols_obj)
            if c in rangos_fisicos:
                lim_min = rangos_fisicos[c]['min']
                lim_max = rangos_fisicos[c]['max']
                rango = max(1.0, lim_max - lim_min)
                
                # Forzar por debajo o por encima
                if rng.random() < 0.5:
                    val_out = lim_min - rng.uniform(0.1, 0.5) * rango
                else:
                    val_out = lim_max + rng.uniform(0.1, 0.5) * rango
                    
                df_out.loc[idx, c] = val_out
                df_out.loc[idx, 'target_anomalia'] = 1
                df_out.loc[idx, 'tipo_anomalia'] = 'Valores Fuera de Rango'

    # ─── 5. Desviación de Correlación (PTEXT vs XTINV / PRAD vs PRGINT) ─────────
    cfg = config_inyeccion['desviacion_correlacion']
    col_fecha = 'Fecha' if 'Fecha' in df_out.columns else df_out.columns[0]
    horas = df_out[col_fecha].dt.hour
    mask_diurna = (horas >= cfg['filtro_hora_inicio']) & (horas <= cfg['filtro_hora_fin']) & (df_out['target_anomalia'] == 0)
    idx_diurnos = df_out.index[mask_diurna].to_numpy()
    
    n_afectadas = int(len(idx_diurnos) * cfg['porcentaje_filas'])
    if n_afectadas > 0 and 'XTINV' in cols_obj and 'PTEXT' in cols_obj:
        chosen_idx = rng.choice(idx_diurnos, size=n_afectadas, replace=False)
        for idx in chosen_idx:
            # Desacoplar temperatura interior respecto a exterior
            df_out.loc[idx, 'XTINV'] = df_out.loc[idx, 'PTEXT'] - rng.uniform(10.0, 20.0)
            df_out.loc[idx, 'target_anomalia'] = 1
            df_out.loc[idx, 'tipo_anomalia'] = 'Desviación de Correlación'

    # ─── 6. Anomalía Contextual (Luz nocturna) ──────────────────────────────────
    cfg = config_inyeccion['contextual_luz']
    mask_nocturna = ((horas >= cfg['hora_noche_inicio']) | (horas <= cfg['hora_noche_fin'])) & (df_out['target_anomalia'] == 0)
    idx_nocturnos = df_out.index[mask_nocturna].to_numpy()
    
    n_afectadas = int(len(idx_nocturnos) * cfg['porcentaje_filas'])
    if n_afectadas > 0 and 'PRAD' in cols_obj:
        chosen_idx = rng.choice(idx_nocturnos, size=n_afectadas, replace=False)
        for idx in chosen_idx:
            df_out.loc[idx, 'PRAD'] = rng.uniform(200.0, 600.0) # Radiación alta de noche
            df_out.loc[idx, 'target_anomalia'] = 1
            df_out.loc[idx, 'tipo_anomalia'] = 'Contextual'

    # Infrastructure uses its own seed and only still-normal, contiguous,
    # finite windows. Existing synthetic types keep their binary definition.
    from config import INFRA_SEED_OFFSET
    df_out = inyectar_episodios_infraestructura(
        df_out, cols_obj, seed=random_state+INFRA_SEED_OFFSET)

    n_total = len(df_out)
    n_anom = (df_out['target_anomalia'] == 1).sum()
    pct_anom = (n_anom / n_total) * 100.0
    
    print(f"Inyección sintetizada completada: {n_anom:,} / {n_total:,} filas etiquetadas como anomalías ({pct_anom:.2f}%).")
    print("Desglose por tipo de anomalía:")
    print(df_out['tipo_anomalia'].value_counts())
    
    return df_out

inyectar_anomalias_sinteticas = aplicar_inyecciones



def inyectar_episodios_infraestructura(
    frame: pd.DataFrame, sensor_cols: List[str], *, seed: int,
) -> pd.DataFrame:
    """Inject separate reproducible 5-10 min outages into one temporal split."""
    from config import (
        MUESTREO_SEGUNDOS, INFRA_EPISODIOS_POR_DIA,
        INFRA_DURACION_MIN_MIN, INFRA_DURACION_MAX_MIN,
        TIPO_CAIDA_SISTEMA, TIPO_FALLO_PARCIAL_SISTEMA,
    )
    out = frame.copy()
    if len(sensor_cols) != 9 or any(col not in out for col in sensor_cols):
        raise ValueError("Infrastructure injection requires all nine target sensors")
    times = pd.to_datetime(out["Fecha"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if len(times) < 2:
        return out
    days = max(1.0, (times[-1]-times[0])/86_400_000_000_000)
    count = int(round(days*INFRA_EPISODIOS_POR_DIA))
    rng = np.random.default_rng(seed)
    step = MUESTREO_SEGUNDOS*1_000_000_000
    min_rows = int(np.ceil(INFRA_DURACION_MIN_MIN*60/MUESTREO_SEGUNDOS))
    max_rows = int(np.ceil(INFRA_DURACION_MAX_MIN*60/MUESTREO_SEGUNDOS))
    clean = (out["target_anomalia"].to_numpy() == 0)
    clean &= np.isfinite(out[sensor_cols].to_numpy(np.float32)).all(axis=1)
    for kind in (TIPO_CAIDA_SISTEMA, TIPO_FALLO_PARCIAL_SISTEMA):
        made = 0
        for _ in range(max(2000, 100*count)):
            if made >= count:
                break
            duration = int(rng.integers(min_rows, max_rows+1))
            if duration > len(out):
                break
            start = int(rng.integers(0, len(out)-duration+1))
            stop = start+duration
            if not clean[start:stop].all() or np.any(np.diff(times[start:stop]) != step):
                continue
            indices = out.index[start:stop]
            if kind == TIPO_CAIDA_SISTEMA:
                out.loc[indices, sensor_cols] = np.nan
            elif made % 2 == 0:
                lost = rng.choice(sensor_cols, 4, replace=False)
                out.loc[indices, lost] = np.nan
            else:
                chosen = rng.choice(sensor_cols, 7, replace=False)
                out.loc[indices, chosen[:3]] = np.nan
                out.loc[indices, chosen[3:]] = 0.0
            out.loc[indices, "target_anomalia"] = 1
            out.loc[indices, "tipo_anomalia"] = kind
            clean[start:stop] = False
            made += 1
        if made != count:
            print(f"Infraestructura: solo {made}/{count} ventanas limpias para {kind}")
    return out
