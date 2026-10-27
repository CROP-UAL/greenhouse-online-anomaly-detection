"""
streaming_detector.py — Motor de Inferencia en Tiempo Real (Edge Streaming + Debouncing)
========================================================================================
Clase para inferencia muestra a muestra en tiempo real con latencia ultra-baja.
Mantiene un búfer circular en memoria para actualizar características tabulares,
aplica guardias deterministas O(1) con imputación forward-fill, clasifica con LightGBM
y aplica filtrado por persistencia condicional/selectiva (debouncing).

PARIDAD FEATURE ENGINEERING: las features calculadas en el hot path coinciden
100% causalmente con `src/features.py` en modo batch.
"""

import time
from collections import deque
import numpy as np
import pandas as pd
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple


class OnlineLightGBMDetector:
    def __init__(
        self,
        model:          Any,
        feature_cols:   List[str],
        sensor_cols:    List[str],
        rangos_fisicos: Dict[str, Dict[str, float]],
        window_size:    int   = 360,          # buffer = 3h @ 30s/muestra
        threshold:      float = 0.48,
        k_persistence:  int   = 5,
        thresh_std:     Optional[float] = None,
        k_long:         Optional[int] = None,
        thresh_high:    float = 0.85,
        k_short:        int   = 1,
        fill_values:    Optional[Mapping[str, float]] = None,
        history:        Optional[Iterable[Mapping[str, Any]]] = None,
        # ── Guardia Sensor Atascado (detectar plateau en buffer) ────────────────
        stuck_window:   int   = 10,           # muestras para la guardia determinista (5 min @ 30s)
        guardia_epsilon:float = 0.02,         # rango mínimo realista (evitar FP en señales estables)
        use_stuck_guard: bool = False,        # DESACTIVADO por defecto: LightGBM ya aprende SA con stuck_flag features

        stuck_sensors:  Optional[Iterable[str]] = None,
        # ── Ventanas de estatismo para features ML ───────────────────────────
        w_stuck_5m:     int   = 10,           # 5 min @ 30s/muestra
        w_stuck_15m:    int   = 30,           # 15 min @ 30s/muestra
        eps_stuck:      float = 1e-6,         # umbral de varianza nula para stuck_flag_*
        # ── Ventanas rolling principales ────────────────────────────────────
        w_30m:          int   = 60,
        w_3h:           int   = 360,
        inter_sensor_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Crea un detector LightGBM en streaming con Debouncer Multi-Ruta / Adaptativo y guardias O(1).
        """
        self.model          = model
        self.feature_cols   = list(feature_cols)
        self.sensor_cols    = list(sensor_cols)
        self.rangos_fisicos = rangos_fisicos
        self.window_size    = window_size

        # Configuración Multi-Ruta / Adaptativa
        self.thresh_std     = float(thresh_std if thresh_std is not None else threshold)
        self.k_long         = int(k_long if k_long is not None else k_persistence)
        self.thresh_high    = float(thresh_high)
        self.k_short        = int(k_short)
        
        # Aliases de compatibilidad
        self.threshold      = self.thresh_std
        self.k_persistence  = self.k_long
        self.fill_values    = dict(fill_values or {})

        # Estado interno de debouncing multi-ruta
        self.consecutive_std_alerts: int = 0
        self.consecutive_high_alerts: int = 0
        self.consecutive_raw_alerts: int = 0

        # Inter-sensor pairs
        if inter_sensor_pairs is None:
            try:
                from config import INTER_SENSOR_PAIRS
                self.inter_sensor_pairs = INTER_SENSOR_PAIRS
            except ImportError:
                self.inter_sensor_pairs = [
                    ('PRAD', 'PRGINT'),
                    ('XTINV', 'PTEXT'),
                    ('XHINV', 'PHEXT'),
                    ('XCO2I', 'PCO2EXT')
                ]
        else:
            self.inter_sensor_pairs = list(inter_sensor_pairs)

        # Guardia Sensor Atascado (excluyendo radiaciones PRAD/PRGINT que se anulan de noche)
        self.stuck_window  = int(stuck_window)
        self.stuck_epsilon = float(guardia_epsilon)
        self.use_stuck_guard = bool(use_stuck_guard)
        default_stuck_sensors = [c for c in self.sensor_cols if c not in ('PRAD', 'PRGINT')]
        self.stuck_sensors = set(stuck_sensors or default_stuck_sensors)

        # Ventanas de features ML
        self.w_stuck_5m  = int(w_stuck_5m)
        self.w_stuck_15m = int(w_stuck_15m)
        self.eps_stuck   = float(eps_stuck)
        self.w_30m       = int(w_30m)
        self.w_3h        = int(w_3h)

        # Búfer circular: max 3h de historia a 30s/muestra
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        if history is not None:
            for sample in history:
                self._append_sample(sample)

    def reset_state(self):
        """Reinicia contadores de estado (debouncing) manteniendo el buffer."""
        self.consecutive_std_alerts = 0
        self.consecutive_high_alerts = 0
        self.consecutive_raw_alerts = 0

    # ──────────────────────────────────────────────────────────────────────────
    # UTILIDADES INTERNAS Y GESTIÓN DE BUFFER (Imputación Forward-Fill)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_last_valid_value(self, col: str) -> float:
        """Devuelve el último valor válido en el buffer para la columna, o fill_value."""
        for row in reversed(self.buffer):
            val = row.get(col)
            if val is not None and not pd.isna(val):
                return float(val)
        return float(self.fill_values.get(col, 0.0))

    def _append_sample(self, sample: Mapping[str, Any]) -> None:
        """Añade una muestra al búfer directamente."""
        self.buffer.append({
            'Fecha': sample.get('Fecha'),
            **{col: sample.get(col, np.nan) for col in self.sensor_cols},
            **{f'{col}_isna': int(bool(sample.get(f'{col}_isna', pd.isna(sample.get(col, np.nan)))))
               for col in self.sensor_cols},
        })

    def _append_imputed_sample(
        self,
        timestamp: pd.Timestamp,
        sample_dict: Mapping[str, Any],
        invalid_cols: Iterable[str]
    ) -> None:
        """
        Añade una muestra al búfer aplicando imputación forward-fill para las columnas inválidas.
        Esto evita propagar NaNs o desbordamientos físicos a las ventanas rodantes de LightGBM.
        """
        imputed = {'Fecha': timestamp}
        invalid_set = set(invalid_cols)

        for col in self.sensor_cols:
            val = sample_dict.get(col, np.nan)
            if col in invalid_set or pd.isna(val):
                imputed[col] = self._get_last_valid_value(col)
            else:
                imputed[col] = float(val)

        # Preserve raw missingness before forward-fill replaces sensor values.
        imputed.update({
            f'{col}_isna': int(pd.isna(sample_dict.get(col, np.nan)))
            for col in self.sensor_cols
        })
        self.buffer.append(imputed)

    @staticmethod
    def _nanmean(values: np.ndarray, fallback: float = 0.0) -> float:
        valid = values[~np.isnan(values)]
        return float(valid.mean()) if valid.size else fallback

    @staticmethod
    def _pandas_std(values: np.ndarray) -> float:
        """std muestral (ddof=1), igual que pandas Series.std(); NaN → 0."""
        valid = values[~np.isnan(values)]
        return float(valid.std(ddof=1)) if valid.size > 1 else 0.0

    def _values_from_buffer(self, col: str, n_last: int) -> np.ndarray:
        """Devuelve los últimos n_last valores del búfer para la columna col."""
        rows = list(self.buffer)
        return np.asarray([r.get(col, np.nan) for r in rows[-n_last:]], dtype=float)

    # ──────────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DEL VECTOR DE FEATURES (hot path con paridad 100% causal)
    # ──────────────────────────────────────────────────────────────────────────

    def _latest_features(
        self,
        timestamp:   pd.Timestamp,
        sample_dict: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Construye el vector de features de la muestra actual.
        La muestra actual (t) ya fue añadida a self.buffer y se encuentra en rows[n - 1].
        """
        seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        features: Dict[str, float] = {
            col: sample_dict.get(col, np.nan) for col in self.sensor_cols
        }
        # Must precede every imputation, matching batch feature engineering.
        for col in self.sensor_cols:
            features[f'{col}_isna'] = int(pd.isna(sample_dict.get(col, np.nan)))

        # ── Features cíclicas ─────────────────────────────────────────────
        features['sin_hour']  = float(np.sin(2 * np.pi * seconds / 86400.0))
        features['cos_hour']  = float(np.cos(2 * np.pi * seconds / 86400.0))

        dow = timestamp.dayofweek
        features['sin_dow']   = float(np.sin(2 * np.pi * dow / 7.0))
        features['cos_dow']   = float(np.cos(2 * np.pi * dow / 7.0))

        mes = timestamp.month - 1   # 0–11
        features['sin_month'] = float(np.sin(2 * np.pi * mes / 12.0))
        features['cos_month'] = float(np.cos(2 * np.pi * mes / 12.0))

        prad_val = float(sample_dict.get('PRAD', 0.0) or 0.0)
        prgint_val = float(sample_dict.get('PRGINT', 0.0) or 0.0)
        is_day_rad = (prad_val > 10.0) or (prgint_val > 10.0)
        is_day_hour = 7 <= timestamp.hour <= 19
        features['is_daytime'] = 1 if (is_day_rad or is_day_hour) else 0

        rows = list(self.buffer)
        n    = len(rows)

        for col in self.sensor_cols:
            current = sample_dict.get(col, np.nan)
            fv      = self.fill_values.get(col, 0.0)

            # ── Lags (t - 1 y t - 30) ─────────────────────────────────────
            prev1  = rows[n - 2].get(col, np.nan) if n >= 2  else np.nan
            lag30  = rows[n - 31].get(col, np.nan) if n >= 31 else np.nan

            features[f'{col}_lag1']  = float(prev1) if not np.isnan(prev1) else fv
            features[f'{col}_lag30'] = float(lag30) if not np.isnan(lag30) else fv

            # ── Diffs (current - past_k) ──────────────────────────────────
            def _diff(k):
                past = rows[n - 1 - k].get(col, np.nan) if n >= 1 + k else np.nan
                if np.isnan(current) or np.isnan(past):
                    return 0.0
                return float(current - past)

            d1 = _diff(1)
            features[f'{col}_diff1']     = d1
            features[f'{col}_abs_diff1'] = abs(d1)
            features[f'{col}_diff3']     = _diff(3)
            features[f'{col}_diff10']    = _diff(10)
            features[f'{col}_diff30']    = _diff(30)


            # ── Features de Estatismo (Sensor Atascado) ────────────────────
            vals_5m  = self._values_from_buffer(col, self.w_stuck_5m)
            vals_15m = self._values_from_buffer(col, self.w_stuck_15m)

            rstd_5m  = self._pandas_std(vals_5m)
            rstd_15m = self._pandas_std(vals_15m)

            features[f'{col}_rstd_5m']        = rstd_5m
            features[f'{col}_rstd_15m']       = rstd_15m
            features[f'{col}_stuck_flag_5m']  = 1 if (len(vals_5m) >= 5 and rstd_5m < self.eps_stuck) else 0
            features[f'{col}_stuck_flag_15m'] = 1 if (len(vals_15m) >= 15 and rstd_15m < self.eps_stuck) else 0

            # ── Ventana Móvil 20 Muestras (10 min @ 30s) Causal (shift(1)) ──
            past20 = rows[max(0, n - 21):n - 1]
            if len(past20) > 0:
                vals20 = np.asarray([r.get(col, np.nan) for r in past20], dtype=float)
                features[f'{col}_rolling_std_20'] = self._pandas_std(vals20)

                valid20 = vals20[~np.isnan(vals20)]
                features[f'{col}_rolling_range_20'] = float(np.ptp(valid20)) if valid20.size > 1 else 0.0

                diffs20 = np.abs(np.diff(vals20))
                valid_d20 = diffs20[~np.isnan(diffs20)]
                features[f'{col}_zero_diff_count_20'] = float((valid_d20 < 1e-4).sum()) if valid_d20.size > 0 else 0.0
            else:
                features[f'{col}_rolling_std_20']     = 0.0
                features[f'{col}_rolling_range_20']   = 0.0
                features[f'{col}_zero_diff_count_20'] = 0.0

            # Residuo diurno/horario (1h = 120 muestras pasadas)
            past1h = rows[max(0, n - 121):n - 1]
            vals1h = np.asarray([r.get(col, np.nan) for r in past1h], dtype=float)
            mean1h = self._nanmean(vals1h, float(current if not np.isnan(current) else fv))
            features[f'{col}_diurnal_residual'] = float(current - mean1h) if not np.isnan(current) else 0.0


            # ── Rolling 30m y 3h ───────────────────────────────────────────
            short_vals = self._values_from_buffer(col, self.w_30m)
            long_vals  = self._values_from_buffer(col, self.w_3h)

            short_mean = self._nanmean(short_vals, fv)
            short_std  = self._pandas_std(short_vals)
            long_mean  = self._nanmean(long_vals, fv)
            long_std   = self._pandas_std(long_vals)

            features[f'{col}_rmean_30m']    = short_mean
            features[f'{col}_rstd_30m']     = short_std
            features[f'{col}_rmean_3h']     = long_mean
            features[f'{col}_rstd_3h']      = long_std
            features[f'{col}_zscore_local'] = (
                float((current - short_mean) / (short_std + 1e-6))
                if not np.isnan(current) else 0.0
            )

        # ── Relaciones Cruzadas y Correlación/Covarianza Rodante Causal ───
        past10_global = rows[max(0, n - 11):n - 1]
        past20_global = rows[max(0, n - 21):n - 1]

        for colA, colB in self.inter_sensor_pairs:
            valA = features.get(colA, np.nan)
            valB = features.get(colB, np.nan)
            if not np.isnan(valA) and not np.isnan(valB):
                diff_abs = abs(valA - valB)
                diff_rel = diff_abs / (abs(valA) + abs(valB) + 1.0)
                features[f'diff_abs_{colA}_{colB}'] = float(diff_abs)
                features[f'diff_rel_{colA}_{colB}'] = float(diff_rel)
            else:
                features[f'diff_abs_{colA}_{colB}'] = 0.0
                features[f'diff_rel_{colA}_{colB}'] = 0.0

            # Causal w=10
            if len(past10_global) >= 3:
                v1_10 = np.asarray([r.get(colA, np.nan) for r in past10_global], dtype=float)
                v2_10 = np.asarray([r.get(colB, np.nan) for r in past10_global], dtype=float)
                mask10 = ~np.isnan(v1_10) & ~np.isnan(v2_10)
                if mask10.sum() >= 3 and v1_10[mask10].std() > 1e-6 and v2_10[mask10].std() > 1e-6:
                    cov10 = float(np.cov(v1_10[mask10], v2_10[mask10])[0, 1])
                    corr10 = float(np.corrcoef(v1_10[mask10], v2_10[mask10])[0, 1])
                else:
                    cov10, corr10 = 0.0, 0.0
            else:
                cov10, corr10 = 0.0, 0.0
            features[f'cov_roll10_{colA}_{colB}']  = 0.0 if np.isnan(cov10) else cov10
            features[f'corr_roll10_{colA}_{colB}'] = 0.0 if np.isnan(corr10) else corr10

            # Causal w=20
            if len(past20_global) >= 5:
                v1_20 = np.asarray([r.get(colA, np.nan) for r in past20_global], dtype=float)
                v2_20 = np.asarray([r.get(colB, np.nan) for r in past20_global], dtype=float)
                mask20 = ~np.isnan(v1_20) & ~np.isnan(v2_20)
                if mask20.sum() >= 5 and v1_20[mask20].std() > 1e-6 and v2_20[mask20].std() > 1e-6:
                    cov20 = float(np.cov(v1_20[mask20], v2_20[mask20])[0, 1])
                    corr20 = float(np.corrcoef(v1_20[mask20], v2_20[mask20])[0, 1])
                else:
                    cov20, corr20 = 0.0, 0.0
            else:
                cov20, corr20 = 0.0, 0.0
            features[f'cov_roll20_{colA}_{colB}']  = 0.0 if np.isnan(cov20) else cov20
            features[f'corr_roll20_{colA}_{colB}'] = 0.0 if np.isnan(corr20) else corr20

        # Compatibilidad con nombres de columnas históricas
        if {'PRAD', 'PRGINT'}.issubset(features):
            features['ratio_rad']          = float(features['PRGINT'] / (features['PRAD'] + 1.0))
            features['rolling_corr_20']     = features.get('corr_roll20_PRAD_PRGINT', 0.0)
            features['rolling_corr_rad_20'] = features['rolling_corr_20']
        if {'XTINV', 'PTEXT'}.issubset(features):
            features['delta_temp']          = float(features['XTINV'] - features['PTEXT'])
            features['rolling_corr_temp_20'] = features.get('corr_roll20_XTINV_PTEXT', 0.0)
        if {'XHINV', 'PHEXT'}.issubset(features):
            features['delta_hum']           = float(features['XHINV'] - features['PHEXT'])
            features['rolling_corr_hum_20']  = features.get('corr_roll20_XHINV_PHEXT', 0.0)
        if {'XCO2I', 'PCO2EXT'}.issubset(features):
            features['gradiente_co2']       = float(features['XCO2I'] - features['PCO2EXT'])
            features['rolling_corr_co2_20']  = features.get('corr_roll20_XCO2I_PCO2EXT', 0.0)

        # Desviación contextual por hora
        for col in self.sensor_cols:
            val = features.get(col, np.nan)
            s_mean = features.get(f'{col}_rmean_3h', val if not np.isnan(val) else 0.0)
            s_std  = features.get(f'{col}_rstd_3h', 1.0)
            features[f'{col}_zscore_hourly'] = float((val - s_mean) / (s_std + 1e-6)) if not np.isnan(val) else 0.0

        return features

    # ──────────────────────────────────────────────────────────────────────────
    # GUARDIAS DETERMINISTAS O(1)
    # ──────────────────────────────────────────────────────────────────────────

    def evaluar_guardia_nan(
        self,
        sample_dict: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Comprueba en O(1) si existen valores NaN en las columnas del sensor."""
        nan_cols = [c for c in self.sensor_cols if c in sample_dict and pd.isna(sample_dict[c])]
        return (len(nan_cols) > 0, nan_cols)

    def evaluar_guardia_determinista(
        self,
        sample_dict: Dict[str, float],
    ) -> Tuple[bool, str, List[str]]:
        """Guardia física de rangos fijos de hardware."""
        out_of_bounds_cols = []
        razones = []
        for col in self.sensor_cols:
            if col in sample_dict and not pd.isna(sample_dict[col]):
                val = sample_dict[col]
                if col in self.rangos_fisicos:
                    r_min = self.rangos_fisicos[col]['min']
                    r_max = self.rangos_fisicos[col]['max']
                    if val < r_min or val > r_max:
                        out_of_bounds_cols.append(col)
                        razones.append(f"{col}={val:.2f}")

        if out_of_bounds_cols:
            return True, f"Fuera de rango físico ({', '.join(razones)})", out_of_bounds_cols
        return False, "Normal", []

    def _detect_stuck(self, timestamp: Optional[pd.Timestamp] = None) -> Tuple[bool, str]:
        """
        Regla Determinista Post-Modelo (Hybrid Fallback):
        Si (std_N(radiación) == 0) AND (is_daytime == True) AND (radiación > 10 W/m²) => Sensor Atascado = 1
        """
        if len(self.buffer) < self.stuck_window:
            return False, "Normal"
        recent = list(self.buffer)[-self.stuck_window:]

        last_sample = recent[-1]
        hour = timestamp.hour if timestamp is not None else 12
        prad_val = float(last_sample.get('PRAD', 0.0) or 0.0)
        prgint_val = float(last_sample.get('PRGINT', 0.0) or 0.0)
        is_daytime = (prad_val > 10.0) or (prgint_val > 10.0) or (7 <= hour <= 19)

        # Regla Híbrida Post-Modelo para Radiación Solar Diurna
        if is_daytime:
            for rad_col in ('PRAD', 'PRGINT'):
                if rad_col in last_sample:
                    rad_vals = np.asarray([row.get(rad_col, np.nan) for row in recent], dtype=float)
                    if not np.isnan(rad_vals).any() and float(np.mean(rad_vals)) > 10.0:
                        if float(np.std(rad_vals)) == 0.0 or float(np.ptp(rad_vals)) < 1e-5:
                            return True, f"Regla Híbrida Determinista: Radiación atascada ({rad_col} > 10 W/m², std=0)"

        return False, "Normal"


    # ──────────────────────────────────────────────────────────────────────────
    # PROCESAMIENTO DE UNA MUESTRA (hot path envolviendo Latencia End-to-End ns)
    # ──────────────────────────────────────────────────────────────────────────

    def process_sample(
        self,
        timestamp:   pd.Timestamp,
        sample_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Procesa una única muestra entrante envolviendo todo el hot path con `time.perf_counter_ns()`
        para medir la latencia end-to-end (guardias, actualización de buffer con imputación,
        inferencia LightGBM y debouncing selectivo).
        """
        t0_ns = time.perf_counter_ns()

        # ── 1. Guardia Determinista O(1): Datos Faltantes (NaNs) ──────────────
        es_nan, nan_cols = self.evaluar_guardia_nan(sample_dict)
        if es_nan:
            # Imputación forward-fill en buffer para mantener ventanas válidas hacia el futuro
            self._append_imputed_sample(timestamp, sample_dict, invalid_cols=nan_cols)
            self.consecutive_raw_alerts += 1
            lat_ms = (time.perf_counter_ns() - t0_ns) / 1e6
            return {
                'timestamp': timestamp,
                'is_anomaly_raw': True,
                'is_anomaly_debounced': True,  # K=1 inmediata para transitoria / NaN
                'is_anomaly': True,
                'anomaly_probability': 1.0,
                'consecutive_alerts': self.consecutive_raw_alerts,
                'k_persistence': 1,
                'method': 'Guardia Determinista',
                'reason': f"Dato Faltante (NaN en {', '.join(nan_cols)})",
                'threshold': self.threshold,
                'latency_ms': lat_ms,
            }

        # ── 2. Guardia Determinista O(1): Fuera de Rango Físico ───────────────
        es_anom_fisica, razon_fisica, out_cols = self.evaluar_guardia_determinista(sample_dict)
        if es_anom_fisica:
            # Imputación forward-fill en buffer para el sensor fuera de rango
            self._append_imputed_sample(timestamp, sample_dict, invalid_cols=out_cols)
            self.consecutive_raw_alerts += 1
            lat_ms = (time.perf_counter_ns() - t0_ns) / 1e6
            return {
                'timestamp': timestamp,
                'is_anomaly_raw': True,
                'is_anomaly_debounced': True,  # K=1 inmediata para picos fuera de rango
                'is_anomaly': True,
                'anomaly_probability': 1.0,
                'consecutive_alerts': self.consecutive_raw_alerts,
                'k_persistence': 1,
                'method': 'Guardia Determinista',
                'reason': razon_fisica,
                'threshold': self.threshold,
                'latency_ms': lat_ms,
            }

        # ── Insertar muestra válida actual en el búfer ────────────────────────
        self._append_sample({'Fecha': timestamp, **sample_dict})

        # ── 3. Guardia Determinista: Sensor Atascado (solo si use_stuck_guard=True) ─────
        # Por defecto desactivada: LightGBM aprende SA con stuck_flag_5m/15m, rstd_5m/15m.
        # Activar solo si se confirma que no genera falsas alarmas en el conjunto de test.
        if self.use_stuck_guard:
            es_atascado, razon_atascado = self._detect_stuck(timestamp)
            if es_atascado:
                self.consecutive_raw_alerts += 1
                lat_ms = (time.perf_counter_ns() - t0_ns) / 1e6
                return {
                    'timestamp': timestamp,
                    'is_anomaly_raw': True,
                    'is_anomaly_debounced': True,
                    'is_anomaly': True,
                    'anomaly_probability': 1.0,
                    'consecutive_alerts': self.consecutive_raw_alerts,
                    'k_persistence': self.k_persistence,
                    'method': 'Regla Híbrida / Guardia Sensor Atascado',
                    'reason': razon_atascado,
                    'threshold': self.threshold,
                    'latency_ms': lat_ms,
                }


        # ── 4. Extraer features causales ──────────────────────────────────────
        feat_vector = self._latest_features(timestamp, sample_dict)

        # ── 5. Inferencia LightGBM solo para Muestras Válidas Complejas ───────
        X_sample = np.asarray(
            [
                self.fill_values.get(col, 0.0) if np.isnan(feat_vector.get(col, np.nan))
                else feat_vector.get(col, 0.0)
                for col in self.feature_cols
            ],
            dtype=np.float64,
        ).reshape(1, -1)

        proba = float(self.model.predict(X_sample)[0])
        is_raw = bool(proba >= self.thresh_std)

        # ── 6. Debouncer Multi-Ruta / Adaptativo (Fusión de Canales) ─────────
        # Ruta A: Alta Persistencia (Sensor Atascado)
        if proba >= self.thresh_std:
            self.consecutive_std_alerts += 1
        else:
            self.consecutive_std_alerts = 0

        # Ruta B: Eventos Instantáneos/Pico (Ruido / Fuera de Rango / Spikes)
        if proba >= self.thresh_high:
            self.consecutive_high_alerts += 1
        else:
            self.consecutive_high_alerts = 0

        is_alert_std  = bool(self.consecutive_std_alerts >= self.k_long)
        is_alert_high = bool(self.consecutive_high_alerts >= self.k_short)

        # Lógica de Fusión: Combina ambos canales mediante un OR lógico
        is_debounced = bool(is_alert_std or is_alert_high)
        self.consecutive_raw_alerts = max(self.consecutive_std_alerts, self.consecutive_high_alerts)

        lat_ms = (time.perf_counter_ns() - t0_ns) / 1e6
        reason_str = (
            'Alerta Multi-Ruta (Efímera)' if is_alert_high else
            ('Alerta Multi-Ruta (Persistencia SA)' if is_alert_std else
            ('Anomalía ML (Raw)' if is_raw else 'Normal'))
        )

        return {
            'timestamp':           timestamp,
            'is_anomaly_raw':      is_raw,
            'is_anomaly_debounced':is_debounced,
            'is_anomaly':          is_debounced,
            'anomaly_probability': proba,
            'consecutive_alerts':  self.consecutive_raw_alerts,
            'k_persistence':       self.k_long,
            'method':              'LightGBM Multi-Ruta Streaming',
            'reason':              reason_str,
            'threshold':           self.thresh_std,
            'latency_ms':          lat_ms,
        }
