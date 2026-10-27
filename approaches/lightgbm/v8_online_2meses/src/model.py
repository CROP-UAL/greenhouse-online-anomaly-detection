"""
model.py — Entrenamiento, Calibración Conjunta y Evaluación del Modelo LightGBM
================================================================================
Entrena el clasificador binario LightGBM para Detección de Anomalías usando
la API nativa lgb.train() con early stopping en el split de Validación.

Novedades:
─ Calibración conjunta de (Umbral, K_persistencia) en el conjunto de Validación.
─ Selección del par (Umbral*, K*) que maximice Recall de Secuencia manteniendo FA/día <= 10.0.
─ Evaluación desglosada y comparativa (Raw vs Debounced por persistencia).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

import lightgbm as lgb


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS PURAS (NumPy puro)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas_binarias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """Calcula accuracy, precision, recall, F1, F2 y ROC-AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total = len(y_true)
    acc   = (tp + tn) / total if total > 0 else 0.0
    prec  = tp / (tp + fp)  if (tp + fp)  > 0 else 0.0
    rec   = tp / (tp + fn)  if (tp + fn)  > 0 else 0.0
    f1    = (2  * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    f2    = (5  * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        idx = np.argsort(y_proba)[::-1]
        yt  = y_true[idx]
        tps = np.cumsum(yt == 1)
        fps = np.cumsum(yt == 0)
        tpr = tps / n_pos
        fpr = fps / n_neg
        auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))

    return {
        'accuracy':         float(acc),
        'precision':        float(prec),
        'recall':           float(rec),
        'f1_score':         float(f1),
        'f2_score':         float(f2),
        'roc_auc':          float(auc),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN DE SECUENCIAS Y FALSAS ALARMAS DIARIAS
# ─────────────────────────────────────────────────────────────────────────────

def aplicar_persistencia_adaptativa_array(
    p_proba: np.ndarray,
    thresh_std: float = 0.48,
    k_long: int = 5,
    thresh_high: float = 0.85,
    k_short: int = 1,
    physical_out_of_bounds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aplica el Debouncer Multi-Ruta / Adaptativo sobre un array continuo de probabilidades:
      - Ruta de Alta Persistencia: Si P(Anomalía) >= thresh_std durante k_long muestras consecutivas.
      - Ruta de Eventos Efímeros / Pico: Si P(Anomalía) >= thresh_high durante k_short muestras consecutivas.
      - Lógica de Fusión: Combina ambos canales mediante un OR lógico.
    """
    p_proba = np.asarray(p_proba, dtype=float)
    n = len(p_proba)
    pred_debounced = np.zeros(n, dtype=int)
    
    count_std = 0
    count_high = 0
    has_physical = physical_out_of_bounds is not None

    for i in range(n):
        if has_physical and physical_out_of_bounds[i]:
            pred_debounced[i] = 1
            count_std = 0
            count_high = 0
            continue

        prob = p_proba[i]

        if prob >= thresh_std:
            count_std += 1
        else:
            count_std = 0

        if prob >= thresh_high:
            count_high += 1
        else:
            count_high = 0

        is_alert_std  = (count_std >= k_long)
        is_alert_high = (count_high >= k_short)

        if is_alert_std or is_alert_high:
            pred_debounced[i] = 1

    return pred_debounced


# Alias de compatibilidad
aplicar_persistencia_array = aplicar_persistencia_adaptativa_array


def evaluar_secuencias_y_alarmas(
    timestamps:   pd.Series,
    labels:       pd.Series,
    predictions:  np.ndarray,
    anomaly_type: str = 'Sensor Atascado',
) -> Dict[str, Any]:
    """
    Calcula Recall por Secuencia para `anomaly_type`, retardo medio de detección
    y número/tasa diaria de falsas alarmas en periodos normales.
    """
    labels      = pd.Series(labels).reset_index(drop=True)
    timestamps  = pd.to_datetime(timestamps).reset_index(drop=True)
    predictions = np.asarray(predictions, dtype=bool)

    mask   = labels.eq(anomaly_type).to_numpy()
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends   = np.flatnonzero(mask & ~np.r_[mask[1:], False])

    detected = 0
    delays = []
    for start, end in zip(starts, ends):
        hits = np.flatnonzero(predictions[start:end + 1])
        if hits.size:
            detected += 1
            delays.append(float(hits[0]))

    normal_mask = labels.eq('Normal').to_numpy()
    false_positive_samples = int((normal_mask & predictions).sum())
    false_alarm_count = int((normal_mask & predictions & ~np.r_[False, normal_mask[:-1] & predictions[:-1]]).sum())
    
    if len(timestamps) > 0:
        total_days = max(1.0, (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 86400.0)
    else:
        total_days = 1.0

    seq_recall = detected / len(starts) if len(starts) > 0 else 0.0
    fa_per_day = false_alarm_count / total_days

    return {
        'anomaly_type':                anomaly_type,
        'sequences':                   int(len(starts)),
        'sequences_detected':          int(detected),
        'sequence_recall':             float(seq_recall),
        'mean_time_to_detect_samples': float(np.mean(delays)) if delays else np.nan,
        'false_alarm_count':           false_alarm_count,
        'false_positive_samples':       false_positive_samples,
        'false_alarms_per_day':        float(fa_per_day),
        'total_days_evaluated':        float(total_days),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRACIÓN ADAPTATIVA MULTI-RUTA EN VALIDACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def calibrar_umbral_adaptativo(
    y_val:              np.ndarray,
    p_val:              np.ndarray,
    timestamps_val:     pd.Series,
    tipo_anomalia_val:  pd.Series,
    df_val_feat:        Optional[pd.DataFrame] = None,
    min_sequence_recall:float = 0.90,
    target_fa_per_day:  float = 10.0,
    grid_thr_start:     float = 0.30,
    grid_thr_stop:      float = 0.70,
    grid_thr_step:      float = 0.04,
    k_grid:             List[int] = [3, 4, 5, 6, 7],
    thresh_high_grid:   List[float] = [0.75, 0.80, 0.85, 0.90],
    k_short_grid:       List[int] = [1, 2],
    alpha:              float = 0.65,
) -> Tuple[float, int, float, int, pd.DataFrame]:
    """
    Búsqueda en grid conjunta sobre la tupla (thresh_std, k_long, thresh_high, k_short) en Validación.
    Encuentra la combinación que maximiza el Recall global y F1 respetando el techo estricto FA/día <= 10.0.
    """
    y_val = np.asarray(y_val, dtype=int)
    p_val = np.asarray(p_val, dtype=float)
    thresholds_std = np.round(np.arange(grid_thr_start, grid_thr_stop + grid_thr_step / 2, grid_thr_step), 4)

    rows = []
    print(f"\n  [Calibración Adaptativa Multi-Ruta] Evaluando grilla de {len(thresholds_std)} thr_std × {len(k_grid)} k_long × {len(thresh_high_grid)} thr_high × {len(k_short_grid)} k_short...")

    for k_long in k_grid:
        for t_std in thresholds_std:
            for t_high in thresh_high_grid:
                if t_high <= t_std:
                    continue
                for k_short in k_short_grid:
                    pred_k = aplicar_persistencia_adaptativa_array(
                        p_val,
                        thresh_std=t_std,
                        k_long=k_long,
                        thresh_high=t_high,
                        k_short=k_short
                    )
                    
                    tp = int(((y_val == 1) & (pred_k == 1)).sum())
                    fp = int(((y_val == 0) & (pred_k == 1)).sum())
                    fn = int(((y_val == 1) & (pred_k == 0)).sum())
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

                    seq_eval = evaluar_secuencias_y_alarmas(
                        timestamps=timestamps_val,
                        labels=tipo_anomalia_val,
                        predictions=pred_k,
                        anomaly_type='Sensor Atascado',
                    )

                    seq_recall_sa = float(seq_eval['sequence_recall'])
                    fa_per_day = float(seq_eval['false_alarms_per_day'])

                    if fa_per_day <= target_fa_per_day:
                        # FA/day is a hard constraint; maximize recall within it.
                        mo_score = rec
                    else:
                        mo_score = -1.0

                    rows.append({
                        'thresh_std':           float(t_std),
                        'k_long':               int(k_long),
                        'thresh_high':          float(t_high),
                        'k_short':              int(k_short),
                        'precision':            float(prec),
                        'recall_pointwise':     float(rec),
                        'f1_score':             float(f1),
                        'seq_recall_sa':        seq_recall_sa,
                        'false_alarms_per_day': fa_per_day,
                        'multiobjective_score': float(mo_score),
                    })

    df_grid = pd.DataFrame(rows)

    valid_candidates = df_grid[df_grid['multiobjective_score'] >= 0]
    if not valid_candidates.empty:
        best_row = valid_candidates.sort_values(
            by=['recall_pointwise', 'f1_score', 'seq_recall_sa', 'false_alarms_per_day'],
            ascending=[False, False, False, True]
        ).iloc[0]
    else:
        best_row = df_grid.sort_values(
            by=['false_alarms_per_day', 'recall_pointwise', 'f1_score'],
            ascending=[True, False, False]
        ).iloc[0]

    b_std   = float(best_row['thresh_std'])
    b_klong = int(best_row['k_long'])
    b_high  = float(best_row['thresh_high'])
    b_kshort= int(best_row['k_short'])

    print(
        f"\n  Configuración Adaptativa Óptima Calibrada (Val):\n"
        f"    Ruta Persistencia (SA)      : Thresh_std = {b_std:.2f} | K_long = {b_klong} muestras ({b_klong*30}s)\n"
        f"    Ruta Efímera (Ruido/OOB)    : Thresh_high = {b_high:.2f} | K_short = {b_kshort} muestra(s)\n"
        f"    Criterio de selección: recall binario          : {best_row['multiobjective_score']:.4f}\n"
        f"    Episodios de falsa alarma / día         : {best_row['false_alarms_per_day']:.2f} alarmas/día (Objetivo ≤ {target_fa_per_day})\n"
        f"    Precisión Punto a Punto      : {best_row['precision']:.2%}\n"
        f"    Recall Punto a Punto         : {best_row['recall_pointwise']:.2%}\n"
        f"    F1-Score Punto a Punto       : {best_row['f1_score']:.2%}"
    )

    return b_std, b_klong, b_high, b_kshort, df_grid


def calibrar_umbral_y_persistencia(*args, **kwargs):
    """Alias para compatibilidad retroactiva."""
    res = calibrar_umbral_adaptativo(*args, **kwargs)
    b_std, b_klong, b_high, b_kshort, df_grid = res
    return b_std, b_klong, df_grid



calibrar_umbral_f2 = calibrar_umbral_y_persistencia


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN DESGLOSADA POR TIPO DE ANOMALÍA
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_por_tipo_anomalia(
    y_true:        np.ndarray,
    y_pred:        np.ndarray,
    tipo_anomalia: pd.Series,
    threshold:     float,
    k_persistence: int = 1,
) -> pd.DataFrame:
    """
    Desglosa el detector binario por tipo real. El modelo no predice el tipo.
    La precision atribuida reutiliza los FP sobre muestras normales en cada tipo.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tipo   = pd.Series(tipo_anomalia).reset_index(drop=True)

    tipos_anomalia = sorted(t for t in tipo.unique() if t != 'Normal')
    fp_normales = int(((tipo == 'Normal').to_numpy() & (y_pred == 1)).sum())
    rows = []

    for t in tipos_anomalia:
        mask_t = (tipo == t).to_numpy()
        tp = int((mask_t & (y_pred == 1)).sum())
        fn = int((mask_t & (y_pred == 0)).sum())
        fp_total = fp_normales
        
        rec  = tp / (tp + fn)  if (tp + fn)  > 0 else 0.0
        prec = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        rows.append({
            'Tipo': t,
            'Soporte': int(mask_t.sum()),
            'TP': tp, 'FN': fn,
            'Precision_atribuida_FP_normales': round(prec, 4),
            'Recall_por_tipo': round(rec, 4),
            'F1_atribuido': round(f1, 4),
        })

    # Fila GLOBAL
    tp_g   = int(((y_true == 1) & (y_pred == 1)).sum())
    fp_g   = int(((y_true == 0) & (y_pred == 1)).sum())
    fn_g   = int(((y_true == 1) & (y_pred == 0)).sum())
    prec_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0.0
    rec_g  = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
    f1_g   = (2 * prec_g * rec_g) / (prec_g + rec_g) if (prec_g + rec_g) > 0 else 0.0

    rows.append({
        'Tipo': '── GLOBAL BINARIO ──',
        'Soporte': int((y_true == 1).sum()),
        'TP': tp_g, 'FN': fn_g,
        'Precision_atribuida_FP_normales': round(prec_g, 4),
        'Recall_por_tipo': round(rec_g, 4),
        'F1_atribuido': round(f1_g, 4),
    })

    df_res = pd.DataFrame(rows)

    sep = '─' * 75
    mode_str = f"Debounced K={k_persistence}" if k_persistence > 1 else "Raw K=1"
    print(f"\n{sep}")
    print(f"  DETECTOR BINARIO — DESGLOSE POR TIPO REAL ({mode_str} | Umbral={threshold:.2f})")
    print("  El modelo predice Anomalía/Sin anomalía; NO predice el tipo.")
    print(f"  FP normales compartidos para precisión atribuida*: {fp_normales}")
    print(sep)
    print(f"  {'Tipo real':<30} {'Soporte':>8} {'Prec.*':>10} {'Recall':>8} {'F1*':>8}")
    print(sep)
    for _, r in df_res.iterrows():
        marker = ' ←' if r['Tipo'] == 'Sensor Atascado' else ''
        print(
            f"  {r['Tipo']:<30} {r['Soporte']:>8,} "
            f"{r['Precision_atribuida_FP_normales']:>10.2%} {r['Recall_por_tipo']:>8.2%} {r['F1_atribuido']:>8.2%}{marker}"
        )
    print(sep)
    print("  * Prec.* = TP_tipo / (TP_tipo + FP_normales compartidos).")
    print("    Use Recall por tipo para comparar sensibilidad entre tipos; la fila GLOBAL es la métrica binaria principal.")

    return df_res


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO LightGBM CON EARLY STOPPING EN VAL
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_modelo_lightgbm(
    df_train:              pd.DataFrame,
    df_val:                pd.DataFrame,
    feature_cols:          List[str],
    lgb_params:            Dict,
    early_stopping_rounds: int = 60,
    sample_weights:        Optional[np.ndarray] = None,
    model_save_path:       Optional[str] = None,
) -> Tuple[lgb.Booster, Dict, pd.DataFrame]:
    """
    Entrena LightGBM binario con early stopping sobre Validación.
    """
    X_train = df_train[feature_cols].copy()
    y_train = np.ascontiguousarray(df_train['target_anomalia'].values, dtype=np.float32)
    X_val   = df_val[feature_cols].copy()
    y_val   = np.ascontiguousarray(df_val['target_anomalia'].values, dtype=np.float32)

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_val   = X_val.fillna(medians).fillna(0.0)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    print(f"  Train: {len(X_train):,} filas | Anomalías: {n_pos:,} ({n_pos/len(y_train)*100:.2f}%)")
    print(f"  Val  : {len(X_val):,}   filas | Anomalías: {int(y_val.sum()):,} ({y_val.mean()*100:.2f}%)")

    if sample_weights is not None:
        sample_weights = np.ascontiguousarray(sample_weights, dtype=np.float32)
        print(f"  Pesos muestra activos: min={sample_weights.min():.1f} max={sample_weights.max():.1f} media={sample_weights.mean():.2f}")
        dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    else:
        dtrain = lgb.Dataset(X_train, label=y_train)

    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    train_params = {k: v for k, v in lgb_params.items()
                    if k not in ('n_estimators', 'random_state')}
    train_params['seed'] = lgb_params.get('random_state', 42)
    if sample_weights is None:
        train_params['scale_pos_weight'] = n_neg / n_pos if n_pos > 0 else 1.0

    num_rounds = lgb_params.get('n_estimators', 400)

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    booster = lgb.train(
        train_params,
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dval],
        valid_names=['val'],
        callbacks=callbacks,
    )

    p_val  = booster.predict(X_val)
    y_pred = (p_val >= 0.50).astype(int)
    metrics = calcular_metricas_binarias(y_val, y_pred, p_val)

    print(f"\n  Mejor ronda: {booster.best_iteration}")
    print(f"  Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  "
          f"F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")

    importances = booster.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump(booster, f)
        txt_path = model_save_path.replace('.pkl', '.txt')
        booster.save_model(txt_path)
        print(f"\n  Modelo guardado:\n    Pickle: {model_save_path}\n    LGBM  : {txt_path}")

    return booster, metrics, imp_df
