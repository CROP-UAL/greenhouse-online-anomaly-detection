"""
config.py — Configuración para la Adaptación Online LightGBM (Dataset 2 Meses)
=============================================================================
Dataset: sensor_data_raw2.csv (Dataset 2 Meses Limpios)
Muestreo regularizado: 30 segundos
Feature Engineering: Cíclico (sin/cos de la hora, día semana, mes) + Tabular PREVIO a inyección.
Modelo: LightGBM Binario (Detección de Anomalías: Normal vs Anomalía).

Split temporal estricto (81 días):
  Día 1          : Warmup (calentamiento de ventanas rolling y lags)
  Días 2–61      : Train (60 días)
  Días 62–71     : Validación (10 días) — early stopping y calibración umbral
  Días 72–81     : Test (10 días) — evaluación final
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# RUTAS DE PROYECTO
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# Prefer a self-contained project layout, while preserving the current
# repository layout where Dataset is next to v8_online_2meses.
DATA_RAW_PATH = BASE_DIR / 'Dataset' / 'Datos2MesesLimpios' / 'sensor_data_raw2.csv'
if not DATA_RAW_PATH.exists():
    DATA_RAW_PATH = REPO_ROOT / 'Dataset' / 'Datos2MesesLimpios' / 'sensor_data_raw2.csv'

DIR_DATA = BASE_DIR / 'data'
DIR_INTERIM = DIR_DATA / 'interim'
DIR_MODELS = DIR_DATA / 'models'
DIR_PLOTS = DIR_DATA / 'plots'

# Crear directorios automáticamente
for d in [DIR_DATA, DIR_INTERIM, DIR_MODELS, DIR_PLOTS]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE DATOS Y COLUMNAS
# ─────────────────────────────────────────────
MUESTREO_SEGUNDOS   = 30
MUESTRAS_POR_MINUTO = 2

# Columnas de sensores principales para feature engineering e inyección
COLUMNAS_SENSORES = [
    'PCO2EXT', 'PHEXT', 'PRAD', 'PRGINT', 'PTEXT', 'PVV',
    'XCO2I', 'XHINV', 'XTINV'
]

# Columnas excluidas o actuadores no considerados para inyección directa
# Reglas de infraestructura validadas en v7/Test_2_anos.
TIPO_CAIDA_SISTEMA = 'Caída Sistema'
TIPO_FALLO_PARCIAL_SISTEMA = 'Fallo Parcial Sistema'
CAIDA_SISTEMA_MIN_DURACION_MIN = 5.0
FALLO_PARCIAL_SISTEMA_MIN_DURACION_MIN = 5.0
FALLO_PARCIAL_SISTEMA_MIN_SENSORES_NAN = 4
FALLO_PARCIAL_SISTEMA_MIN_SENSORES_CERO = 4
FALLO_PARCIAL_SISTEMA_MIN_SENSORES_NAN_CON_CEROS = 3

# Episodios por split; RNG separado para conservar las inyecciones existentes.
INFRA_EPISODIOS_POR_DIA = 1
INFRA_DURACION_MIN_MIN = 5
INFRA_DURACION_MAX_MIN = 10
INFRA_SEED_OFFSET = 91873

COLUMNAS_EXCLUIDAS = ['UVENT_cen', 'UVENT_lN', 'XTS']

# ─────────────────────────────────────────────
# SPLIT TEMPORAL ESTRICTO (81 días)
# ─────────────────────────────────────────────
# Día 1        → Warmup  (inicialización de ventanas rolling sin NaN)
# Días 2–61    → Train   (60 días)
# Días 62–71   → Val     (10 días, early stopping y calibración umbral)
# Días 72–81   → Test    (10 días, evaluación final no vista)
N_DIAS_WARMUP = 1
N_DIAS_TRAIN  = 60
N_DIAS_VAL    = 10
N_DIAS_TEST   = 10

# ─────────────────────────────────────────────
# RANGOS FÍSICOS DE SENSORES (GUARDIA DETERMINISTA)
# ─────────────────────────────────────────────
RANGOS_FISICOS = {
    'PCO2EXT': {'min': 300.0, 'max':  800.0},
    'PHEXT':   {'min':   0.0, 'max':  100.0},
    'PRAD':    {'min':   0.0, 'max': 1500.0},
    'PRGINT':  {'min':   0.0, 'max': 1200.0},
    'PTEXT':   {'min': -20.0, 'max':   60.0},
    'PVV':     {'min':   0.0, 'max':   50.0},
    'XCO2I':   {'min': 200.0, 'max': 3000.0},
    'XHINV':   {'min':   0.0, 'max':  100.0},
    'XTINV':   {'min':  -5.0, 'max':   60.0},
}

# ─────────────────────────────────────────────
# VENTANAS PARA FEATURES DE ESTATISMO (SENSOR ATASCADO)
# ─────────────────────────────────────────────
# A 30s/muestra: 10 muestras = 5 min, 30 muestras = 15 min
FE_W_STUCK_5M  = 10   # muestras equivalentes a 5 min (features ML)
FE_W_STUCK_15M = 30   # muestras equivalentes a 15 min (features ML)
FE_EPS_STUCK   = 1e-6 # umbral de varianza nula para stuck_flag en FEATURES ML

# Guardia determinista de Sensor Atascado (streaming detector):
# Ventana MAYOR para evitar falsas alarmas en señales naturalmente estables
# (PRAD=0 toda la noche, PCO2EXT en atmósfera tranquila, etc.).
# Se usa solo como guardia en process_sample(); NO en el feature engineering batch.
STUCK_GUARDIA_WINDOW  = 60    # 30 min @ 30s/muestra — para confirmar stuck real
STUCK_GUARDIA_EPSILON = 0.05  # ruido mínimo realista para sensores de precisión analógica

# ─────────────────────────────────────────────
# VENTANAS ROLLING PRINCIPALES
# ─────────────────────────────────────────────
# 30 min = 60 muestras, 3h = 360 muestras
FE_W_30M = 60
FE_W_3H  = 360
FE_W_30M_MIN_PERIODS = 5
FE_W_3H_MIN_PERIODS  = 30

# Lags (en muestras): 1 muestra = 30s, 30 muestras = 15 min
FE_LAG1  = 1
FE_LAG30 = 30

# ─────────────────────────────────────────────
# INYECCION DE ANOMALÍAS (SINTÉTICAS)
# ─────────────────────────────────────────────
INYECCION_CONFIG = {
    'datos_faltantes': {
        'porcentaje_filas': 0.02,
        'num_sensores_por_fila': 1,
    },
    'sensor_atascado': {
        # Densidad adaptativa por días: 3.33 secuencias/día (~200 secuencias en 60d Train, ~33 en 10d Val/Test)
        # Esto mantiene una tasa de anomalías homogénea (~10% a 12%) entre Train, Val y Test, evitando Data Drift.
        'secuencias_por_dia': 3.33,
        'num_secuencias': 200, # Base para 60 días
        'duracion_min': 6,    # 6 min mínimo → 12 muestras; stuck_flag_5m se activa ≥2 muestras
        'duracion_max': 25,   # minutos máximos de stuck
    },
    'ruido': {
        'porcentaje_filas': 0.03,
        'factor_std_min': 3.0,
        'factor_std_max': 6.0,
        'num_sensores_por_fila': 1,
    },
    'fuera_rango': {
        'porcentaje_filas': 0.02,
        'num_sensores_por_fila': 1,
        'margen_error_factor': 0.15,
    },
    'desviacion_correlacion': {
        'porcentaje_filas': 0.03,
        'filtro_hora_inicio': 7,
        'filtro_hora_fin': 18,
    },
    'contextual_luz': {
        'porcentaje_filas': 0.01,
        'hora_noche_inicio': 23,
        'hora_noche_fin': 4,
        'prad_umbral_cero': 1.0,
        'prgint_min': 20.0,
        'prgint_max': 50.0,
    }
}

# Semillas de inyección por split (para reproducibilidad y evitar data leakage)
SEED_INYECCION_TRAIN = 42
SEED_INYECCION_VAL   = 1234
SEED_INYECCION_TEST  = 20260918

# ─────────────────────────────────────────────
# CONFIGURACIÓN DEL MODELO LIGHTGBM (Optimizado para Drift & Recall SA)
# ─────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    # AUC optimiza el ranking (separación normal/anomalía)
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 400,        # más rounds con early stopping
    'learning_rate': 0.04,
    'num_leaves': 31,           # reducido de 63 a 31 para controlar complejidad y prevenir overfitting/drift
    'subsample': 0.70,          # subsample reducido para regularización
    'colsample_bytree': 0.70,   # colsample reducido para regularización
    'min_child_samples': 25,    # incrementado de 10 a 25 para evitar hojas sobreajustadas
    'max_depth': 6,             # profundidad máxima controlada
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Early stopping: rondas sin mejora en AUC de val antes de parar
EARLY_STOPPING_ROUNDS = 60

# ─────────────────────────────────────────────
# LISTA DE PARES DE SENSORES (FEATURES INTER-SENSOR)
# ─────────────────────────────────────────────
INTER_SENSOR_PAIRS = [
    ('PRAD', 'PRGINT'),
    ('XTINV', 'PTEXT'),
    ('XHINV', 'PHEXT'),
    ('XCO2I', 'PCO2EXT')
]

# ─────────────────────────────────────────────
# PESOS POR MUESTRA EN ENTRENAMIENTO (POLÍTICA HOMOGÉNEA)
# ─────────────────────────────────────────────
# Política de pesos balanceada/homogénea (SA_SAMPLE_WEIGHT = 1.0)
SA_SAMPLE_WEIGHT     = 2.0   # Sensor Atascado: refuerzo moderado de recall
OTHER_ANOM_WEIGHT   = 1.0   # Otras anomalías: peso homogéneo
NORMAL_SAMPLE_WEIGHT = 1.0   # Normal: peso base

# ─────────────────────────────────────────────
# DETECTOR DE STREAMING ADAPTATIVO (MULTI-RUTA DEBOUNCER)
# ─────────────────────────────────────────────
# Ruta A: Alta Persistencia / Sensor Atascado
K_LONG     = 3
THRESH_STD = 0.20

# Ruta B: Eventos Efímeros / Ruido / Fuera de Rango
K_SHORT     = 1
THRESH_HIGH = 0.85

# Calibración multiobjetivo
K_PERSISTENCE_MIN   = 2
K_PERSISTENCE_GRID  = [2, 3, 4]
ALPHA_MULTIOBJECTIVE = 0.65   # Priorizar Recall respetando techo FA/día
TARGET_FA_PER_DAY   = 10.0   # Techo estricto de falsas alarmas diarias en TEST

# Grid de umbrales a explorar en calibración
THRESHOLD_GRID_START = 0.20
THRESHOLD_GRID_STOP  = 0.55
THRESHOLD_GRID_STEP  = 0.02

# Umbral por defecto antes de calibrar
ONLINE_THRESHOLD = 0.48


