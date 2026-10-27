"""
config.py - Configuración central de v6_palmerillas_vent
====================================================
Dataset: Las Palmerillas (IFAPA, Almería, 2019-2023)
Campaña Almería: mediados agosto -> finales mayo/principios julio

Reparto temporal de esta carpeta Test_2_anos:
  Preparacion/inyeccion : 15 ago 2019 -> 12 may 2021
  Train supervisado     : 15 ago 2019 -> 31 ene 2021
  Validacion sintetica  : 01 feb 2021 -> 12 may 2021
  Test real no visto    : reservado fuera de esta carpeta/experimento
  Excluir: periodos sin cultivo, sensores fuera de contexto
"""

import os

# -----
# RUTAS
# -----

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# Test_2_anos vive dentro de v7_palmerillas_vent_null. Los CSV raw siguen
# estando en la carpeta data del proyecto padre Clasificaci-n-y-correcci-n.
DATA_RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))
DATA_INTERIM = os.path.join(BASE_DIR, 'data', 'interim')
DATA_MODELS   = os.path.join(BASE_DIR, 'data', 'models')
DATA_PLOTS    = os.path.join(BASE_DIR, 'data', 'plots')
DATA_RESULTS  = os.path.join(BASE_DIR, 'data', 'results')

DATASET_NAME        = 'palmerillas_all_data_30s_null'
MUESTRAS_POR_MINUTO = 2
MUESTREO_SEGUNDOS   = int(60 / MUESTRAS_POR_MINUTO)
DATASET_PATH        = os.path.join(DATA_RAW_DIR, f'{DATASET_NAME}.csv')
COLUMNA_USAR_EN_TRAIN = 'usar_en_train'
CSV_NA_VALUES = ['null', 'NULL']
CSV_FECHA_FORMAT = '%d/%m/%Y %H:%M:%S'
VALIDACION_CSV_ANIO_MIN = 2019
VALIDACION_CSV_ANIO_MAX = 2023
VALIDACION_CSV_MEDIANAS = {
    'PCO2EXT': {'min': 350.0, 'max': 600.0, 'esperado': '380-540 ppm', 'unidad': 'ppm'},
    'XTINV':   {'min':   8.0, 'max':  45.0, 'esperado': '15-35 C',    'unidad': 'C'},
}

# Ventana de preparacion para 01/02/03: solo train + validacion sinttica.
# El test real no visto NO debe inyectarse en esta fase.
FECHA_INICIO = '2019-08-15'
FECHA_FIN    = '2021-05-12 23:59:30'

# Reservado para futuras inferencias reales; no se usa en 01/02/03 de este experimento.
FECHA_TEST_INICIO = None
FECHA_TEST_FIN    = None

# FECHA_TEST_INICIO = '2022-08-15'
# FECHA_TEST_FIN    = '2023-06-22'
# -----
# COLUMNAS DEL DATASET
# -----

# Sensores objetivo de inyección y reglas deterministas.
COLUMNAS_SENSORES = [
    'PCO2EXT', 'PHEXT', 'PRAD', 'PRGINT', 'PTEXT', 'PVV',
    'XCO2I', 'XHINV', 'XTINV',
]
VALIDACION_CSV_COLUMNAS_ESPERADAS = COLUMNAS_SENSORES

# UVENT eliminado de features: importancia RF = 0.40% (posiciones 257-388/404).
# Los datos son poco fiables (sensor problemático en Palmerillas) y el modelo
# los ignoraba por completo. Ver bitácora 2026-04-27.
COLUMNAS_CONTEXTO_MODELO = []
COLUMNAS_VENTILACION     = ['UVENT_cen', 'UVENT_lN']  # solo para inyección CO2 contextual

COLUMNAS_EXCLUIR_FEATURES = [
    'Fecha', 'etiqueta_deteccion', 'etiqueta_tipo_anomalia',
    'sensor_anomalia', 'anomalia_id',
    COLUMNA_USAR_EN_TRAIN,
    'XTS',
    'UVENT_cen', 'UVENT_lN',
]

ETIQUETA_NORMAL   = 'normal'
ETIQUETA_ANOMALIA = 'anomalia'

TIPOS_ANOMALIA = [
    'Caida Sistema',
    'Fallo Parcial Sistema',
    'Datos Faltantes',
    'Sensor Atascado',
    'Ruido',
    'Valores Fuera de Rango',
    'Desviacion de Correlacion',
    'Contextual'
]

# Anomalia determinista de infraestructura: todas las variables objetivo sin dato.
# No se reconstruye ni se usa como caso de entrenamiento para correctores.
TIPO_CAIDA_SISTEMA = 'Caida Sistema'
TIPO_FALLO_PARCIAL_SISTEMA = 'Fallo Parcial Sistema'
CAIDA_SISTEMA_MIN_DURACION_MIN = 5.0
FALLO_PARCIAL_SISTEMA_MIN_SENSORES_NAN = 4
FALLO_PARCIAL_SISTEMA_MIN_SENSORES_CERO = 4
FALLO_PARCIAL_SISTEMA_MIN_SENSORES_NAN_CON_CEROS = 3
FALLO_PARCIAL_SISTEMA_MIN_DURACION_MIN = 5.0

# -----
# RANGOS FSICOS DE HARDWARE
# -----

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

# -----
# PERODOS SIN CULTIVO (veranos Almería)
# -----
# Campaña: ~15 ago -> ~31 may / principios jul
# Verano: ~1 jun -> ~14 ago - sensores sin contexto operativo

PERIODOS_SIN_CULTIVO = [
    ('2019-06-01', '2019-08-14'),
    ('2020-06-01', '2020-08-14'),
    ('2021-06-01', '2021-08-14'),
    ('2022-06-01', '2022-08-14'),
    ('2023-06-01', '2023-08-14'),
]

# -----
# INYECCIÓN DE ANOMALAS
# -----

INYECCION = {
    'datos_faltantes': {
        'porcentaje_filas': 0.02,
        'num_sensores_por_fila': 1,
    },
    'sensor_atascado': {
        'num_secuencias': 270,
        'duracion_min': 8,
        'duracion_max': 20,
        # v8: cuotas temporales por sensor para que el fold final tenga
        # cobertura real de todos los sensores. No relaja filtros; si un sensor
        # no tiene ventanas limpias suficientes en train/test, se crean menos
        # y se avisa.
        # En Test_2_anos la clave historica "test" del inyector se usa como
        # validacion sintetica. No es test real.
        'fecha_inicio_test_final': '2021-02-01 00:00:00',
        'secuencias_por_sensor_temporal': {
            'train': {
                'PCO2EXT': 40,  # doblar: más cobertura estacional en 3 años de training
                'PHEXT':   40,
                'PRGINT':  40,
                'PTEXT':   40,
                'PVV':     40,
                'XCO2I':   40,  # 41% recall - más training ayuda
                'XHINV':   40,  # 50% recall - ídem
                'XTINV':   40,
            },
            'test': {
                'PCO2EXT': 10,
                'PHEXT':   10,
                'PRGINT':  10,
                'PTEXT':   10,
                'PVV':     10,
                'XCO2I':   10,
                'XHINV':   10,
                'XTINV':   10,
            },
        },
        # Modo legado (solo si no existe secuencias_por_sensor_temporal):
        # objetivo global por sensor.
        'secuencias_por_sensor': None,
        # Experimento v6: reequilibrar SA hacia sensores con poco soporte.
        # Se interpretan como pesos relativos; el notebook los normaliza.
        # Solo se usa si no existe secuencias_por_sensor.
        'pesos_por_sensor': {
            'PTEXT':   0.20,
            'PCO2EXT': 0.15,
            'XHINV':   0.15,
            'XTINV':   0.10,
            'PVV':     0.08,
            'PHEXT':   0.08,
            'PRGINT':  0.07,
            'XCO2I':   0.07,
        },
    },
    'ruido': {
        'porcentaje_filas': 0.03,   # restaurado a 036: 0.02 no aportaba mejora y era peor con no-determinismo
        'factor_std_min': 3,
        'factor_std_max': 6,
        'num_sensores_por_fila': 1,
    },
    'fuera_rango': {
        'porcentaje_filas': 0.02,
        'num_sensores_por_fila': 1,
        'margen_error_factor': 0.1,
        # Cuotas temporales por sensor: garantiza cobertura en el fold de test
        # para los 3 sensores interiores (XCO2I/XHINV/XTINV) que con inyección
        # aleatoria por porcentaje caían con 0 instancias en test.
        # En Test_2_anos la clave historica "test" del inyector se usa como
        # validacion sintetica. No es test real.
        'fecha_inicio_test_final': '2021-02-01 00:00:00',
        'instancias_por_sensor_temporal': {
            'train': {
                'PCO2EXT': 3500, 'PHEXT': 3500, 'PRAD': 3500, 'PRGINT': 3500,
                'PTEXT':   3500, 'PVV':  3500, 'XCO2I': 3500, 'XHINV':  3500, 'XTINV': 3500,
            },
            'test': {
                'PCO2EXT': 900, 'PHEXT': 900, 'PRAD': 900, 'PRGINT': 900,
                'PTEXT':   900, 'PVV':  900, 'XCO2I': 900, 'XHINV':  900, 'XTINV': 900,
            },
        },
    },
    'desviacion_correlacion': {
        'porcentaje_filas': 0.03,
        'filtro_hora_inicio': 7,
        'filtro_hora_fin': 18,
        # Cuotas temporales por par: garantiza cobertura en test para todos los pares,
        # especialmente (XCO2I, UVENT_*) que con inyección aleatoria obtenían solo ~55 en test.
        # En Test_2_anos la clave historica "test" del inyector se usa como
        # validacion sintetica. No es test real.
        'fecha_inicio_test_final': '2021-02-01 00:00:00',
        # Pares sustituidos: (XCO2I/XTINV, UVENT_*) descartados porque UVENT=0 la
        # mayor parte del tiempo -> p25~=p75~=0 -> inyeccion siempre falla.
        # Nuevos pares con correlaciones fisicas reales y percentiles bien distribuidos:
        #   (PTEXT, XTINV): temp exterior<->interior  corr~=0.94
        #   (PHEXT, XHINV): humedad exterior<->interior corr~=0.75
        'instancias_por_par_temporal': {
            'train': {
                ('PRAD',  'PRGINT'): 500,   # 200->500: más cobertura estacional
                ('XTINV', 'XHINV'):  500,   # 200->500: ídem
                ('PTEXT', 'XTINV'):  800,   # 400->800: par con mayor FN en 034-035
                ('PHEXT', 'XHINV'):  800,   # 400->800: ídem
            },
            'test': {
                ('PRAD',  'PRGINT'): 200,   # 80->200: estadística más robusta por par
                ('XTINV', 'XHINV'):  200,
                ('PTEXT', 'XTINV'):  200,
                ('PHEXT', 'XHINV'):  200,
            },
        },
    },
    # Tipo 1: PRAD alto exterior pero PRGINT anómalamente bajo interior
    # Simula cubierta sucia / obstrucción parcial de luz
    # Solo de día (PRAD > prad_min) -> el modelo aprende la relación, no el horario
    'contextual_prad_prgint': {
        'porcentaje_filas': 0.020,   # prueba contextual: subir cobertura de validacion
        'prad_min': 200.0,           # solo días con radiación notable
        'ratio_normal_min': 0.40,    # candidatos: PRGINT/PRAD actualmente > 40% (estado normal)
        'ratio_anomalo_max': 0.08,   # inyectar PRGINT = PRAD x (2-8%)
    },
    # Tipo 2: CO2 interior alto cuando ventilación está abierta
    # Simula fallo de ventilación: UVENT dice "abierta" pero CO2 no baja
    'contextual_co2_vent': {
        'num_secuencias': 90,
        'umbral_vent_abierta': 40.0,
        'prad_min_diurno': 50.0,     # al menos algo de radiación (diurno)
        'co2_anomalo_min': 700,
        'co2_anomalo_max': 1200,
        'duracion_min': 3,
        'duracion_max': 8,
        # En Test_2_anos la clave historica "test" del inyector se usa como
        # validacion sintetica. No es test real.
        'fecha_inicio_test_final': '2021-02-01 00:00:00',
        'secuencias_por_split': {'train': 100, 'test': 60},
    },
    # Tipo 3: temperatura interior < exterior con alta radiación solar
    # Simula pérdida térmica inesperada: XTINV < PTEXT cuando PRAD > 400
    'contextual_temp_inv': {
        'porcentaje_filas': 0.016,   # prueba contextual: subir cobertura de validacion
        'prad_min': 400.0,           # alta radiación (día soleado)
        'ptext_min': 20.0,           # exterior cálido
        'delta_min': 3.0,            # XTINV = PTEXT - (3-10)C
        'delta_max': 10.0,
    },
}

# -----
# FEATURE ENGINEERING
# -----

FE_SENSORES       = COLUMNAS_SENSORES + COLUMNAS_CONTEXTO_MODELO
FE_SENSORES_STUCK = COLUMNAS_SENSORES.copy()
FE_VENTANAS = {
    'rolling_30m': 60,
    'rolling_3h':  360,
}

FE_W_STUCK  = 10
FE_W_STUCK2 = 20
FE_ROLLING_30M_MIN_PERIODS = 5
FE_ROLLING_3H_MIN_PERIODS = 30
FE_LAG_30M = FE_VENTANAS['rolling_30m']
FE_EPS = 1e-6

FE_STUCK_MIN_PERIODS = 3
FE_STUCK_ACCEL_MIN_PERIODS = 2
FE_STUCK_PREV_WINDOW_MULT = 3
FE_STUCK_PREV_MIN_PERIODS = 5
FE_PRAD_ACTIVO_MIN = 10.0
FE_TOP_VECINOS_STUCK = {
    'PCO2EXT': ['PTEXT',  'PHEXT'],
    'PHEXT':   ['XHINV',  'PRAD'],
    'PRAD':    ['PRGINT', 'XHINV'],
    'PRGINT':  ['PRAD',   'XHINV'],
    'PTEXT':   ['XTINV',  'PHEXT'],
    'PVV':     ['PHEXT',  'PRAD'],
    'XCO2I':   ['PTEXT',  'XHINV'],
    'XHINV':   ['XTINV',  'PRGINT'],
    'XTINV':   ['PTEXT',  'XHINV'],
}

FE_PARES_DELTA_REF = {
    'XHINV':  ('XTINV',  [15, 30]),
    'PTEXT':  ('XTINV',  [15, 30]),
    'PRGINT': ('PRAD',   [15, 30]),
}

FE_CICLICA_PERIODOS = {
    'Mes': 12,
    'Hora': 24,
    'DiaSemana': 7,
    'DiaAnyo': 365,
}

FE_NOCHE_HORA_MIN = 21
FE_NOCHE_HORA_MAX = 6
FE_HORA_ACTIVA_MIN = 7
FE_HORA_ACTIVA_MAX = 19
FE_MESES_INVIERNO = [11, 12, 1, 2]
FE_MESES_TRANSICION = [3, 4, 9, 10]
FE_PRGINT_PRAD_RSTD_MIN = 2.0
FE_ROLLING_PRAD_PRGINT_MIN_PERIODS = 20
FE_VENT_APERTURA_OFFSET = 5.0
FE_RATIO_RAD_OFFSET = 1.0

FE_M7_WINDOW = 40
FE_M7_MIN_PERIODS = 5
FE_ZERO_DIFF_EPS = 1e-9

# -----
# HIPERPARMETROS DE MODELOS
# -----

MODELO_1_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
}

UMBRAL_DETECCION = 0.80

MODELO_2_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
}

MODELO_1_LGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'subsample': 0.90,
    'colsample_bytree': 0.90,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'verbosity': -1,
}

MODELO_2_LGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'subsample': 0.90,
    'colsample_bytree': 0.90,
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'verbosity': -1,
}

TSCV_N_SPLITS = 4
RANDOM_STATE = 42

# -----
# REGLAS POST-PROCESADO SENSOR ATASCADO
# -----

STUCK_SENSORES_REGLA      = ['PCO2EXT', 'PVV']
STUCK_CONSECUTIVOS_MIN    = 10
STUCK_EXCLUIR_VALOR_CERO  = ['PVV']

STUCK_VALOR_MINIMO_INYECCION = {
    'PVV':    0.5,
    'PRGINT': 10.0,
}

STUCK_FILTROS_CONTEXTUALES = {
    'PRGINT': {'ref_col': 'PRAD',  'ref_min':  5.0},
    'XTINV':  {'ref_col': 'PTEXT', 'ref_min':  5.0},
    'XHINV':  {'ref_col': 'XTINV', 'ref_min': 10.0},
    'PTEXT':  {'ref_col': 'PRAD',  'ref_min': 10.0},
}

STUCK_REF_VARIABILIDAD_MIN = {
    'PRGINT': 2.0,
    'XTINV':  0.1,   # 0.3->0.1: PTEXT no necesita variar mucho para validar XTINV SA
    'XHINV':  0.5,
    'PTEXT':  0.0,   # 1.0->0.0: solo exigir PRAD>10 (diurno), no que varíe
}

STUCK_DURACION_POR_SENSOR = {
    # Duración en samples a 30s/muestra - derivada del análisis de plateaus naturales
    # en Palmerillas (2019-2022) solo durante períodos activos (filtros SA aplicados).
    # Criterio: duracion_min > MAX plateau natural activo para evitar ambigüedad.
    # -----
    # Sensor     MAX natural activo   P99 activo   duracion_min (x2 margen)
    # PCO2EXT    10 s (5 min)          5 s           6  -> 12 (exec006) -> 20 (exec013)
    #            Criterio aplicado: x2 del MAX natural, igual que PRGINT/XTINV.
    #            Margen anterior (12) era solo x1.2 del MAX, el más bajo de todos los sensores.
    #            Descartado filtro contextual PHEXT: correlacion PCO2EXT-PHEXT = 0.06 (casi nula).
    #            El CO2 exterior y la humedad exterior no estan fisicamente acoplados.
    # PRAD excluido del protocolo final de Sensor Atascado: sus plateaus
    # diurnos pueden ser comportamiento fisico normal y dieron bajo recall.
    # PRGINT     12 s (6 min)          6 s           30 [OK]
    # PVV        15 s (7.5 min)        4 s           20 [OK]
    # XCO2I      9 s (4.5 min)         5 s           20 [OK]
    # PHEXT      29 s (14.5 min)       9 s           32 -> subido desde 16
    # XHINV      24 s (12 min)        11 s           28 -> subido desde 16
    # PTEXT      24 s (12 min)        10 s           28 -> subido desde 8
    # XTINV      27 s (13.5 min)      14 s           30 -> subido desde 8
    'PCO2EXT': {'duracion_min': 20, 'duracion_max': 40},
    'PTEXT':   {'duracion_min': 16, 'duracion_max': 40},   # 28->16, 45->40: rango más estrecho para dataset con muchos NaN
    'XTINV':   {'duracion_min': 20, 'duracion_max': 60},   # 30->20, 80->60: ídem, facilita encontrar ventanas limpias
    'XHINV':   {'duracion_min': 28, 'duracion_max': 70},
    'PHEXT':   {'duracion_min': 32, 'duracion_max': 80},
    'PVV':     {'duracion_min': 20, 'duracion_max': 60},
    'XCO2I':   {'duracion_min': 20, 'duracion_max': 60},
    'PRGINT':  {'duracion_min': 20, 'duracion_max': 60},   # 30->20, 90->60: NaN dispersos (6.95%) rompen ventanas largas; P(20 limpias)~=23% vs 12% con 30
}

# Separación mínima (en múltiplos de sigma) entre el stuck value y el promedio
# de la ventana de inyección. Evita inyecciones "degeneradas" donde el atasco
# queda al mismo nivel que la señal real -> err_inj~=0 -> corrección siempre parece
# sobre-corrección aunque sea correcta.
STUCK_MIN_SIGMA_SEPARACION = 0.5

RUIDO_VALOR_MINIMO_INYECCION = {
    'PRGINT':  100.0,   # subido 10->100: FN tenían spike=6 W/m2 vs TP=639 W/m2 (inyección nocturna inútil)
    'PVV':       5.0,   # revertido 10->5: con 10 m/s solo había 11 instancias test (fold sin viento fuerte)
    'PCO2EXT': 400.0,   # evita spikes desde CO2 bajo (400->450 ppm parece variacion outdoor)
}

# Sensores donde el ruido solo debe inyectarse como spike positivo (hacia arriba).
# Un spike negativo simula fenómeno físico plausible (calma de viento, nube pasando,
# descenso de radiación) y el modelo lo clasifica correctamente como normal -> FN inevitable.
RUIDO_SOLO_POSITIVO = ['PVV', 'PRGINT', 'PRAD']  # PRAD añadido: val_FN=13 vs val_TP=1260 (spike negativo = nube)

DATOS_FALTANTES_VALOR_MINIMO_INYECCION = {
    'PVV':    3.0,   # solo inyectar NaN cuando haya viento real (0 = calma natural)
    'PRGINT': 50.0,  # solo inyectar NaN cuando haya radiacion interior real (0 = noche)
}

STUCK_ACTIVIDAD_PREVIA = {
    'XTINV':  {'ventana': 10, 'std_min': 0.20},   # 0.50->0.20: dataset _null tiene muchos NaN; umbral alto bloqueaba ventanas
    'PTEXT':  {'ventana': 20, 'std_min': 0.20},   # 0.50->0.20: ídem, temp exterior varía lentamente en otoño/invierno
    'XHINV':  {'ventana': 10, 'std_min': 0.50},
    'PRGINT': {'ventana': 10, 'std_min': 2.00},
}

# Franjas horario-estacionales excluidas de la inyeccion SA por sensor.
# PRAD queda fuera del protocolo final de Sensor Atascado; PRGINT se mantiene
# porque su recall de deteccion SA fue alto en el analisis por sensor.
STUCK_EXCLUIR_FRANJA_TEMPORAL = {}

# -----
# ETIQUETADO DIRECTO (validado con Palmerillas)
# -----
# En v5 entrenamos CON Palmerillas, así que el etiquetado directo
# solo aplica a anomalías de fábrica conocidas del propio dataset.

PRGINT_ETIQUETADO_DIRECTO = {
    'consecutive_min': 10,
    'prad_min':        10.0,
    'prad_rstd_min':    2.0,
    'prgint_min':      10.0,
}

XHINV_ETIQUETADO_DIRECTO = {
    'consecutive_min': 30,
    'xhinv_min':       10.0,
    'xhinv_max':       98.0,
}

PTEXT_ETIQUETADO_DIRECTO = {
    'aplicar':          False,
    'consecutive_min':  20,
    'phext_rstd_min':   0.2,
}

# -----
# CORRECCIÓN
# -----

FACTOR_UMBRAL_CORRECCION = 2.0

# Umbrales mínimos de duración (samples) para DETECCIÓN en corrección.
# Por defecto se usa STUCK_DURACION_POR_SENSOR[col]['duracion_min'].
# Solo se sobreescribe cuando el umbral de inyección es demasiado permisivo
# y genera falsos positivos en la corrección (plateaus naturales).
STUCK_DURACION_MIN_CORRECCION = {
    # Sobreescribir umbral de inyección solo cuando el umbral de corrección
    # necesita ser más conservador que duracion_min de inyección.
    # PRAD queda fuera de la inyeccion SA final; si aparece en inferencia real
    # como posible SA, se mantiene excluido de correccion automatica.
}

# Sensores excluidos de la corrección de Sensor Atascado.
# La detección puede ser correcta (alto recall) pero la RECONSTRUCCIÓN del valor
# verdadero es imposible -> corregir empeora más de lo que mejora.
#   PRAD: fuera de la inyeccion SA final por bajo recall y ambiguedad fisica.
#   PRGINT: se mantiene en la prueba de Sensor Atascado porque su recall de
#           deteccion es alto. La reconstruccion se evalua separadamente.
STUCK_SA_EXCLUIR_CORRECCION = ['PRAD']

IMPUTER_RF_PARAMS = {
    'n_estimators': 30,
    'random_state': 42,
    'max_depth':    10,
    'min_samples_leaf': 5,
    'n_jobs': -1,
}
IMPUTER_MAX_ITER = 10

# -----
# RUTAS DE FICHEROS DE SALIDA
# -----

PARQUET_01 = os.path.join(DATA_INTERIM, '01_datos_cargados.parquet')
PARQUET_02 = os.path.join(DATA_INTERIM, '02_datos_inyectados.parquet')
PARQUET_03 = os.path.join(DATA_INTERIM, '03_datos_features.parquet')
PARQUET_04 = os.path.join(DATA_INTERIM, '04_modelo1_predicciones.parquet')
PARQUET_04_LGBM = os.path.join(DATA_INTERIM, '04b_modelo1_predicciones_lightgbm.parquet')
PARQUET_06 = os.path.join(DATA_INTERIM, '06_datos_corregidos.parquet')
PARQUET_06E_LGBM = os.path.join(DATA_INTERIM, '06e_datos_corregidos_v4_lightgbm.parquet')
PARQUET_06F_RECON_LGBM = os.path.join(DATA_INTERIM, '06f_datos_corregidos_lgbm_regressor.parquet')

MODELO_1_PATH     = os.path.join(DATA_MODELS, 'modelo_1_detector.joblib')
IMPUTER_M1_PATH   = os.path.join(DATA_MODELS, 'imputer_modelo_1.joblib')
FEATURES_M1_PATH  = os.path.join(DATA_MODELS, 'features_modelo_1.joblib')
MODELO_2_PATH     = os.path.join(DATA_MODELS, 'modelo_2_clasificador.joblib')
LABEL_ENC_M2_PATH = os.path.join(DATA_MODELS, 'label_encoder_modelo_2.joblib')
IMPUTER_FALT_PATH = os.path.join(DATA_MODELS, 'imputer_datos_faltantes.joblib')

MODELO_1_LGBM_PATH     = os.path.join(DATA_MODELS, 'modelo_1_detector_lightgbm.joblib')
IMPUTER_M1_LGBM_PATH   = os.path.join(DATA_MODELS, 'imputer_modelo_1_lightgbm.joblib')
FEATURES_M1_LGBM_PATH  = os.path.join(DATA_MODELS, 'features_modelo_1_lightgbm.joblib')
BASELINES_CTX_LGBM_PATH = os.path.join(DATA_MODELS, 'baselines_ctx_lightgbm.joblib')
MODELO_2_LGBM_PATH     = os.path.join(DATA_MODELS, 'modelo_2_clasificador_lightgbm.joblib')
LABEL_ENC_M2_LGBM_PATH = os.path.join(DATA_MODELS, 'label_encoder_modelo_2_lightgbm.joblib')

# -----
# ARTEFACTOS Y REGLAS FULL_V3
# -----
# Configuracion historica full_v3, integrada aqui con nombres
# versionados para no sobrescribir los artefactos base usados por otros
# notebooks.

EXPERIMENTO_FINAL = 'full_v3'
EXPERIMENTO_TEMPORAL_PAPER_V3 = 'temporal_paper_v3'

FECHA_TRAIN_INICIO_TEMPORAL_PAPER_V3 = '2019-08-15'
FECHA_TRAIN_FIN_TEMPORAL_PAPER_V3 = '2021-01-31 23:59:59'
FECHA_VAL_INICIO_TEMPORAL_PAPER_V3 = '2021-02-01'
FECHA_VAL_FIN_TEMPORAL_PAPER_V3 = '2021-05-12 23:59:30'

PARQUET_02_FULL_V3 = os.path.join(DATA_INTERIM, '02_datos_inyectados_full_v3.parquet')
PARQUET_03_FULL_V3 = os.path.join(DATA_INTERIM, '03_datos_features_full_v3.parquet')
PARQUET_04_LGBM_FULL_V3 = os.path.join(
    DATA_INTERIM, '04b_modelo1_predicciones_lightgbm_full_v3.parquet'
)
PARQUET_06E_LGBM_FULL_V3 = os.path.join(
    DATA_INTERIM, '06e_datos_corregidos_v4_lightgbm_full_v3.parquet'
)
PARQUET_08_REAL_FULL_V3 = os.path.join(DATA_INTERIM, '08_datos_corregidos_real_full_v3.parquet')
PARQUET_04_TEMPORAL_PAPER_V3 = os.path.join(DATA_INTERIM, '04_modelo1_predicciones_temporal_paper_v3.parquet')
PARQUET_05_TEMPORAL_PAPER_V3 = os.path.join(DATA_INTERIM, '05_modelo2_predicciones_temporal_paper_v3.parquet')
PARQUET_06_TEMPORAL_PAPER_V3 = os.path.join(DATA_INTERIM, '06_datos_corregidos_temporal_paper_v3.parquet')

CSV_01_AUDITORIA_CORTES_RESUMEN_FULL_V3 = os.path.join(
    DATA_RESULTS, '01_auditoria_cortes_sistema_resumen_full_v3.csv'
)
CSV_01_AUDITORIA_CORTES_SENSOR_FULL_V3 = os.path.join(
    DATA_RESULTS, '01_auditoria_cortes_sistema_por_sensor_full_v3.csv'
)
CSV_01_AUDITORIA_GAPS_OPERATIVOS_FULL_V3 = os.path.join(
    DATA_RESULTS, '01_auditoria_gaps_temporales_operativos_full_v3.csv'
)

MODELO_1_LGBM_FULL_V3_PATH = os.path.join(DATA_MODELS, 'modelo_1_detector_lightgbm_full_v3.joblib')
IMPUTER_M1_LGBM_FULL_V3_PATH = os.path.join(DATA_MODELS, 'imputer_modelo_1_lightgbm_full_v3.joblib')
FEATURES_M1_LGBM_FULL_V3_PATH = os.path.join(DATA_MODELS, 'features_modelo_1_lightgbm_full_v3.joblib')
BASELINES_CTX_LGBM_FULL_V3_PATH = os.path.join(DATA_MODELS, 'baselines_ctx_lightgbm_full_v3.joblib')
BASELINES_CTX_FULL_V3_PATH = BASELINES_CTX_LGBM_FULL_V3_PATH
MODELO_2_LGBM_FULL_V3_PATH = os.path.join(DATA_MODELS, 'modelo_2_clasificador_lightgbm_full_v3.joblib')
LABEL_ENC_M2_LGBM_FULL_V3_PATH = os.path.join(DATA_MODELS, 'label_encoder_modelo_2_lightgbm_full_v3.joblib')
IMPUTER_FALT_FULL_V3_PATH = os.path.join(DATA_MODELS, 'imputer_datos_faltantes_full_v3.joblib')

MODELO_1_LGBM_TEMPORAL_PAPER_V3_PATH = os.path.join(DATA_MODELS, 'modelo_1_detector_lightgbm_temporal_paper_v3.joblib')
IMPUTER_M1_LGBM_TEMPORAL_PAPER_V3_PATH = os.path.join(DATA_MODELS, 'imputer_modelo_1_lightgbm_temporal_paper_v3.joblib')
FEATURES_M1_LGBM_TEMPORAL_PAPER_V3_PATH = os.path.join(DATA_MODELS, 'features_modelo_1_lightgbm_temporal_paper_v3.joblib')
THRESHOLD_M1_LGBM_TEMPORAL_PAPER_V3_PATH = os.path.join(DATA_MODELS, 'threshold_modelo_1_lightgbm_temporal_paper_v3.joblib')
BASELINES_CTX_LGBM_TEMPORAL_PAPER_V3_PATH = os.path.join(DATA_MODELS, 'baselines_ctx_lightgbm_temporal_paper_v3.joblib')
MODELO_2_LGBM_TEMPORAL_PAPER_V3_PATH = os.path.join(DATA_MODELS, 'modelo_2_clasificador_lightgbm_temporal_paper_v3.joblib')
LABEL_M2_LGBM_TEMPORAL_PAPER_V3_PATH = os.path.join(DATA_MODELS, 'label_encoder_modelo_2_lightgbm_temporal_paper_v3.joblib')
IMPUTER_FALT_TEMPORAL_PAPER_V3_PATH = os.path.join(DATA_MODELS, 'imputer_datos_faltantes_temporal_paper_v3.joblib')

CSV_08_AUDITORIA_REAL_FULL_V3 = os.path.join(DATA_RESULTS, '08_auditoria_cambios_real_full_v3.csv')
CSV_08_LIMITES_REAL_FULL_V3 = os.path.join(DATA_RESULTS, '08_valores_cerca_limites_fisicos_full_v3.csv')
OUT_EVENTOS_STUCK_FULL_V3 = os.path.join(DATA_RESULTS, '02_stuck_full_v3_eventos.csv')
OUT_AUDITORIA_STUCK_FULL_V3 = os.path.join(DATA_RESULTS, '02_stuck_full_v3_auditoria.csv')
CSV_02_RUIDO_EVENTOS_FULL_V3 = os.path.join(DATA_RESULTS, '02_ruido_full_v3_eventos.csv')
CSV_02_RUIDO_RESUMEN_FULL_V3 = os.path.join(DATA_RESULTS, '02_ruido_full_v3_resumen.csv')
CSV_02_RUIDO_AUDITORIA_FULL_V3 = os.path.join(DATA_RESULTS, '02_ruido_full_v3_auditoria.csv')
CSV_02_ANOMALIAS_SENSOR_SPLIT_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '02_anomalias_inyectadas_por_sensor_split_temporal_paper_v3.csv'
)
CSV_02_ANOMALIAS_SENSOR_TIPO_SPLIT_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '02_anomalias_inyectadas_por_sensor_tipo_split_temporal_paper_v3.csv'
)
CSV_02_RESUMEN_VISUALIZACION_SENSOR_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '02_resumen_visualizacion_sensor_temporal_paper_v3.csv'
)
CSV_04_M1_SENSIBILIDAD_UMBRAL_VALIDACION_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '04_m1_sensibilidad_umbral_validacion_temporal_paper_v3.csv'
)
CSV_04_M1_RECALL_POR_TIPO_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '04_m1_recall_por_tipo_temporal_paper_v3.csv'
)
CSV_04_M1_SENSIBILIDAD_UMBRAL_EVALUACION_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '04_m1_sensibilidad_umbral_evaluacion_temporal_paper_v3.csv'
)
CSV_04_M1_CORRELACION_SENSORES_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '04_m1_correlacion_sensores_validacion_temporal_paper_v3.csv'
)
CSV_04_M1_SENSOR_ATASCADO_POR_SENSOR_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '04_m1_sensor_atascado_por_sensor_temporal_paper_v3.csv'
)
CSV_05_M2_REPORT_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '05_m2_report_temporal_paper_v3.csv'
)
CSV_06E_TIEMPOS_CORRECCION_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '06e_tiempos_correccion_full_v3.csv'
)
CSV_06E_RUIDO_METRICAS_TEMPORAL_PAPER_V3 = os.path.join(DATA_RESULTS, '06e_ruido_metricas.csv')
CSV_06E_OOR_METRICAS_TEMPORAL_PAPER_V3 = os.path.join(DATA_RESULTS, '06e_oor_metricas.csv')
CSV_06E_STUCK_DINAMICOS_METRICAS_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '06e_stuck_dinamicos_metricas.csv'
)
CSV_06E_STUCK_ESTABLES_METRICAS_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '06e_stuck_estables_metricas.csv'
)
CSV_06E_IMPUTACION_FINAL_SELECTIVA_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '06e_imputacion_final_selectiva_full_v3.csv'
)
CSV_06E_CORRDEV_METRICAS_TEMPORAL_PAPER_V3 = os.path.join(DATA_RESULTS, '06e_corrdev_metricas.csv')
CSV_06E_REPARACION_FINAL_DATOS_FALTANTES_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '06e_reparacion_final_datos_faltantes_full_v3.csv'
)
CSV_06E_AUDITORIA_CAMBIOS_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '06e_auditoria_cambios_full_v3.csv'
)
CSV_06E_AUDITORIA_EJEMPLOS_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '06e_auditoria_ejemplos_cambios_fuera_sensor_full_v3.csv'
)
CSV_06_CORRECCION_METRICAS_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '06_correccion_metricas_temporal_paper_v3.csv'
)
CSV_07_REDUCCION_ERROR_SENSOR_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '07_reduccion_error_por_sensor_temporal_paper_v3.csv'
)
CSV_07_COMPARATIVAS_SENSOR_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '07_comparativas_por_sensor_temporal_paper_v3.csv'
)
CSV_07_ZOOM_RECONSTRUCCION_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '07_zoom_reconstruccion_temporal_paper_v3.csv'
)
JSON_07_RESUMEN_FINAL_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '07_resumen_final_temporal_paper_v3.json'
)
JSON_04_RESUMEN_MODELO_DETECCION_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_RESULTS, '04_resumen_modelo_deteccion_temporal_paper_v3.json'
)

PNG_04_M1_SENSIBILIDAD_UMBRAL_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '04_m1_sensibilidad_umbral_temporal_paper_v3.png'
)
PNG_02_ANOMALIAS_SENSOR_SPLIT_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '02_anomalias_inyectadas_por_sensor_split_temporal_paper_v3.png'
)
PNG_02_ANOMALIAS_SENSOR_TIPO_SPLIT_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '02_anomalias_inyectadas_por_sensor_tipo_split_temporal_paper_v3.png'
)
PNG_02_SERIE_ANOMALIAS_SENSOR_TEMPLATE_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '02_serie_anomalias_sensor_{sensor}_temporal_paper_v3.png'
)
PNG_04_M1_FP_FN_UMBRAL_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '04_m1_fp_fn_por_umbral_temporal_paper_v3.png'
)
PNG_04_M1_CORRELACION_SENSORES_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '04_m1_correlacion_sensores_validacion_temporal_paper_v3.png'
)
PNG_04_M1_MATRIZ_CONFUSION_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '04_m1_matriz_confusion_temporal_paper_v3.png'
)
PNG_04_M1_DESGLOSE_TIPO_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '04_m1_desglose_tipo_anomalia_temporal_paper_v3.png'
)
PNG_04_M1_PROBABILIDAD_TIPO_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '04_m1_probabilidad_por_tipo_temporal_paper_v3.png'
)
PNG_06E_RUIDO_CORRECCION_TEMPORAL_PAPER_V3 = os.path.join(DATA_PLOTS, '06e_ruido_correccion.png')
PNG_06E_OOR_CORRECCION_TEMPORAL_PAPER_V3 = os.path.join(DATA_PLOTS, '06e_oor_correccion.png')
PNG_06E_STUCK_DINAMICOS_CORRECCION_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '06e_stuck_dinamicos_correccion.png'
)
PNG_06E_STUCK_ESTABLES_CORRECCION_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '06e_stuck_estables_correccion.png'
)
PNG_06E_CORRDEV_CORRECCION_TEMPORAL_PAPER_V3 = os.path.join(DATA_PLOTS, '06e_corrdev_correccion.png')
PNG_07_COMPARATIVA_IMPACTO_TEMPLATE_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '07_comparativa_impacto_{sensor}_temporal_paper_v3.png'
)
PNG_07_COMPARATIVA_CORRECCION_TEMPLATE_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '07_comparativa_correccion_{sensor}_temporal_paper_v3.png'
)
PNG_07_COMPARATIVA_LIMPIEZA_TEMPLATE_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '07_comparativa_limpieza_{sensor}_temporal_paper_v3.png'
)
PNG_07_ZOOM_RECONSTRUCCION_TEMPLATE_TEMPORAL_PAPER_V3 = os.path.join(
    DATA_PLOTS, '07_zoom_reconstruccion_temporal_paper_v3_{tipo_slug}_{sensor}.png'
)

STUCK_DURACION_MIN_CORRECCION_FULL_V3 = dict(STUCK_DURACION_MIN_CORRECCION)
STUCK_DURACION_MIN_CORRECCION_FULL_V3.update({
    'PHEXT': 10,
    'XHINV': 10,
})

SENSORES_STUCK_V25 = ['PCO2EXT', 'PHEXT', 'PTEXT', 'PVV', 'XCO2I', 'XHINV', 'XTINV']
SENSORES_TEMP_ESPECIALES_V25 = {'PTEXT', 'XTINV'}

STUCK_DURACION_V25 = {
    'PCO2EXT': {'duracion_min': 20, 'duracion_max': 40},
    'PHEXT':   {'duracion_min': 16, 'duracion_max': 30},
    'PTEXT':   {'duracion_min': 40, 'duracion_max': 60},
    'PVV':     {'duracion_min': 20, 'duracion_max': 60},
    'XCO2I':   {'duracion_min': 20, 'duracion_max': 60},
    'XHINV':   {'duracion_min': 16, 'duracion_max': 30},
    'XTINV':   {'duracion_min': 40, 'duracion_max': 60},
}

STUCK_MIN_SIGMA_SEPARACION_V25 = {
    'PCO2EXT': 0.75,
    'PHEXT':   0.50,
    'PTEXT':   0.50,
    'PVV':     0.75,
    'XCO2I':   1.00,
    'XHINV':   0.50,
    'XTINV':   0.50,
}

TEMP_PERIODOS_PERMITIDOS_V25 = {'morning', 'afternoon'}
TEMP_PRAD_MEDIA_MIN_V25 = 5.0
TEMP_DELTA_MIN_ABS_V25 = {'PTEXT': 1.50, 'XTINV': 1.50}
TEMP_DELTA_MAX_ABS_V25 = {'PTEXT': 4.0, 'XTINV': 5.0}
TEMP_RANGO_MIN_ABS_V25 = {'PTEXT': 0.75, 'XTINV': 0.75}

XCO2I_PREV_OUTLIER_MAX_SIGMA_V25 = 1.0
XCO2I_SEV_MAX_SIGMA_V25 = 5.0
XCO2I_FRAC_GT3_MAX_V25 = 0.10
XCO2I_FRAC_DEBIL_MAX_V25 = 0.60
XCO2I_PREV_WINDOW_V25 = 10

CUOTAS_STUCK_V25 = {
    'train': {'PCO2EXT': 40, 'PHEXT': 40, 'PTEXT': 40, 'PVV': 40, 'XCO2I': 40, 'XHINV': 40, 'XTINV': 40},
    'test':  {'PCO2EXT': 10, 'PHEXT': 10, 'PTEXT': 10, 'PVV': 10, 'XCO2I': 10, 'XHINV': 10, 'XTINV': 10},
}

RUIDO_V2_DELTAS = {
    'PTEXT':   {'min': 3.0,  'max': 8.0,   'signos': [-1, 1]},
    'XTINV':   {'min': 3.0,  'max': 8.0,   'signos': [-1, 1]},
    'PHEXT':   {'min': 10.0, 'max': 25.0,  'signos': [-1, 1]},
    'XHINV':   {'min': 10.0, 'max': 25.0,  'signos': [-1, 1]},
    'PCO2EXT': {'min': 25.0, 'max': 75.0,  'signos': [-1, 1]},
    'XCO2I':   {'min': 60.0, 'max': 160.0, 'signos': [-1, 1]},
    'PVV':     {'min': 5.0,  'max': 9.0,   'signos': [1]},
    'PRAD':    {'min': 150.0, 'max': 500.0, 'signos': [1]},
    'PRGINT':  {'min': 80.0, 'max': 250.0, 'signos': [1]},
}

RUIDO_V2_CUOTAS = {
    'train': {s: 5000 for s in RUIDO_V2_DELTAS},
    'test':  {s: 1200 for s in RUIDO_V2_DELTAS},
}
RUIDO_V2_PRAD_MIN = 50.0
RUIDO_V2_PRAD_HORAS = set(range(8, 20))
RUIDO_V2_PRGINT_MIN = 100.0
RUIDO_V2_PVV_MIN = 5.0

COLUMNAS_PERCENTILES_INYECCION = [
    'PRAD', 'PRGINT', 'XTINV', 'XHINV', 'PTEXT', 'PHEXT', 'XCO2I',
    'UVENT_cen', 'UVENT_lN',
]

PARES_DESVIACION_CORRELACION = [
    ('PRAD',  'PRGINT'),
    ('XTINV', 'XHINV'),
    ('PTEXT', 'XTINV'),
    ('PHEXT', 'XHINV'),
]

TIPOS_EXCLUIR_AUDITORIA_INYECCION = {
    'NaN Real Corregible Excluido Train',
    'NaN No Corregible',
    'excluido_inyeccion',
}

CLASES_DETERMINISTICAS_MODELO2 = {
    'Datos Faltantes',
    'Valores Fuera de Rango',
    TIPO_CAIDA_SISTEMA,
    TIPO_FALLO_PARCIAL_SISTEMA,
}

M1_EXCLUDE_EXTRA_COLS = [
    'fuente_anomalia',
    'split',
    'pred_deteccion',
    'pred_deteccion_base',
    'pred_proba_anomalia',
    'pred_tipo_anomalia',
]
M1_CTX_GROUP_COLS = ['Hora', 'Mes']
M1_CTX_EPS = FE_EPS
M1_CTX_FILL_VALUE = 1.0

MOSTRAR_GRAFICAS_SENSOR_07 = True
GUARDAR_GRAFICAS_SENSOR_07 = True
CASOS_ZOOM_07 = [
    ('Contextual', 'XCO2I', 8),
    ('Valores Fuera de Rango', 'PRGINT', 8),
    ('Ruido', 'PRGINT', 8),
    ('Ruido', 'XCO2I', 8),
    ('Sensor Atascado', 'PTEXT', 15),
    ('Sensor Atascado', 'XTINV', 15),
    ('Sensor Atascado', 'XCO2I', 15),
    ('Datos Faltantes', 'XCO2I', 10),
    ('Datos Faltantes', 'PVV', 10),
    ('Desviacion de Correlacion', 'PRAD', 12),
]
M1_THRESHOLDS_VALIDACION = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
M1_RECALL_MIN_VALIDACION = 0.97

# Datos Faltantes en inferencia real y validacion sintetica:
# la prueba global a 10 min aumento cobertura, pero empeoro sensores dinamicos
# como PRGINT/PVV. Se mantiene 5 min como regla final conservadora.
DATOS_FALTANTES_MAX_GAP_MIN_FULL_V3 = 5.0
DATOS_FALTANTES_ANCHOR_MAX_GAP_MIN_FULL_V3 = 30.0
DATOS_FALTANTES_DELTA_MAX_FULL_V3 = {
    'PCO2EXT': 80.0,
    'PHEXT': 8.0,
    'PRAD': 250.0,
    'PRGINT': 150.0,
    'PTEXT': 2.5,
    'PVV': 3.0,
    'XCO2I': 120.0,
    'XHINV': 8.0,
    'XTINV': 2.5,
}

# Balance de NaNs reales usados para entrenamiento/evaluacion sintetica.
# Se mantienen todos los Datos Faltantes sinteticos; de los NaNs reales
# cortos/suaves se conserva solo una muestra determinista para no dominar
# la clase Datos Faltantes.
DATOS_FALTANTES_REAL_MAX_TRAIN_FULL_V3 = 20000
DATOS_FALTANTES_REAL_BALANCE_SEED_FULL_V3 = 20260510
DATOS_FALTANTES_REAL_MAX_POR_COMBO_FULL_V3 = 5000
