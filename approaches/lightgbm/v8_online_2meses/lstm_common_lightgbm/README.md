# LightGBM sobre los parquets físicos de LSTM

Esta variante vive por completo en `v8_online_2meses/lstm_common_lightgbm`. Lee los tres parquets de `Dataset/MismoTest` y no escribe en el proyecto CIESOL/LSTM. El run definitivo es `data/runs/20261021T120316012126Z` (contrato `lstm-common-lightgbm-v2`).

## Cuadernos

1. `01_entrenamiento_binario.ipynb`: audita entradas y entrena/calibra M1. Por defecto carga el run definitivo; active `RUN_TRAINING=True` para generar otro run.
2. `02_entrenamiento_multiclase.ipynb`: entrena M2 solo con anomalías de train y usa validation para parada temprana. Active `RUN_TRAINING=True` solo con un run binario todavía incompleto.
3. `03_evaluacion_y_latencia.ipynb`: métricas, traza, figura PRAD, comparación opcional con el CSV LSTM y benchmark. Por defecto muestra los resultados existentes. `RUN_EVALUATION=True` crea una nueva evaluación; `RUN_BENCHMARK=True` mide latencia en la máquina donde se ejecuta.

Use el entorno `v8_online_2meses/.venv` y abra los cuadernos desde el repositorio o un subdirectorio. Cada artefacto nuevo se guarda bajo `data/runs/<id>`.

## Protocolo y resultados definitivos

- Train: 172.873 filas físicas, de las cuales las primeras 360 calientan el extractor causal. Validation: 28.800 filas. Test: 31.680 filas y 4.395 anomalías.
- M1 y M2 emplean las 66 características actuales de LightGBM. Train/validation se usan para ajuste y calibración. El test se utiliza solo para la evaluación final.
- Los umbrales binarios congelados son 0,4 durante 4 observaciones o 0,75 durante 1, más las reglas deterministas existentes.
- Test completo: F1 binario **0,894104**, TP 3.943, FP 482, FN 452, TN 26.803; macro-F1 de Normal y seis anomalías **0,816247**. Se enviaron 4.425 filas a M2.
- M2 condicionado a las anomalías de validation: macro-F1 **0,879931**. Este valor no es la métrica final de siete salidas.
- La figura del 21 de diciembre conserva los valores ausentes de PRAD. Hay 479 filas en la ventana y un marcador dirigido a M2 sin PRAD visible.

El reloj causal interno avanza 30 segundos por fila para mantener continuidad de las ventanas y reglas. `cos_hour` usa la hora física de `FECHA`. Como el train físico contiene 73 filas más que una cuadrícula uniforme de 60 días, el reloj interno llega a separarse unos 36 minutos de `FECHA`; los valores exactos están en `clock_audit.json`. Los resultados y gráficos siempre se alinean por `FECHA` original.

## Comparación y latencia

`workflow.ipynb` de LSTM exporta las predicciones de secuencias de 20 pasos desde la fila 20 del test. El tercer cuaderno exige su CSV y coincidencia exacta de 31.661 timestamps y etiquetas antes de producir métricas emparejadas. El CSV no estaba disponible localmente durante este run.

El benchmark online mide extracción causal, M1, reglas, encaminamiento y M2 condicional. Excluye lectura del parquet y carga de modelos; hace una pasada completa de calentamiento y registra media, p50, p95, p99, throughput, hardware y paridad con la traza. Se verificó paridad en las 31.680 filas en este PC; la cifra publicable debe obtenerse ejecutando el mismo cuaderno en la Jetson Nano.

