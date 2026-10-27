# Test_2_anos - Desarrollo Supervisado Palmerillas full_v3

Esta carpeta es una copia aislada del proyecto base para entrenar, validar y auditar el sistema con anomalias sinteticas en Palmerillas sin tocar el periodo de test real.

## Escenario Actual

- Preparacion e inyeccion: `2019-08-15` -> `2021-05-12 23:59:30`
- Train: `2019-08-15` -> `2021-01-31`
- Validacion sintetica: `2021-02-01` -> `2021-05-12 23:59:30`
- Test real no visto: `2021-08-15` -> `2022-05-31`, evaluado en `08_test_real_temporal_paper_v3.ipynb` sin inyeccion sintetica.
- `Caida Sistema` se trata como anomalia determinista de infraestructura: todos los sensores objetivo estan sin dato a la vez. No se reconstruye ni entra al Modelo 2.
- `Fallo Parcial Sistema` se trata como anomalia determinista de infraestructura: varias senales caen a la vez o se combinan varios NaN con ceros sospechosos. No se reconstruye ni entra al Modelo 2.
- Regla final de `Datos Faltantes`: huecos reconstruibles hasta `5 min` con anclas cercanas y delta maximo por sensor. La prueba global a `10 min` se descarta porque aumenta cobertura, pero empeora sensores dinamicos como `PRGINT` y `PVV`.

En la ejecucion actual, los datos reales disponibles llegan hasta `2021-05-12 23:59:30`. Por eso `FECHA_FIN` se fija en esa fecha real.

## Configuracion

`config.py` define:

```python
FECHA_INICIO = '2019-08-15'
FECHA_FIN    = '2021-05-12 23:59:30'
FECHA_TEST_INICIO = None
FECHA_TEST_FIN    = None
```

Por tanto, `01/02/03` solo preparan e inyectan anomalias en train + validacion. No hay filas posteriores a `2021-05-12 23:59:30` en `02_datos_inyectados_full_v3.parquet`.

Nota: algunas reglas de inyeccion conservan nombres historicos como `fecha_inicio_test_final`, `split=train`, `split=test` o columnas como `afectados_test`. En este experimento son nombres heredados del pipeline antiguo: `test` dentro de esos CSVs significa validacion sintetica (`>= 2021-02-01`), no test real no visto. El periodo real se verifica siempre por la columna `Fecha`.

## Cuadernos

1. `01_carga_datos.ipynb`
   - Carga Palmerillas desde `DATASET_PATH`.
   - Filtra el rango `2019-08-15 -> 2021-05-12 23:59:30`.
   - Excluye periodos sin cultivo y audita cortes/NaNs historicos.
   - Salida principal: `data/interim/01_datos_cargados.parquet`.

2. `02_inyeccion_anomalias_full_v3.ipynb`
   - Inyecta anomalias sinteticas solo dentro del parquet cargado por `01`.
   - Tipos: Caida Sistema, Fallo Parcial Sistema, Datos Faltantes, Sensor Atascado, Ruido, Valores Fuera de Rango, Desviacion de Correlacion y Contextual.
   - Audita visualmente las anomalias atribuidas por sensor en train y validacion.
   - Genera series temporales por sensor con bandas de Caida/Fallo Parcial Sistema y puntos de anomalias propias, equivalente a la visualizacion del `08`.
   - Salida principal: `data/interim/02_datos_inyectados_full_v3.parquet`.
   - Auditorias/eventos: `data/results/02_*.csv`.
   - Graficas principales:
     - `data/plots/02_anomalias_inyectadas_por_sensor_split_temporal_paper_v3.png`
     - `data/plots/02_anomalias_inyectadas_por_sensor_tipo_split_temporal_paper_v3.png`
     - `data/plots/02_serie_anomalias_sensor_*_temporal_paper_v3.png`
     - `data/results/02_resumen_visualizacion_sensor_temporal_paper_v3.csv`

3. `03_features_full_v3.ipynb`
   - Calcula features temporales, rolling, stuck, variabilidad y contexto.
   - Salida principal: `data/interim/03_datos_features_full_v3.parquet`.

4. `04_modelo_deteccion_temporal_paper_v3.ipynb`
   - Entrena Modelo 1 con train.
   - Selecciona umbral en validacion.
   - Evalua deteccion binaria en validacion.
   - Umbral actual validado: `0.80`.
   - Salidas principales:
     - `data/interim/04_modelo1_predicciones_temporal_paper_v3.parquet`
     - `data/results/04_resumen_modelo_deteccion_temporal_paper_v3.json`
     - `data/results/04_m1_recall_por_tipo_temporal_paper_v3.csv`
     - `data/results/04_m1_sensibilidad_umbral_validacion_temporal_paper_v3.csv`

5. `05_modelo_clasificacion_temporal_paper_v3.ipynb`
   - Entrena Modelo 2 con anomalias no deterministas de train.
   - Evalua clasificacion de tipo en validacion.
   - Salidas principales:
     - `data/interim/05_modelo2_predicciones_temporal_paper_v3.parquet`
     - `data/results/05_m2_report_temporal_paper_v3.csv`

6. `06_correccion_temporal_paper_v3.ipynb`
   - Opcional para esta fase, pero ya ejecutado.
   - Corrige la validacion sintetica usando M1/M2 y el umbral validado.
   - Salidas principales:
     - `data/interim/06_datos_corregidos_temporal_paper_v3.parquet`
     - `data/results/06_correccion_metricas_temporal_paper_v3.csv`

7. `07_evaluacion_temporal_paper_v3.ipynb`
   - Resume M1, M2 y correccion.
   - Genera graficas comparativas por sensor: original vs inyectado, original vs corregido e inyectado vs corregido.
   - Genera zooms cualitativos de reconstruccion por tipo de anomalia y sensor.
   - Salidas principales:
     - `data/results/07_resumen_final_temporal_paper_v3.json`
     - `data/results/07_reduccion_error_por_sensor_temporal_paper_v3.csv`
     - `data/results/07_comparativas_por_sensor_temporal_paper_v3.csv`
     - `data/results/07_zoom_reconstruccion_temporal_paper_v3.csv`
     - `data/plots/07_comparativa_*_temporal_paper_v3.png`
     - `data/plots/07_zoom_reconstruccion_temporal_paper_v3_*.png`

8. `08_test_real_temporal_paper_v3.ipynb`
   - Aplica los modelos entrenados de `04/05` al periodo real no visto.
   - No inyecta anomalias sinteticas.
   - Filtra filas no modelables: `usar_en_train=False` y periodos sin cultivo.
   - Audita `Caida Sistema` en el raw completo antes de filtrar filas no modelables.
   - Audita `Fallo Parcial Sistema` en el raw completo antes de filtrar filas no modelables.
   - Separa `Caida Sistema` cuando todos los sensores objetivo estan a NaN dentro de filas modelables.
   - Separa `Fallo Parcial Sistema` cuando hay varias senales objetivo perdidas o una mezcla sospechosa de NaNs y ceros.
   - Detecta reglas deterministas de datos faltantes y fuera de rango fisico.
   - Clasifica con Modelo 2 las anomalias no deterministas detectadas por Modelo 1.
   - Nota: `sensor_inferencia` es exacto para Caida Sistema/Fallo Parcial Sistema/Datos Faltantes/Fuera de Rango, y aproximado para Ruido, Sensor Atascado, Contextual y Desviacion de Correlacion.
   - Las graficas por sensor sombrean Caida Sistema y Fallo Parcial Sistema para explicar huecos sin puntos dibujables.
   - Genera zooms cualitativos de detecciones reales por tipo/sensor para figuras de paper.
   - Salidas principales:
     - `data/interim/08_test_real_predicciones_temporal_paper_v3.parquet`
     - `data/interim/08_test_real_features_temporal_paper_v3.parquet`
     - `data/results/08_test_real_resumen_temporal_paper_v3.json`
     - `data/results/08_test_real_caidas_sistema_raw_temporal_paper_v3.csv`
     - `data/results/08_test_real_fallos_parciales_sistema_raw_temporal_paper_v3.csv`
     - `data/results/08_test_real_resumen_por_tipo_temporal_paper_v3.csv`
     - `data/results/08_test_real_resumen_por_sensor_tipo_temporal_paper_v3.csv`
     - `data/results/08_test_real_anomalias_por_mes_tipo_temporal_paper_v3.csv`
     - `data/results/08_test_real_resumen_visualizacion_sensor_temporal_paper_v3.csv`
     - `data/results/08_test_real_zoom_detecciones_temporal_paper_v3.csv`
     - `data/plots/08_test_real_anomalias_por_tipo_temporal_paper_v3.png`
     - `data/plots/08_test_real_anomalias_por_mes_tipo_temporal_paper_v3.png`
     - `data/plots/08_test_real_sensor_*_temporal_paper_v3.png`
     - `data/plots/08_test_real_zoom_*_temporal_paper_v3.png`

9. `08b_correccion_real_temporal_paper_v3.ipynb`
   - Aplica correccion conservadora al test real no visto usando las predicciones guardadas por `08`.
   - Solo modifica el sensor atribuido en `sensor_inferencia`; no corrige toda la fila.
   - No reconstruye `Caida Sistema` ni `Fallo Parcial Sistema`; esos eventos se reportan como fallos de infraestructura/adquisicion.
   - Reconstruye `Datos Faltantes` solo si el hueco es corto, tiene anclas cercanas y la dinamica entre anclas es compatible. El maximo final es `5 min`.
   - Genera zooms de correccion para paper con formato original real / detectado antes de corregir / corregido / tramo anomalo.
   - Salidas principales:
     - `data/interim/08b_test_real_corregido_temporal_paper_v3.parquet`
     - `data/results/08b_correccion_real_resumen_temporal_paper_v3.csv`
     - `data/results/08b_correccion_real_eventos_temporal_paper_v3.csv`
     - `data/results/08b_correccion_real_zooms_temporal_paper_v3.csv`
     - `data/plots/08b_zoom_correccion_*_temporal_paper_v3.png`

## Auditoria de Fechas Actual

Parquets generados:

```text
01_datos_cargados.parquet                    1,534,765 filas  2019-08-15 00:00:00 -> 2021-05-12 23:59:30
02_datos_inyectados_full_v3.parquet          1,534,765 filas  2019-08-15 00:00:00 -> 2021-05-12 23:59:30
03_datos_features_full_v3.parquet            1,394,247 filas  2019-08-15 00:00:00 -> 2021-05-12 12:39:30
04_modelo1_predicciones_temporal_paper_v3      163,158 filas  2021-02-01 00:00:00 -> 2021-05-12 12:39:30
05_modelo2_predicciones_temporal_paper_v3      163,158 filas  2021-02-01 00:00:00 -> 2021-05-12 12:39:30
06_datos_corregidos_temporal_paper_v3          245,245 filas  2021-02-01 00:00:00 -> 2021-05-12 23:59:30
```

Comprobacion clave:

```text
Filas posteriores a 2021-05-12 23:59:30 en 02_datos_inyectados_full_v3.parquet: 0
```

## Anomalias Inyectadas

En train:

```text
Filas train: 1,289,520
Anomalias train: 124,880
```

Desglose de train:

```text
Ruido                         45,000
Datos Faltantes               32,384
Valores Fuera de Rango        31,500
Sensor Atascado                8,741
Contextual                     4,655
Desviacion de Correlacion      2,600
```

En validacion:

```text
Filas validacion: 245,245
Anomalias validacion: 29,326
```

Desglose de validacion:

```text
Ruido                         10,800
Valores Fuera de Rango         8,100
Datos Faltantes                7,250
Sensor Atascado                1,952
Desviacion de Correlacion        800
Contextual                       424
```

Auditoria de eventos con `split=test` heredado, interpretado aqui como validacion:

```text
Sensor Atascado train: PCO2EXT=40, PHEXT=40, PTEXT=13, PVV=40, XCO2I=40, XHINV=40, XTINV=40
Sensor Atascado val:   PCO2EXT=10, PHEXT=9, PVV=10, XCO2I=9, XHINV=10, XTINV=10
Ruido train:           5,000 eventos por sensor en los 9 sensores
Ruido val:             1,200 eventos por sensor en los 9 sensores
```

## Resultados Actuales

Modelo 1, deteccion en validacion:

```text
umbral validado = 0.80
accuracy        = 99.09%
precision       = 97.60%
recall          = 97.33%
F1              = 97.46%
```

Recall M1 por tipo:

```text
Datos Faltantes              100.00%
Valores Fuera de Rango       100.00%
Contextual                    99.76%
Desviacion de Correlacion     99.25%
Ruido                         95.47%
Sensor Atascado               85.25%
```

Modelo 2, clasificacion no determinista:

```text
accuracy = 99.30%
```

Correccion en validacion:

```text
changed_normal_total_06 = 0
reduccion MAE media     = 96.86%
reduccion RMSE media    = 94.03%
```

Reduccion de error por sensor:

```text
PTEXT    MAE -99.76%   RMSE -98.76%
XTINV    MAE -99.72%   RMSE -98.84%
XCO2I    MAE -98.82%   RMSE -98.78%
PHEXT    MAE -97.26%   RMSE -88.98%
PRGINT   MAE -97.02%   RMSE -93.74%
PRAD     MAE -96.88%   RMSE -91.80%
XHINV    MAE -95.93%   RMSE -88.71%
PCO2EXT  MAE -95.42%   RMSE -94.95%
PVV      MAE -90.94%   RMSE -91.72%
```

## Ejecucion Recomendada

Para reproducir todo desde cero:

```powershell
.\run_experimento_principal_paper.ps1 -DesdeCero
```

O manualmente:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 08b
```

Si solo se quiere validar deteccion/clasificacion sin correccion:

```text
01 -> 02 -> 03 -> 04 -> 05
```

## Comprobaciones Importantes

- `02` debe cubrir solo `2019-08-15 -> 2021-05-12 23:59:30`.
- `04` no debe usar `2021-08-15 -> 2022-05-31`.
- El test real no visto se debe tratar en otro cuaderno de inferencia, sin anomalias sinteticas.
- `08` debe usar los modelos ya guardados por `04/05` y no debe reentrenar.
- `08b` debe ejecutarse despues de `08`; toma `08_test_real_predicciones_temporal_paper_v3.parquet` como entrada.
- `08b` debe mantener `modificados_fuera_mascara_correccion = 0` en la auditoria de alcance.
- `Caida Sistema` no debe corregirse. Si aparece en `08`, se reporta como fallo de infraestructura.
- `Fallo Parcial Sistema` no debe corregirse. Si aparece en `08`, se reporta como fallo parcial de infraestructura/adquisicion.
- En `06`, `changed_normal_total_06` debe ser `0`.
- Sensor Atascado sigue siendo el punto mas dificil de M1, especialmente por `XTINV`.
