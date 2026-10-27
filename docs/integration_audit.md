# Auditoría previa a la integración

Fecha: 2026-10-27. Esta auditoría se escribió antes de importar datos o código al repositorio destino. Los repositorios fuente se inspeccionaron en modo lectura.

## Inventario y versiones

| Fuente | Estructura y versión relevante | Commit inspeccionado | Papel |
| --- | --- | --- | --- |
| `../Clasificaci-n-y-correcci-n` | `v7_palmerillas_vent_null/{Test_2_anos,Antigua,_archive,paper,papers}`; además `data/`, `Dataset/` y versiones v2–v8 | `83399ba380124f9bc0754990f2afeb195df94d29`, rama `feat/no-ref/same-setting` | v7 es base conceptual y documental; usa Palmerillas, no el test Agroconnect compartido. Su introducción en el árbol figura en `dce5772` (2026-09-16). |
| Misma fuente, v8 | `v8_online_2meses/{src,seis_anomalias,lstm_common_lightgbm,data,tests,scripts}` y notebooks de entrenamiento, inferencia y edge | `48bf174` introdujo la variante sobre los parquets físicos LSTM; `83399ba` es HEAD | `lstm_common_lightgbm` es la referencia online de seis tipos sobre el contexto físico común. `seis_anomalias` es antecedente distinto con cuadrícula a 30 s. |
| `../CIESOL_PROJECT` | `src/` para preparación, inyección, faltantes, escalado y secuencias; `notebooks/01`–`16`; `figures/`, `reports/` | `f3a5f500c7a4b45e9485653d79aa6b67857657f4`, rama `main` | Repositorio fuente del código LSTM binario y multiclase v2; los datos del experimento compartido proceden de Agroconnect. El último commit renombró archivos; los pesos y parquets intermedios descritos en README no están en el árbol local. |

Orígenes Git: `https://github.com/vfd003-ual/Clasificaci-n-y-correcci-n.git` y `https://github.com/helqadiri03/CIESOL_PROJECT.git`. El primero tiene 41 commits alcanzables desde HEAD y el segundo 6. No se localizaron archivos de licencia en las raíces inspeccionadas. Autores del último commit: `vfd003-ual <vfd003@inlumine.ual.es>` y `CIESOL Dev <dev@ciesol.com>`; la autoría completa debe conservarse en Git.

La procedencia de los **datos físicos compartidos** es Agroconnect. `CIESOL_PROJECT` se menciona únicamente como identificador del repositorio de código LSTM y de sus autores históricos, nunca como fuente de datos.

## Datos y preparación

- CSV Agroconnect: `Dataset/Datos2MesesLimpios/sensor_data_raw2.csv`, 30,24 MiB y 233.381 filas según el notebook de ingestión. Los tres parquets físicos en `Dataset/MismoTest` ocupan aproximadamente 8,17, 1,46 y 1,57 MiB. Sus hashes SHA-256 son, respectivamente, `64f43cd15e121c8a147565aa691ebb3d6bf2d371ffd86824f2ab626dcba5e1a2`, `044d3d23f7a06b98afed94cee741d8f206beb5a8319c70e1a2c56859689a7107` y `f6ad833659d0d379dc3a1677a8a5c0f9478536dadbb5b80bc6eadf82fa60d4f3`. Coinciden con el manifiesto LightGBM `20261021T120316012126Z`.
- Train tiene 172.873 filas (2020-10-10 a 2020-12-08), validation 28.800 (2020-12-09 a 2020-12-18) y test 31.680 (2020-12-19 a 2020-12-29). Los tres hashes, intervalos y distribuciones por etiqueta constan en `binary_manifest.json` de esa ejecución.
- La diferencia neta de **−28** respecto al CSV tiene una explicación registrada por los notebooks LSTM: `01_raw_data_ingestion.ipynb` muestra 48 timestamps duplicados eliminados (`233381 → 233333`); `04_gap_imputation_and_splitting.ipynb` muestra 20 timestamps reconstruidos en dos huecos (`233333 → 233353`). Queda pendiente repetir la transformación sobre el CSV disponible y verificar igualdad de filas, timestamps y valores frente a los parquets. El README LSTM cita además 233.616 filas y 283 timestamps insertados para otra preparación; no se atribuye esa cifra al run compartido.
- LSTM conserva el jitter real de 30–31 s, inserta filas solo en huecos detectados, marca ausencias y limita la interpolación interior. LightGBM compartido lee esos mismos parquets y conserva `FECHA` física como objetivo, pero genera un reloj interno de 30 s por fila para características y reglas; `cos_hour` usa `FECHA`. Esta diferencia interna debe figurar en toda interpretación temporal.
- Ambos flujos compartidos tienen nueve sensores objetivo: `PCO2EXT`, `PHEXT`, `PRAD`, `PRGINT`, `PTEXT`, `PVV`, `XCO2I`, `XHINV`, `XTINV`. La fuente LSTM también enumera tres sensores de contexto. Sus parquets ya contienen las seis etiquetas españolas; el mapeo a las seis salidas inglesas será explícito y auditable.
- v7 usa el CSV Palmerillas `palmerillas_all_data_30s_null`, train sintético 2019-08-15–2021-01-31, validación sintética desde 2021-02-01 y test real posterior sin inyección. Define ocho anomalías, incluyendo `Caida Sistema` y `Fallo Parcial Sistema` como fallos de infraestructura. Su configuración y notebooks fundamentan reglas, preparación e inyección; ni sus datos ni sus métricas son comparables directamente con Agroconnect.

## Modelos, salidas, métricas y edge

- LightGBM compartido usa 66 características causales, 360 filas de contexto, Model 1 binario, reglas deterministas y persistencia de umbral `0,4 × 4` o `0,75 × 1`; Model 2 clasifica seis anomalías solo tras una decisión positiva. El run `20261021T120316012126Z` contiene ambos modelos de aproximadamente 1,32 MiB, calibración y manifiestos. Su README informa F1 binario 0,894104 y macro-F1 final 0,816247; son resultados históricos, no resultados comparativos finales.
- LSTM construye ventanas de 20 filas con objetivo en la última observación. Model 1 usa dos capas LSTM y decisión binaria; Model 2 v2 usa seis clases. Escala con parámetros ajustados en train e incorpora indicadores de faltantes. El README describe un checkpoint binario y otro multiclase, pero ninguno está disponible localmente. Informa macro-F1 aislado de Model 2 en test 0,8382 y latencia XLA 0,892 ms/secuencia en CPU; no equivalen a la salida final ni a latencia Orin Nano.
- En v7 hay reglas deterministas de infraestructura y una taxonomía de ocho tipos; en v8 la variante de seis tipos mapea esos fallos a datos faltantes para auditoría. El flujo LSTM consultado no documenta filtros binarios equivalentes a los de LightGBM. Se verificará al construir el adaptador y se conservará la diferencia.
- El benchmark LightGBM compartido mide extracción, Model 1, reglas y Model 2 condicional por fila. El benchmark LSTM histórico citado mide una secuencia; otros experimentos incluyen un cuaderno de cascada. No se convertirán unidades ni se presentarán cifras históricas de PC o Jetson Nano como mediciones de Orin Nano.

## Importación propuesta y estrategia Git

Primero crear un commit propio del destino. Después incorporar los dos DAG fuente mediante merges de historiales no relacionados con estrategia que conserve el árbol curado del destino; esto hace visibles ambos historiales en `git log --graph` sin poblar el árbol de trabajo con todas sus versiones. v7 y v8 llegan con el mismo DAG. Comprobar ancestros y tamaño antes y después. Los archivos seleccionados conservarán además ruta y hash de origen en `docs/source_provenance.md`.

Importar de v7 `Test_2_anos/config.py`, `README_TEST_2_ANOS.md` y extractos ejecutables o notebooks pertinentes de `Antigua` para inyección, detección y clasificación, una vez aisladas sus dependencias. De v8 importar `src/`, `seis_anomalias/{data,pipeline,settings}`, `lstm_common_lightgbm/{pipeline,comparison}`, requisitos y manifiestos/modelos del run compartido. De LSTM importar `src/` y los notebooks de ingestión, preparación, inyección, faltantes, escalado, secuencias y entrenamiento necesarios para reproducir ambos modelos. Mantener enfoques separados.

Excluir `env`/`.venv`, cachés, `scratch`, notebooks y papers redundantes, datos Palmerillas masivos, resultados de corrección y figuras generadas. Los notebooks v7 de evaluación e inferencia pesan aproximadamente 10,8 y 5,5 MiB; se citarán o extraerán solo si resultan imprescindibles. No borrar archivos fuente.

## Puertas técnicas

1. Verificar el CSV y la reconstrucción `−48 +20`, los hashes y la paridad de los tres parquets antes de aceptar datos regenerados.
2. Registrar semilla, configuración y algoritmo exactos de la inyección LSTM; los parquets congelados son el contrato inicial, no una demostración de regeneración.
3. Reentrenar LSTM como run nuevo con configuración de referencia, pues faltan pesos locales; instalar dependencias en el destino si el entorno disponible no basta.
4. Definir `episode_id` solo si los metadatos contienen un identificador común verificable. Para Model 2 aislado se requiere una traza diagnóstica adicional sobre todas las anomalías verdaderas.
5. No publicar comparación final ni benchmark Orin hasta disponer de ambas trazas alineadas y mediciones en el dispositivo indicado.

## Adenda de privacidad de repositorio (2026-10-27)

La inspección posterior del historial completo de LightGBM encontró CSV de análisis y un modelo `.pkl` ya versionados en commits antiguos. Por ello, los merges exactos de ambos DAG, aunque se ensayaron localmente y dejan visible la historia, **no son seguros para publicar** bajo el requisito posterior de no subir datos ni artefactos. La rama publicable se basa en el commit inicial de esta auditoría y versiona solo código, notebooks sin salidas, requisitos, documentación y metadatos de procedencia. Los commit originales y URL siguen registrados; recuperar ambos DAG completos en el historial publicable requeriría filtrar y reescribir los commits fuente para retirar todos los blobs de datos y artefactos, cambiando sus hashes. No se ha publicado ni subido ningún archivo.
