# Traslado local y benchmark en Jetson Orin Nano

Hay dos recorridos. Si la Orin Nano es el único equipo Linux, los tres notebooks se ejecutan en ella y no hace falta un bundle. Si se entrena en otro PC Linux, el bundle traslada los modelos y las referencias para el benchmark. Los datos y modelos se transfieren por un canal privado; no se suben al repositorio.

## Los tres notebooks directamente en la Orin Nano

Obtener el código con un clon limpio de `main`. No copiar `.git` desde este checkout de trabajo, que conserva referencias fuente locales. En el pendrive, mantener estas rutas relativas al repositorio:

| Archivo local ignorado por Git | Necesario para | Tamaño aproximado |
| --- | --- | ---: |
| `data_contract/snapshots/train_injected_physical.parquet` | Entrenar y preparar LSTM/LightGBM | 8,6 MB |
| `data_contract/snapshots/validation_injected_physical.parquet` | Validación y contexto del test | 1,5 MB |
| `data_contract/snapshots/test_injected_physical.parquet` | Trazas, comparación y benchmark | 1,7 MB |
| Los ocho archivos de `approaches/lightgbm/v8_online_2meses/reference_run/` | Calibración, reglas y referencia LightGBM | 2,8 MB |

Estos archivos, unos 14,5 MB en total, bastan para los tres notebooks. Si también se van a ejecutar `verify-data` y todas las pruebas de reconstrucción, añadir `data_contract/snapshots/sensor_data_raw2.csv` (31,7 MB) y `data_contract/manifests/injection_replay.parquet` más `.json` (2,6 MB). No hace falta llevar modelos LSTM, trazas, informes ni un bundle: se generan en la placa. Tampoco se copia el `.venv` de Windows.

Usar el entorno TensorFlow que ya funcione en la Orin Nano. Comprobar que el mismo `.venv` importa `tensorflow`, `lightgbm`, `pandas`, `pyarrow`, `sklearn`, `IPython` y `jupyterlab`, y que `python -m pip check` no informa conflictos. No instalar a ciegas `requirements-notebooks-linux.txt` sobre ese entorno: fija TensorFlow para el PC Linux x86-64. Registrar `python -m pip freeze`, versión de JetPack y memoria disponible para documentar el run.

Desde la raíz, ejecutar `notebooks/01_train_and_export.ipynb` y `02_compare_quality.ipynb`. El segundo crea `results/metrics/paired_test.json` solo cuando ambas trazas tienen los mismos 31.680 objetivos, etiquetas y manifiesto. Luego abrir `03_jetson_benchmark.ipynb` con `TRANSFER = None` (valor inicial). El tercer notebook selecciona los runs indicados por `model_version` en el informe emparejado, aunque existan otros runs locales; cualquier ruta manual debe coincidir con ese informe. Carga las trazas antes de repetir inferencia para comprobar paridad y después cronometra. No publicar las salidas de los notebooks.

## Si el entrenamiento se hizo en otro PC Linux

La evaluación de calidad se termina allí con `notebooks/01_train_and_export.ipynb` y `02_compare_quality.ipynb`. Un bundle solo se crea si existen ambos runs y el informe emparejado de los **31.680** objetivos de test.

## En el PC Linux

Desde la raíz del repositorio, con `.venv`:

```bash
./.venv/bin/python -m shared.reporting.edge_bundle \
  --lightgbm-run experiments/runs/lightgbm_night_v1/REEMPLAZAR_RUN \
  --lstm-run experiments/runs/lstm_causal_v2/REEMPLAZAR_RUN \
  --destination /ruta/privada/edge-transfer
```

La primera invocación solo muestra la lista, los tamaños, los hashes y el total de bytes. Revisar esa lista antes de copiar. Repetir el mismo comando con `--build` para crear el directorio nuevo. `--build` falla si la ruta ya existe y no sobrescribe archivos. Verificarlo antes del traslado:

```bash
./.venv/bin/python -m shared.reporting.edge_bundle \
  --destination /ruta/privada/edge-transfer --verify
```

El bundle contiene tres parquets físicos, modelos/manifiestos LightGBM y LSTM, archivos necesarios del run LightGBM de referencia, dos trazas de test e informe emparejado. No incluye el CSV original ni `.venv`. El manifiesto del bundle registra el commit del código y el hash y tamaño de cada archivo. No versionar ni enviar este directorio a GitHub.

## En la placa

Confirmar si el equipo es **Jetson Orin Nano**. Registrar JetPack, sistema operativo, Python, modo de potencia y versiones de LightGBM y TensorFlow. Crear otro `.venv` adaptado a ese JetPack: el pin TensorFlow 2.21 de `requirements-lstm-linux.txt` corresponde al PC Linux y **no se presupone compatible con Jetson**. La versión usada debe poder cargar los `.keras` y reproducir sus predicciones; si no puede, registrar la incompatibilidad y preparar otra variante de despliegue, sin atribuirle las métricas del run original.

Hacer checkout del commit indicado en `bundle_manifest.json`. Con el bundle copiado localmente a la placa, verificar e instalar sus archivos ignorados:

```bash
./.venv/bin/python -m shared.reporting.edge_bundle \
  --destination /ruta/privada/edge-transfer --verify
./.venv/bin/python -m shared.reporting.edge_bundle \
  --destination /ruta/privada/edge-transfer --install
```

La instalación exige el mismo commit, comprueba hashes y no reemplaza un archivo diferente. Abrir `notebooks/03_jetson_benchmark.ipynb`, configurar `TRANSFER` y ejecutar sus celdas. Primero comprueba que las predicciones coinciden con las del PC; después ejecuta los benchmarks. Los informes locales quedan en `results/reports/`.

Los cronómetros actuales tienen alcances diferentes: LightGBM incluye extracción causal y reglas por objetivo; LSTM mide los modelos sobre ventanas ya preparadas. Registrar ambos valores con `prediction_unit` y `timing_scope`, sin llamarlos una latencia total equivalente. Tampoco convertir ms/secuencia LSTM a ms/observación por división automática.
