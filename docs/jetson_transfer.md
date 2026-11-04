# Traslado local y benchmark en Jetson Orin Nano

La evaluación de calidad se termina en un PC Linux con `notebooks/01_train_and_export.ipynb` y `02_compare_quality.ipynb`. Un bundle solo se crea si existen ambos runs y el informe emparejado de los **31.680** objetivos de test. Los datos y modelos se transfieren por un canal privado; no se suben al repositorio.

## En el PC Linux

Desde la raíz del repositorio, con `.venv`:

```bash
./.venv/bin/python -m shared.reporting.edge_bundle \
  --lightgbm-run experiments/runs/lightgbm_night_v1/REEMPLAZAR_RUN \
  --lstm-run experiments/runs/lstm_causal_v1/REEMPLAZAR_RUN \
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
