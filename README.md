# Greenhouse online anomaly comparison

Comparación controlada entre LightGBM y LSTM sobre los mismos 233.353 instantes físicos de Agroconnect. LightGBM conserva características causales y reglas; LSTM conserva ventanas de 20 observaciones y tratamiento de ausencias. `v7_palmerillas_vent_null` se conserva como fundamento documentado, con datos y resultados Palmerillas separados.

## Estado

- El CSV original, las tres particiones congeladas, los modelos de referencia y el plan de replay son **archivos locales ignorados por Git**. Sus hashes y rutas esperadas están en `data_contract/`. La reconstrucción desde el CSV reproduce 48 duplicados eliminados, 20 marcas insertadas y los valores normales de los 12 sensores. Un plan de replay local reproduce exactamente las filas inyectadas; el generador aleatorio histórico sigue documentado en el notebook importado.
- La traza de referencia LightGBM v8 y la evaluación aislada de Model 2 ya se generan sobre todo el test. `lightgbm-night-context-v1` se entrenó y exportó localmente como otro experimento: añade `is_night` explícito (20:00–07:00) y el gradiente firmado de CO₂ interior menos exterior a ambos árboles; la referencia de 66 características permanece intacta. Su run y sus trazas están excluidos de Git.
- LSTM se ha preparado como `lstm-causal-v1`: mantiene las arquitecturas y parámetros de entrenamiento de los notebooks y 19/146 entradas por instante, pero sustituye características e imputaciones que consultaban el futuro por alternativas causales. Su entrenamiento y las métricas comparativas emparejadas requieren Linux con TensorFlow funcional. Este Windows no dispone de WSL ni Docker y la biblioteca nativa TensorFlow instalada en `.venv` no carga.
- No hay cifras comparativas finales ni benchmark Jetson Orin Nano hasta ejecutar ambos runs y verificar paridad de trazas en ese dispositivo.

## Reproducción

Crear `.venv` siguiendo [docs/environment.md](docs/environment.md). Comandos desde la raíz:

```text
python run_comparison.py hydrate --lightgbm-source ../Clasificaci-n-y-correcci-n
python run_comparison.py verify-data
python run_comparison.py export-lightgbm
python run_comparison.py train-lightgbm-night
python run_comparison.py export-lightgbm-night --run experiments/runs/lightgbm_night_v1/<run>
python run_comparison.py train-lstm
python run_comparison.py export-lstm --run experiments/runs/lstm_causal_v1/<run>
python run_comparison.py evaluate
python run_comparison.py benchmark-lightgbm --rows 2000 --repeats 5
python run_comparison.py benchmark-lightgbm-night --run experiments/runs/lightgbm_night_v1/<run> --rows 2000 --repeats 5
python run_comparison.py benchmark-lstm --run experiments/runs/lstm_causal_v1/<run> --rows 2000 --repeats 5
```

La evaluación emparejada usa por defecto LightGBM con noche y contexto exterior/interior; `evaluate --lightgbm-reference` permite examinar aparte la referencia v8. Falla si faltan trazas o difieren timestamps, verdad o manifiesto físico. Los resultados generados quedan bajo `results/` y los modelos nuevos bajo `experiments/runs/`, excluidos de Git. `hydrate` exige que los archivos ignorados estén disponibles en el checkout fuente local y verifica cada hash; un clon de Git sin esos archivos debe obtenerlos por separado. Las fuentes, versiones y límites se describen en [la auditoría](docs/integration_audit.md), [el protocolo](docs/comparison_protocol.md), [la procedencia de datos](docs/data_lineage.md) y [la procedencia de código](docs/source_provenance.md).
