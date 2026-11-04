# Diagnóstico de `Sensor Atascado` en LSTM

## Qué cifras se pueden comparar

El README importado de `CIESOL_PROJECT` registra para Model 2 v2, evaluado
aisladamente en test, **133/523 aciertos** de `Sensor Atascado` (recall
0,2543; F1 0,3883). La ejecución guardada el 4 de noviembre de 2026 en
`notebooks/02_compare_quality.ipynb`, antes de limpiar sus salidas, registró
para el nuevo `lstm-causal-v1-20261104T114520Z`:

| Evaluación | Aciertos `stuck sensor` | Total | Recall |
| --- | ---: | ---: | ---: |
| Model 2 aislado | 27 | 523 | 0,0516 |
| Model 2 tras Model 1 | 13 | 56 | 0,2321 |
| Cascada completa | 13 | 523 | 0,0249 |

Model 1 clasificó **467/523** casos `stuck sensor` como normales. Por eso
solo 56 llegaron a Model 2. La caída del Model 2 aislado es distinta de la
caída de la cascada: la primera corresponde a clasificación de tipos y la
segunda añade los fallos del detector binario. En el experimento original,
Model 1 tenía 31.661 ventanas de test y 515 anomalías omitidas en total;
el nuevo protocolo puntúa los 31.680 timestamps físicos. Esas cifras
globales tampoco son una comparación fila a fila de `stuck sensor`.

## Causas comprobadas y límites

Contraste directo con el checkout local `CIESOL_PROJECT/notebooks`:
`13a_model2_multiclass_lstm_training_v1.ipynb` registró en validation
42/757 aciertos de `Sensor Atascado` (recall 0,0555). El v2 registró
320/757 (0,4227) en validation y 133/523 (0,2543) en test. Por tanto,
la mejora del trabajo fuente corresponde al v2; comparar con v1 daría
otra conclusión. El notebook fuente `11b_model2_diagnostic_feature_selection`
lista 146 entradas para v2. El adaptador actual también produce 146, pero
reemplaza exactamente las 18 variables `*_delta_next` y
`*_abs_delta_next` de nueve sensores por `*_delta_prev_lag1` y
`*_abs_delta_prev_lag1`. El orden de las variables compartidas tampoco es
idéntico; los modelos se entrenan por separado, así que esto no implica
por sí solo un fallo de alineación.

`lstm-causal-v1` es un entrenamiento nuevo. Su `causal_data.py` sustituye
las diferencias con el siguiente punto (`shift(-1)`), ventanas centradas y
la interpolación con observaciones futuras del flujo fuente por diferencias
y ventanas pasadas, forward fill y mediana de train. Conserva la arquitectura
LSTM, pero cambia las entradas; por tanto, no reproduce el checkpoint ni
sus métricas. La documentación fuente ya identificaba `Sensor Atascado`
como la clase más débil y describía diferencias de sensores y horarios
entre train y validation. Es plausible que la pérdida de contexto futuro
afecte especialmente a una señal plana, pero estas cifras por sí solas no
demuestran cuánto explica cada transformación.

La ejecución de Model 2 aislado usa los mismos 523 objetivos de test que
el informe fuente. La comprobación emparejada del notebook 02 exige los
mismos 31.680 timestamps, etiquetas y manifiesto para LightGBM y LSTM;
no compara el checkpoint de CIESOL con el nuevo LSTM. No se debe ajustar
el modelo mirando repetidamente este test. Para atribuir la diferencia,
comparar variantes de preparación en validation con los mismos objetivos,
registrando recall por sensor afectado, duración y hora, y evaluar en test
solo la variante elegida.

## Cambio preparado para una nueva validación

El código anterior ya incluía en Model 2 las variables de forma del evento
que CIESOL v2 añadió para separar ruido y atascos, con diferencias y ventanas
causales. Por eso copiar de nuevo esas variables no resolvería el principal
fallo de cascada: Model 1 omite 467/523 atascos. `lstm-causal-v2` añade
duraciones de lecturas consecutivas idénticas por sensor a ambos modelos,
solo para los ocho sensores elegibles en la inyección fuente. Las entradas
pasan de 19 a 27 y de 146 a 154. La variante queda separada de los runs v1.
No existe aún una métrica de v2: la efectividad debe medirse primero en
validation y solo después en test. El resultado de test observado arriba
corresponde exclusivamente a v1.

## Estado de ejecución de los notebooks

Las salidas que estaban versionadas muestran que 01 y 02 terminaron en una
Jetson Linux sin una excepción visible; el 02 produjo las tablas anteriores.
El notebook 03 no tenía contadores ni salidas de ejecución. La presencia de
salidas en 01 y 02 incumplía `tests/test_notebook_workflow.py` y se limpió
para mantener los notebooks versionados sin resultados. En este checkout
Windows no hay un run LSTM local, `results/metrics/paired_test.json` ni una
Jetson disponible: no se puede afirmar que el notebook 03 complete su
inferencia y benchmark reales aquí. Antes de otro entrenamiento, ejecutar
las comprobaciones de sintaxis/bootstrap y verificar datos, dependencias y
artefactos en la máquina Linux; después ejecutar 03 con el run emparejado.
