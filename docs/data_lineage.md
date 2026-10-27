# Procedencia de datos físicos de Agroconnect

Los datos físicos proceden de **Agroconnect**. El CSV `sensor_data_raw2.csv` copiado localmente desde `../Clasificaci-n-y-correcci-n/Dataset/Datos2MesesLimpios` tiene SHA-256 `621f3aa590aa3501cf2f8bdd5cd624be26f44eff772633e1254e1666db6a39ba`. El CSV y los parquets están excluidos de Git. `python run_comparison.py hydrate --lightgbm-source <ruta>` los restaura desde una copia local de la fuente y verifica los hashes.

El cuaderno LSTM `01_raw_data_ingestion.ipynb` ordena por `FECHA` y conserva la primera fila por timestamp. La ejecución importada registra **233.381 − 48 = 233.333** observaciones originales. `04_gap_imputation_and_splitting.ipynb` identifica dos huecos y reconstruye **20** instantes, produciendo **233.353** filas. `shared/data_preparation/verify.py` volvió a ejecutar esas funciones sobre el CSV importado: la secuencia completa de timestamps, las máscaras de filas insertadas y todos los valores de los 12 sensores en filas normales coinciden con las particiones congeladas.

| Partición | Filas | SHA-256 del parquet físico |
| --- | ---: | --- |
| Train | 172.873 | `64f43cd15e121c8a147565aa691ebb3d6bf2d371ffd86824f2ab626dcba5e1a2` |
| Validation | 28.800 | `044d3d23f7a06b98afed94cee741d8f206beb5a8319c70e1a2c56859689a7107` |
| Test | 31.680 | `f6ad833659d0d379dc3a1677a8a5c0f9478536dadbb5b80bc6eadf82fa60d4f3` |

`data_manifest_id` es el SHA-256 de un JSON ordenado con esos tres hashes: `4cc10ef96d923a326a86c0882a356a4d09d35a6da6553bfdbb60dd80a5f0737c`. Los timestamps físicos no se redondean. LightGBM v8 usa además un reloj nominal interno de 30 segundos por fila para sus ventanas y reglas; `cos_hour`, la traza y la comparación se alinean con `FECHA` física.

Las anomalías inyectadas se conservan en las tres particiones locales. `shared/injection/replay.py` deriva un plan local `injection_replay.parquet` de los parquets congelados, aplica sus 33.591 filas anómalas a la preparación reconstruida y comprueba igualdad completa. El plan ocupa aproximadamente 2,52 MiB y queda excluido de Git. **Replay exacto** no implica que se haya repetido el muestreo aleatorio del notebook histórico: la implementación de ese generador y su semilla base 42 se han importado para una auditoría posterior de paridad evento por evento.
