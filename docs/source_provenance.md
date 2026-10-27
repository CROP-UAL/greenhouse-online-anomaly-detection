# Procedencia de las fuentes

La lista completa de **65 archivos copiados localmente** durante la integración, con ruta fuente, ruta destino, hash original, hash importado, tamaño y transformación, está en `data_contract/manifests/imported_files.json`. La fecha de importación es 2026-10-27. Los CSV, parquets y ocho archivos del run de modelos LightGBM quedan ignorados por Git; el inventario conserva solo su procedencia y hashes.

| Repositorio | URL | Commit importado | Autores y licencia |
| --- | --- | --- | --- |
| LightGBM, v7 y v8 | `https://github.com/vfd003-ual/Clasificaci-n-y-correcci-n.git` | `83399ba380124f9bc0754990f2afeb195df94d29` | Git conserva autores por commit; no se encontró licencia raíz. |
| LSTM | `https://github.com/helqadiri03/CIESOL_PROJECT.git` | `f3a5f500c7a4b45e9485653d79aa6b67857657f4` | Git conserva autores por commit; no se encontró licencia raíz. |

La fila LSTM identifica el repositorio del código. La fuente de los datos físicos compartidos es Agroconnect, como registra [data_lineage.md](data_lineage.md).

Los DAG completos se conectaron inicialmente en una integración local preservada en `refs/archive/full-source-history`. Una revisión posterior detectó CSV y un modelo `.pkl` en el historial antiguo LightGBM; por el requisito de no subir datos ni artefactos, esa referencia no debe publicarse. `main` conserva URL, commit, autores, rutas y hashes en este documento y en el inventario, sin incorporar esos commits fuente a sus ancestros. Una publicación de `main` no envía esa referencia; una publicación con `--mirror` sí podría enviarla y debe evitarse. Los notebooks importados perdieron sus salidas y contadores de ejecución. En seis archivos LSTM del destino también se corrigió la atribución textual de los datos a Agroconnect, sin cambiar rutas ni lógica. La copia de `approaches/lstm/src/preprocessing.py` usa una importación relativa de `data_loader` para evitar colisiones entre los dos paquetes `src`. Los hashes de origen y destino y todas estas transformaciones constan en el inventario.
