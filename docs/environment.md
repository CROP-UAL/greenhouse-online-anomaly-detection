# Entorno reproducible

Crear el entorno en la raíz del repositorio. En Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Para entrenamiento LSTM en Linux, usar un checkout propio y crear `.venv` allí:

```bash
python3.10 -m venv .venv
./.venv/bin/python -m pip install -r requirements-lstm-linux.txt
./.venv/bin/python run_comparison.py train-lstm
```

Para ejecutar el recorrido controlado en notebooks en un PC Linux, instalar las dependencias opcionales en el mismo `.venv` y registrar el kernel:

```bash
./.venv/bin/python -m pip install -r requirements-notebooks-linux.txt
./.venv/bin/python -m ipykernel install --user --name greenhouse-comparison --display-name "Python (.venv) greenhouse"
./.venv/bin/python -m jupyter lab
```

Abrir Jupyter desde la raíz y seguir `notebooks/01_train_and_export.ipynb`, `02_compare_quality.ipynb` y `03_jetson_benchmark.ipynb` en ese orden. Los dos primeros se ejecutan en el PC Linux; el tercero, en la placa. Los notebooks se versionan sin ejecuciones ni salidas. El kernel registrado apunta al `.venv` local, que debe recrearse en cada máquina.

El desarrollo LSTM original documenta Linux, Python 3.10 y TensorFlow 2.21.0. NVIDIA Jetson Orin Nano también ejecutará Linux, pero antes del benchmark deben registrarse su distribución, JetPack, Python y TensorFlow reales: compartir familia de sistema operativo no garantiza una compilación binaria idéntica. El lock LightGBM para Orin queda archivado bajo `approaches/lightgbm/v8_online_2meses/requirements-orin-py310.lock`.

Las dependencias se instalan solo en `.venv` del destino. Los artefactos del entorno no se versionan. `requirements.txt` fija el entorno común y `requirements-lstm-linux.txt` añade TensorFlow 2.21.0. En este Windows, TensorFlow 2.21 no cargó su biblioteca LiteRT y TensorFlow 2.20 tampoco cargó su runtime nativo; por eso no se ejecuta el entrenamiento LSTM aquí. Cada ejecución debe guardar `pip freeze`, Python, sistema operativo y hashes de entrada junto con sus modelos.
