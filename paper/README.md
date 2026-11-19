# Material del artículo

Este directorio se reserva para texto y figuras derivados de runs comparativos validados. La documentación de v7 Palmerillas es una referencia histórica bajo `approaches/lightgbm/v7_reference/`.

La figura PRAD emparejada se regenera a partir de las trazas validadas y del parquet físico de test con `python -m shared.reporting.plot_paired_prad`. El intervalo predeterminado es el 24 de diciembre de 2020, 10:00:19–14:00:19: contiene las seis categorías en la verdad y en las salidas finales de ambos modelos, incluida «Sensor atascado». Genera PNG y PDF separados para LightGBM y LSTM y una figura conjunta con verdad. Las salidas locales quedan en `results/figures/`, excluidas de Git. Los atascos verdaderos de ese intervalo corresponden a PCO2EXT y XHINV; PRAD proporciona la señal visible de fondo.
