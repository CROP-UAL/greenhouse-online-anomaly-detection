# CIESOL Sensor Anomaly Detection

A time-series anomaly detection project for Agroconnect sensor data. The
system uses a **two-stage LSTM approach**:

1.  **Model 1 --- Binary anomaly detection:** determines whether a
    sensor-data sequence is normal or anomalous.
2.  **Model 2 --- Multiclass anomaly classification:** runs after Model
    1 flags an anomaly and predicts its anomaly type.

The project covers raw-data loading and quality checks, timestamp-aware
preprocessing, chronological train/validation/test splitting, controlled
anomaly injection, feature engineering, missing-value handling, scaling,
sequence generation, LSTM training, evaluation, and initial
inference-latency experiments.

> **Current status:** Model 1 has a strong binary-classification
> baseline. Model 2 v2 performs well on five anomaly classes but has a
> significant generalization problem with **Sensor Atascado** (stuck
> sensor). The current priority is to improve the diversity and
> representativeness of stuck-sensor training examples without tuning
> against the test set.

------------------------------------------------------------------------

## Table of contents

-   [Project goals](#project-goals)
-   [System overview](#system-overview)
-   [Repository and data
    organization](#repository-and-data-organization)
-   [Dataset](#dataset)
-   [Data preparation and quality
    decisions](#data-preparation-and-quality-decisions)
-   [Anomaly injection and labels](#anomaly-injection-and-labels)
-   [Feature engineering](#feature-engineering)
-   [Missing-value handling](#missing-value-handling)
-   [Scaling and sequence generation](#scaling-and-sequence-generation)
-   [Model 1: binary LSTM detector](#model-1-binary-lstm-detector)
-   [Model 2: multiclass LSTM
    classifier](#model-2-multiclass-lstm-classifier)
-   [Model 2 v2 evaluation](#model-2-v2-evaluation)
-   [What the results mean](#what-the-results-mean)
-   [Inference and latency
    experiments](#inference-and-latency-experiments)
-   [Reproducibility notes](#reproducibility-notes)
-   [Known limitations](#known-limitations)
-   [Recommended next steps](#recommended-next-steps)
-   [Glossary](#glossary)

------------------------------------------------------------------------

## Project goals

The goal is to identify anomalous behavior in environmental and
greenhouse-related sensor time series, then classify the detected
anomaly into an interpretable category.

The main objectives are:

-   Preserve the original temporal information instead of blindly
    resampling all readings to a fixed interval.
-   Identify timestamp gaps and distinguish normal timing jitter from
    missing-data episodes.
-   Build reusable processing logic in `src/`, with notebooks used to
    execute, inspect, and document each stage.
-   Inject known anomalies into separate chronological data splits so
    that supervised models can be trained and evaluated.
-   Avoid leaking target labels, traceability metadata, or future
    information into model features.
-   Use sequences of 20 time steps to capture temporal behavior.
-   Evaluate the binary detector and multiclass classifier with
    per-class metrics, not accuracy alone.
-   Improve generalization to sensor failures that vary by sensor, time
    of day, and environmental conditions.

## System overview

``` text
Raw sensor CSV
    |
    v
Load, parse timestamps, inspect data quality
    |
    v
Timestamp-aware preprocessing
    |
    v
Chronological train / validation / test split
    |
    v
Inject labeled anomalies into each split
    |
    v
Feature engineering
    |
    v
Missing-value handling and derived-feature refresh
    |
    v
Fit scaler on training data; transform validation/test
    |
    v
Create 20-timestep sequences
    |
    +-------------------------------+
    |                               |
    v                               v
Model 1: Normal / Anomaly       Model 2: anomaly type
(binary LSTM)                   (six-class LSTM)
    |                               ^
    +-- if anomalous ----------------+
```

Model 2 is intended to be conditional: it classifies the anomaly type
only after Model 1 has predicted an anomaly. Model 2 was also evaluated
independently on its labeled multiclass test sequences to establish a
baseline.

------------------------------------------------------------------------

## Repository and data organization

The repository root used during development is:

``` text
/home/jetson/Desktop/ciesol-anomaly-detection
```

The project uses a Python virtual environment at:

``` text
/home/jetson/Desktop/ciesol-anomaly-detection/env
```

The exact contents of the repository may evolve. The paths below reflect
files and directories used during the current workflow.

``` text
ciesol-anomaly-detection/
├── data/
│   ├── raw/
│   │   └── sensor_data_raw2.csv
│   ├── processed/
│   │   └── clean_data.parquet
│   ├── injected/
│   │   ├── train_injected_physical.parquet
│   │   └── ... validation/test injected data ...
│   ├── model2/
│   │   ├── train_model2.parquet
│   │   ├── validation_model2.parquet
│   │   └── test_model2.parquet
│   ├── scaled/
│   │   └── model2/
│   │       └── ... scaled arrays and preprocessing artifacts ...
│   └── sequences/
│       ├── ... Model 1 sequence arrays ...
│       └── model2_v2/
│           ├── train_model2_X.npy
│           ├── train_model2_y.npy
│           ├── validation_model2_X.npy
│           ├── validation_model2_y.npy
│           ├── test_model2_X.npy
│           └── test_model2_y.npy
├── models/
│   ├── lstm_classifier_best.keras
│   └── model2_v2/
│       └── lstm_model2_best.keras
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_sensor_eda.ipynb
│   └── ... modeling notebooks ...
└── src/
    └── ... reusable preprocessing, feature, injection, and sequence logic ...
```

This is a **logical guide**, not a guarantee that every listed path or
notebook currently exists with exactly this name. In particular, the
project has used additional notebooks for model preparation, training,
and evaluation.

### Important file-protection rule

`data/processed/clean_data.parquet` is the accepted cleaned-data output
and **must not be overwritten** by later processing experiments. Write
new intermediate or experimental outputs to clearly named
files/directories.

### Notebook working directory

VS Code notebooks may start with the working directory set to
`notebooks/`. Relative paths can therefore accidentally write files
under `notebooks/data/` instead of the repository's top-level `data/`.

Use a project-root variable and build paths from it, rather than relying
on the current working directory. Reusable project logic should live in
`src/`; notebooks should call that logic rather than duplicate it.

------------------------------------------------------------------------

## Dataset

### Raw source

-   File: `data/raw/sensor_data_raw2.csv`
-   Delimiter: semicolon (`;`)
-   Original timestamp column: `Fecha`
-   Original period: **10 October 2020 to 29 December 2020**
-   Original number of rows: **233,381**
-   Original sensor variables: **12**

The raw timestamp format is day-first in at least some records. Parsing
with `%m/%d/%Y` failed on a value such as `13/10/2020 00:00:06`; the
parsing logic must handle the actual date format consistently.

### Sensor variables

  -----------------------------------------------------------------------
  Sensor                              Description recorded during the
                                      project
  ----------------------------------- -----------------------------------
  `PCO2EXT`                           Exterior CO₂

  `PHEXT`                             Exterior relative humidity

  `PRAD`                              Solar radiation

  `PRGINT`                            Interior/global radiation
                                      measurement

  `PTEXT`                             Exterior air temperature

  `PVV`                               Wind

  `UVENT_cen`                         Vent/window opening percentage,
                                      central

  `UVENT_lN`                          Vent/window opening percentage,
                                      another opening measurement

  `XCO2I`                             Interior CO₂

  `XHINV`                             Interior humidity-related
                                      measurement; source documentation
                                      has a naming mismatch (`XHINV`
                                      versus `XHINT`)

  `XTINV`                             Interior air temperature

  `XTS`                               Soil temperature
  -----------------------------------------------------------------------

The sensor descriptions should be checked against the authoritative
Agroconnect instrumentation documentation before publication or operational
use. `UVENT_cen` and `UVENT_lN` are treated as contextual variables in
the anomaly-injection design, while `XTS` is excluded from the selected
anomaly-target sensors.

### Initial exploratory observations

The early sensor EDA identified several relationships and ranges worth
considering during feature engineering:

-   `PRAD` and `PRGINT` were highly correlated (approximately **0.99**).
-   `PRAD` had a positive association with `UVENT_cen` (approximately
    **0.75**), `UVENT_lN` (approximately **0.76**), and `PTEXT`
    (approximately **0.61**).
-   `PRAD` had a negative association with `PHEXT` (approximately
    **−0.44**).
-   `PCO2EXT` was often around 410--465 in the observed data.
-   `PTEXT` was roughly 4--27 and `XTINV` roughly 7--36 over the
    inspected period.
-   `XHINV` showed a strong day/night pattern, with some nighttime
    values around 80--90 and daytime values around 30.
-   `UVENT_lN` was frequently around 50% from mid-December in the
    inspected data.

These are exploratory observations, not guaranteed physical bounds or
universal rules.

------------------------------------------------------------------------

## Data preparation and quality decisions

### Timestamp parsing and interval inspection

The raw timestamp column is `Fecha`; later train/validation/test tables
use `FECHA`.

The original timestamps are not perfectly spaced. Observed differences
included:

  Difference between records       Count   Approximate share
  ---------------------------- --------- -------------------
  30 seconds                     138,875              59.51%
  31 seconds                      84,907              36.38%
  20 seconds                       6,076                 ---
  21 seconds                       2,898                 ---

The important decision was **not to resample all data blindly to exactly
30 seconds**. Real 30--31 second timing variation is retained.

### Accepted timestamp-aware preprocessing

The accepted preprocessing output contained:

-   **233,616 rows** after processing
-   **283 inserted timestamps**
-   **265 detected episodes**
-   **283 rows** marked with `timestamp_was_missing = True`
-   **283 rows** marked as `short_gap`

The gap/timing categories used in the project were:

  -----------------------------------------------------------------------
  Category                Approximate interval    Treatment
                          rule                    
  ----------------------- ----------------------- -----------------------
  `normal_jitter`         29--32 seconds          Preserve as normal
                                                  timing variation

  `timing_jitter`         \>32 to 60 seconds      Preserve and flag as
                                                  timing jitter

  `short_gap`             \>60 seconds to 5.5     Insert expected
                          minutes                 timestamps and track
                                                  missingness

  `medium_gap`            \>5.5 to 10 minutes     Treat as unsafe for
                                                  interpolation; leave
                                                  missing values

  `long_gap`              \>10 minutes            Treat as unsafe for
                                                  interpolation; leave
                                                  missing values
  -----------------------------------------------------------------------

These thresholds are project-specific choices. The preprocessing process
records inserted timestamps and gap categories rather than silently
treating every inserted row as an observed sensor reading.

### Missing values and interpolation policy

Later missing-value handling uses an interpolation limit of up to 60
seconds and restricts interpolation to interior gaps
(`limit_area="inside"`). This is designed to avoid extrapolating at the
start or end of a series and to avoid filling long outages as though
they were reliable measurements.

The processed file `data/processed/clean_data.parquet` is retained as a
protected output. Later stages should consume it or create new named
outputs instead of overwriting it.

### Chronological split

The project uses chronological train/validation/test partitions to
respect time-series order and reduce temporal leakage. The split was
described during development as approximately **60 days train / 10 days
validation / 10 days test**.

The injection and sequence stages have these observed row/sequence
counts, which are not identical because preprocessing, labels, and
sequence windows affect the final counts:

  -----------------------------------------------------------------------
  Split                    Data rows in Model 2 Model 2 labeled sequences
                                   source table 
  ------------------- ------------------------- -------------------------
  Train                                 172,873                    24,463

  Validation                             28,800                     4,730

  Test                                   31,680                     4,395
  -----------------------------------------------------------------------

Use the saved split artifacts as the source of truth for exact
boundaries and counts in any future run.

------------------------------------------------------------------------

## Anomaly injection and labels

The project uses controlled anomaly injection so that the models can be
trained with known target labels. Train, validation, and test data all
received injected anomalies, allowing evaluation against labels for each
split.

Injection types used in the binary detector and multiclass classifier
include:

1.  **Contextual**
2.  **Ruido** (noise)
3.  **Desviacion de Correlacion** (correlation deviation)
4.  **Datos Faltantes** (missing data)
5.  **Valores Fuera de Rango** (out-of-range values)
6.  **Sensor Atascado** (stuck sensor)

The Model 2 class IDs are fixed as follows:

    Class ID Class label
  ---------- ---------------------------
           0 Contextual
           1 Ruido
           2 Desviacion de Correlacion
           3 Datos Faltantes
           4 Valores Fuera de Rango
           5 Sensor Atascado

Keep this mapping identical during training, evaluation, inference, and
reporting.

### Target and contextual sensors

The selected anomaly-target sensors are:

``` text
PCO2EXT, PHEXT, PRAD, PRGINT, PTEXT, PVV, XCO2I, XHINV, XTINV
```

The contextual-only ventilation variables are:

``` text
UVENT_cen, UVENT_lN
```

`XTS` was excluded from the selected anomaly-target sensors.

### Traceability metadata

Model 2 source DataFrames include columns such as:

-   `sensor_anomalia`
-   `anomalia_id`
-   `tipo_anomalia`
-   `fuente_anomalia`
-   `valor_original_anomalia`
-   `valor_inyectado_anomalia`
-   `any_sensor_was_missing`

These are useful for diagnostics and auditability. They must **not**
accidentally enter the model input features if they directly reveal the
target or the injection process.

An earlier injection bug was found where some noise/out-of-range
injections wrote older `etiqueta_*` column names. That caused **8,643
rows** to have `sensor_anomalia` but empty traceability. The injection
implementation was then directed to use consistent traceability names.
Always run label/traceability consistency checks after injection.

An injected `PRGINT` example also showed that a `radiation_ratio`
feature could become extreme (including a negative ratio after
injection). `radiation_ratio` was removed from the feature set to avoid
that fragile derived feature.

### Injection methodology and pipeline mechanics

The end-to-end anomaly injection process is fully implemented in [`notebooks/05_data_injection.ipynb`](notebooks/05_data_injection.ipynb). It operates on unscaled physical sensor values chronologically across `df_train_injected`, `df_validation_injected`, and `df_test_eval`.

#### 1. Pipeline Execution & Overlap Prevention
Injection is executed **sequentially** in a strict pipeline to prevent conflicting or overlapping anomalies on any single row:
```text
Clean Unscaled Data (Train / Val / Test)
       │
       ▼
① Sensor Atascado (Stuck Sensor)
       │
       ▼
② Datos Faltantes (Missing Data)
       │
       ▼
③ Ruido / Picos (Noise / Spikes)
       │
       ▼
④ Valores Fuera de Rango (Out-of-Range)
       │
       ▼
⑤ Desviación de Correlación (Correlation Deviation)
       │
       ▼
⑥ Contextual Anomalies (3 Sub-types)
       │
       ▼
Final Audit & Verification
```
- A global cumulative boolean mask `used_rows` tracks every modified row.
- Before injecting an anomaly, the candidate selector checks `~used_rows`. Any row previously modified by an earlier injection type is excluded.
- Unmodified reference copies (`df_train_clean`, `df_validation_clean`, `df_test_clean`) are preserved untouched throughout execution.
- Reproducibility across splits is maintained using BLAKE2b deterministic hashing (`stable_seed(base_seed, dataset_name, anomaly_type, sensor)`).

#### 2. Detailed Injection Logic by Anomaly Type

1. **Sensor Atascado (Stuck Sensor):**
   - **Target sensors:** 8 sensors (`PCO2EXT`, `PHEXT`, `PRGINT`, `PTEXT`, `PVV`, `XCO2I`, `XHINV`, `XTINV`; `PRAD` excluded).
   - **Configuration:** 4 multi-step events per sensor. Durations vary per physical sensor dynamics:
     - `PTEXT`: 16–40 timesteps
     - `PCO2EXT`, `XTINV`, `PVV`, `XCO2I`, `PRGINT`: 20–60 timesteps
     - `XHINV`: 28–70 timesteps
     - `PHEXT`: 32–80 timesteps
   - **Candidate Selection:**
     - Must occur in periods with natural variability (rolling 30-step standard deviation $\ge$ 20th percentile).
     - Temperature sensors (`PTEXT`, `XTINV`) are restricted to morning/afternoon hours with solar radiation `PRAD` $\ge$ 5.0 W/m² and minimum temperature swing $\ge$ 1.5°C to ensure the freeze is physically detectable.
     - `XCO2I` stuck candidates are verified against local median windows to avoid creating extreme outlier events.
   - **Injection:** Replaces the window `[start, end)` with the immediately preceding value: $x_{t} = x_{\text{start}-1}$.

2. **Datos Faltantes (Missing Data):**
   - **Target:** 2% of available complete rows; exactly 1 target sensor selected per row.
   - **Condition:** Must have non-null readings in all target sensors; respects physical activity thresholds (e.g., `PVV` $\ge$ 3.0 m/s, `PRGINT` $\ge$ 50.0 W/m²).
   - **Injection:** Selected sensor value is set to `np.nan`. Traceability values (`valor_original_anomalia`, `valor_inyectado_anomalia`) remain `NaN` by convention.

3. **Ruido / Picos (Noise / Spikes):**
   - **Target:** 3% of available rows distributed evenly across sensors.
   - **Injection:** Injects additive or subtractive impulses drawn from realistic, sensor-specific delta intervals:
     - `PTEXT`, `XTINV`: $\pm 3.0$ to $8.0\ ^\circ\text{C}$
     - `PHEXT`, `XHINV`: $\pm 10.0$ to $25.0\%$
     - `PCO2EXT`: $\pm 25.0$ to $75.0\ \text{ppm}$
     - `XCO2I`: $\pm 60.0$ to $160.0\ \text{ppm}$
     - `PVV`: $+5.0$ to $9.0\ \text{m/s}$ (positive only, requires current wind $\ge 5.0\ \text{m/s}$)
     - `PRAD`: $+150.0$ to $500.0\ \text{W/m}^2$ (daytime only: 8:00–19:00, radiation $\ge 50\ \text{W/m}^2$)
     - `PRGINT`: $+80.0$ to $250.0\ \text{W/m}^2$ (radiation $\ge 100\ \text{W/m}^2$)
   - Clipped to ensure the resulting value remains strictly within valid sensor physical bounds (`RANGOS_FISICOS`).

4. **Valores Fuera de Rango (Out-of-Range):**
   - **Target:** 2% of available rows across target sensors.
   - **Mechanism:** Injects an impossible reading by displacing the physical limit by an error factor of 10% ($\text{span} \times 0.10$):
     - 50% probability: $x_{\text{new}} = \text{sensor\_min} - 0.10 \times (\text{sensor\_max} - \text{sensor\_min})$
     - 50% probability: $x_{\text{new}} = \text{sensor\_max} + 0.10 \times (\text{sensor\_max} - \text{sensor\_min})$
   - Ensures values break physical laws (e.g., relative humidity $< 0\%$ or $> 100\%$).

5. **Desviación de Correlación (Correlation Deviation):**
   - **Target:** 3% of available daytime rows (hours 7:00 to 18:00).
   - **Reference Statistics (No Data Leakage):** Correlation matrix and quartile boundaries ($P_{25}, P_{50}, P_{75}$) are computed **exclusively on clean Train** data.
   - **Physical Pairs Injected:**
     - `PRAD` $\leftrightarrow$ `PRGINT` (positive correlation)
     - `XTINV` $\leftrightarrow$ `XHINV` (negative correlation)
     - `PTEXT` $\leftrightarrow$ `XTINV` (positive correlation)
     - `PHEXT` $\leftrightarrow$ `XHINV` (positive correlation)
   - **Mechanism:** Keeps reference sensor $s_B$ intact, while setting target sensor $s_A$ to contradict $s_B$:
     - For positive correlation ($r > 0.3$): if $s_B > P_{75}(s_B)$, force $s_A = P_{25}(s_A)$; if $s_B < P_{25}(s_B)$, force $s_A = P_{75}(s_A)$.
     - For negative correlation ($r < -0.3$): if $s_B > P_{75}(s_B)$, force $s_A = P_{75}(s_A)$; if $s_B < P_{25}(s_B)$, force $s_A = P_{25}(s_A)$.

6. **Contextual Anomalies (3 Scenarios):**
   - **Sub-type 1 (PRAD / PRGINT Ratio Discrepancy):** Injects on 2% of rows where outdoor radiation `PRAD` $> 200\ \text{W/m}^2$ and normal interior/exterior ratio $> 0.40$, forcing `PRGINT` to drop to an unnaturally dark ratio ($0.02$ to $0.08 \times \text{PRAD}$).
   - **Sub-type 2 (CO₂ Accumulation with Open Vents):** Injects 11 multi-step episodes (duration 3–8 steps) during daytime (`PRAD` $> 50\ \text{W/m}^2$) where vents are open (`UVENT_cen` or `UVENT_lN` $> 40\%$), forcing interior `XCO2I` to an elevated level of 700–1200 ppm despite active natural ventilation.
   - **Sub-type 3 (Excessive Solar Temperature Inversion):** Injects on 1.6% of rows where `PRAD` $> 400\ \text{W/m}^2$ and outdoor temperature `PTEXT` $> 20\ ^\circ\text{C}$, elevating interior temperature `XTINV` by $+3$ to $+10\ ^\circ\text{C}$ above exterior temperature.

#### 3. Post-Injection Verification & Traceability Audits
Each injection step runs automated verification functions (`audit_sensor_atascado`, `audit_missing_data`, `audit_ruido`, `audit_oor`, `audit_desviacion_correlacion`, `validate_anomaly_id_consistency`):
- Verifies exact match between event records and modified dataframe rows.
- Asserts that all injected values differ from originals (except missing data).
- Confirms zero missing values across the 6 metadata columns for every anomaly row.
- Validates 1:1 mapping between `anomalia_id`, anomaly type, sensor, and source generator.

### Injection output examples

An earlier physical-injection output was stored at:

``` text
data/injected/train_injected_physical.parquet
```

Model 2's diagnostic source tables were stored at:

``` text
data/model2/train_model2.parquet
data/model2/validation_model2.parquet
data/model2/test_model2.parquet
```

Treat these as generated artifacts, and regenerate them only through the
documented pipeline so that labels and traceability remain consistent.

------------------------------------------------------------------------

## Feature engineering

Feature engineering is applied separately to the injected train,
validation, and test data. The feature-engineering function is intended
to create features; it does not itself inject anomalies or fit a scaler.

The original 12 sensor variables are combined with time and physically
motivated derived features. Model 1's sequence input contains **19
features**. The documented additions include:

-   `hour_sin`
-   `hour_cos`
-   `is_night`
-   `delta_temp = XTINV - PTEXT`
-   `delta_hum = XHINV - PHEXT`
-   `co2_gradient = XCO2I - PCO2EXT`

The feature count can differ between intermediate stages because
missingness indicators are added later. For example,
`any_sensor_was_missing` and per-sensor `<sensor>_was_missing` flags are
created by missing-value handling. Do not assume that every intermediate
DataFrame has the same feature count as the final model input.

Model 2 v2 uses **146 features per timestep** in its final selected
feature list (`MODEL2_FINAL_FEATURES`). The scaler and feature-list
artifacts are stored under `data/scaled/model2/`. The full feature list
and its order must be loaded from the saved project artifact rather than
reconstructed manually.

### Leakage prevention

Before training:

-   Exclude class labels and direct injection traceability from `X`.
-   Exclude any identifiers that would allow the model to memorize an
    injection record.
-   Fit transformations using training data only.
-   Apply the saved training transformations unchanged to validation and
    test.
-   Check that the target sequence's features do not include future
    observations beyond its 20-timestep window.

------------------------------------------------------------------------

## Missing-value handling

The missing-value stage adds explicit missingness indicators and applies
the project's short-gap interpolation policy.

Key behavior:

-   Creates `<sensor>_was_missing` indicators for applicable sensors.
-   Creates `any_sensor_was_missing`.
-   Uses limited interpolation (maximum 60 seconds) only for interior
    gaps.
-   Recomputes derived features after filling, so differences and other
    engineered features reflect the handled sensor values.
-   Includes checks such as `find_unfilled_injected_nans` and
    `find_non_injected_sensor_nans`.
-   Checks that `FECHA` is monotonic and has no duplicate timestamps.

Missingness can itself be an anomaly class. Therefore, filling a value
must not erase the fact that the original value was missing: missingness
flags and injected anomaly labels should be preserved.

------------------------------------------------------------------------

## Scaling and sequence generation

### Scaling

The scaler is fitted using training data only, then reused for
validation and test. Do not fit a new scaler on validation or test data.
Keep the ordered final feature list alongside the scaler.

Model 2 scaled arrays were saved under:

``` text
data/scaled/model2/
```

The workflow has used files such as:

``` text
X_train_model2_scaled.npy
```

along with related validation/test arrays and serialized preprocessing
artifacts. Exact filenames should be confirmed from the current
directory before rerunning or loading them.

### Sequences

Both models use a sliding window of **20 timesteps**. Each sequence
represents recent history, and the target label corresponds to the final
timestep in that sequence.

Model 1 saved sequence files include:

``` text
data/sequences/
    train_lstm_sequences_X.npy
    train_lstm_sequences_y.npy
    validation_lstm_sequences_X.npy
    validation_lstm_sequences_y.npy
    test_lstm_sequences_X.npy
    test_lstm_sequences_y.npy
```

Model 1's known array shapes include:

-   `X_train`: `(172854, 20, 19)`
-   `X_test`: `(31661, 20, 19)`
-   `y_train`: `(172854,)`

Model 2 v2 files are under:

``` text
data/sequences/model2_v2/
```

  Split        X shape              y shape
  ------------ -------------------- ------------
  Train        `(24463, 20, 146)`   `(24463,)`
  Validation   `(4730, 20, 146)`    `(4730,)`
  Test         `(4395, 20, 146)`    `(4395,)`

Model 2's final timestep is the target row; the preceding 19 steps
provide temporal context.

------------------------------------------------------------------------

## Model 1: binary LSTM detector

### Task

Model 1 classifies a sequence as:

-   `0` --- Normal
-   `1` --- Anomaly

It is the first stage of the intended inference pipeline. If it predicts
Normal, the pipeline can stop. If it predicts Anomaly, Model 2 can be
called to predict the anomaly type.

### Architecture

The binary model is an LSTM sequence classifier with two recurrent
layers, followed by dense classification layers. The recorded
architecture included:

-   LSTM with 64 units and sequence output
-   LSTM with 32 units
-   Dense output for binary classification

Training used class weights calculated from training labels,
EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau. The best saved
model is:

``` text
models/lstm_classifier_best.keras
```

The recorded class weights were approximately:

-   Normal: `0.5824`
-   Anomaly: `3.5330`

### Training notes

-   Training ran for up to 50 epochs, with early stopping.
-   Best validation PR-AUC was approximately **0.9265** at epoch 6.
-   Best validation ROC-AUC was approximately **0.9579** at epoch 5.
-   Lowest validation loss was approximately **0.1228** at epoch 3.
-   Early stopping ended training at epoch 14 in the recorded run.

The best checkpoint was selected using validation performance rather
than simply using the final epoch.

### Test results

The recorded test threshold was **0.24**.

  Metric        Result
  ----------- --------
  ROC-AUC       0.9688
  PR-AUC        0.9410
  Accuracy      0.9792
  Precision     0.9645
  Recall        0.8828
  F1-score      0.9218

Confusion matrix, with rows as actual labels and columns as predicted
labels:

``` text
                 Pred Normal   Pred Anomaly
Actual Normal        27123          143
Actual Anomaly         515         3880
```

The test set contained 27,266 Normal sequences and 4,395 Anomaly
sequences. At threshold 0.24, the model detected 3,880 of the 4,395
anomalous sequences, with 143 false positives. This is a strong
baseline, although the threshold should be selected according to the
intended operational cost of false positives versus missed anomalies.

------------------------------------------------------------------------

## Model 2: multiclass LSTM classifier

### Task and labels

Model 2 predicts one of six anomaly categories, using the class mapping
in [Anomaly injection and labels](#anomaly-injection-and-labels). It is
intended to run only after Model 1 flags an anomaly.

Model 2 v2's input shape is:

``` text
(sequence_length, number_of_features) = (20, 146)
```

### Architecture

The saved v2 architecture is:

``` python
model2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
])

model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

The training procedure used:

-   Class weights calculated from training data only.
-   A custom validation macro-F1 callback.
-   ModelCheckpoint monitoring `val_macro_f1`.
-   EarlyStopping monitoring `val_macro_f1`.
-   ReduceLROnPlateau monitoring validation loss.

The run stopped at epoch 12 and restored the best checkpoint from epoch
4. The best validation macro-F1 was approximately **0.8528**. The gap
between training performance and validation performance indicated
overfitting risk.

Saved best model:

``` text
models/model2_v2/lstm_model2_best.keras
```

### Validation results

  Class                          Precision       Recall     F1-score   Support
  --------------------------- ------------ ------------ ------------ ---------
  Contextual                        0.9731       0.9607       0.9669     1,093
  Ruido                             0.6980       0.8426       0.7635       864
  Desviacion de Correlacion         0.7361       0.9329       0.8229       864
  Datos Faltantes                   1.0000       0.9931       0.9965       576
  Valores Fuera de Rango            0.9983       0.9965       0.9974       576
  Sensor Atascado                   0.8743       0.4227       0.5699       757
  **Macro average**             **0.8800**   **0.8581**   **0.8528**       ---
  **Weighted average**          **0.8701**   **0.8562**   **0.8472**       ---

Validation accuracy was **0.8562**.

Validation confusion matrix (actual rows, predicted columns; class order
is 0 through 5):

``` text
[[1050   22   20    0    0    1]
 [   1  728   99    0    0   36]
 [  27   27  806    0    1    3]
 [   0    0    0  572    0    4]
 [   0    0    0    0  574    2]
 [   1  266  170    0    0  320]]
```

The principal weakness is `Sensor Atascado`: only 320 of 757 examples
were classified correctly. Many were predicted as `Ruido` (266) or
`Desviacion de Correlacion` (170).

### Test results

The saved best v2 model was then evaluated on the test set.

  Class                          Precision       Recall     F1-score   Support
  --------------------------- ------------ ------------ ------------ ---------
  Contextual                        0.9830       0.9844       0.9837       705
  Ruido                             0.7418       0.8558       0.7947       950
  Desviacion de Correlacion         0.7839       0.9621       0.8639       950
  Datos Faltantes                   1.0000       0.9968       0.9984       633
  Valores Fuera de Rango            1.0000       1.0000       1.0000       634
  Sensor Atascado                   0.8210       0.2543       0.3883       523
  **Macro average**             **0.8883**   **0.8422**   **0.8382**       ---
  **Weighted average**          **0.8734**   **0.8689**   **0.8506**       ---

Test accuracy was **0.8689**.

Test confusion matrix (actual rows, predicted columns):

``` text
[[694   8   2   0   0   1]
 [  0 813 115   0   0  22]
 [  4  27 914   0   0   5]
 [  0   1   0 631   0   1]
 [  0   0   0   0 634   0]
 [  8 247 135   0   0 133]]
```

For `Sensor Atascado`, only 133 of 523 test examples were correctly
identified. The other 390 were predicted as `Ruido` (247),
`Desviacion de Correlacion` (135), or `Contextual` (8).

**Evaluation protocol note:** The test results above are an observed
baseline. Since the test set has already been evaluated, future
iterations should be selected using validation data. Avoid repeatedly
adjusting injection, features, thresholds, or architecture based on test
performance. Use the test set again only for a final evaluation of a
selected candidate, and document that the baseline test metrics have
already been observed.

------------------------------------------------------------------------

## What the results mean

Model 2 v2 is not uniformly weak. It recognizes several classes
extremely well:

-   `Datos Faltantes` and `Valores Fuera de Rango` have near-perfect
    validation and test results.
-   `Contextual` is also classified reliably.
-   `Ruido` and `Desviacion de Correlacion` have useful recall, but some
    confusion remains between these classes.
-   `Sensor Atascado` is the main failure mode, with recall falling from
    **42.27% on validation** to **25.43% on test**.

This pattern means that simply changing the softmax decision threshold
is unlikely to solve the problem. Diagnostics showed that, for missed
validation stuck-sensor examples, the true-class probability was very
low for most examples while the model assigned high probability to
another class. The model is often making a confident wrong
classification, rather than assigning the correct class a high score
that narrowly misses a threshold.

### Evidence of a distribution mismatch

The train/validation diagnostics found meaningful differences among
`Sensor Atascado` examples:

-   The time-of-day distributions differ. The training set had examples
    across morning, afternoon, evening, and night; validation had no
    examples in the defined evening period.
-   The mean hour was approximately 12.73 in train and 10.92 in
    validation.
-   The affected-sensor composition differed. For example, `XTINV`
    represented approximately 12.43% of train stuck-sensor examples but
    0% of validation examples; `XHINV` and `PRGINT` represented larger
    shares in validation than train.
-   Several scaled environmental and engineered features differed
    substantially between train and validation stuck-sensor examples,
    including `XTS`, `XTINV`, radiation-related features, temperature
    context, humidity, and ventilation-related features.

These findings support a **distribution-shift / insufficient-diversity
hypothesis**, but they do not prove it is the only cause. The injection
implementation, feature construction, label semantics, and
sequence-building logic should also be checked.

### Should more data be collected?

More representative examples are likely to help, but the priority is
**coverage and diversity**, not simply a larger row count. The training
examples should represent stuck-sensor behavior across:

-   All intended target sensors.
-   Different times of day and environmental regimes.
-   Different stuck values and durations.
-   Flatlines at realistic low, middle, and high values.
-   Transitions into and out of the stuck state.
-   Different background conditions and correlated sensor behavior.

If new real labeled failure data can be collected, it would be valuable.
In the meantime, improve the injection strategy so it covers realistic
variations without creating shortcuts that make injected examples
artificially easy or different from real failures.

------------------------------------------------------------------------

## Inference and latency experiments

A simple one-sequence-at-a-time `model.predict()` benchmark showed
substantial overhead in the development environment:

-   One run reported an average around 123 ms per sequence, with a
    maximum around 691 ms.
-   A later run reported an average around 189.7 ms, median 189.3 ms,
    and 95th percentile around 193.9 ms.

An optimized TensorFlow function using XLA compilation was also tested:

``` python
@tf.function(jit_compile=True, reduce_retracing=True)
def predict_one(x):
    return best_model(x, training=False)
```

The benchmark used a float32 tensor and warmed up the compiled function
before timing. The recorded result was:

  Metric                                        Result
  ------------------------ ---------------------------
  Mean latency                       0.892 ms/sequence
  Median latency                     0.874 ms/sequence
  95th percentile                             0.998 ms
  99th percentile                             1.037 ms
  Maximum                                     2.619 ms
  Approximate throughput     1,120.94 sequences/second

These figures are **benchmark-specific**, not production guarantees.
They depend on hardware, TensorFlow build, compilation warm-up, input
shape, batching, and the exact timing method. The development
environment reported no TensorFlow GPU devices
(`tf.config.list_physical_devices("GPU")` returned `[]`), so these
measurements should not be described as GPU performance.

Before production use, benchmark the complete path, including feature
calculation, missing-value handling, scaling, sequence buffering, Model
1, conditional Model 2 inference, and any logging or communications.
Measure cold-start time, warm latency, sustained throughput, memory use,
and behavior when readings are missing or delayed. XLA can be useful in
production, but the compiled path must be tested on the target
deployment hardware and TensorFlow environment.

------------------------------------------------------------------------

## Reproducibility notes

### Environment

The recorded development environment included:

-   Python 3.10 virtual environment under `env/`
-   TensorFlow 2.21.0
-   No TensorFlow GPU device detected in the inspected environment

A fresh setup should use the project's pinned dependency file if
available. If no dependency lock file exists yet, create one from a
verified working environment before relying on exact reproducibility. Do
not assume that every package version is captured by this README.

### Safe rerun order

The intended processing order is:

1.  Load the raw CSV and parse `Fecha`.
2.  Inspect duplicates, timestamp order, interval distribution, and
    missing values.
3.  Run timestamp-aware preprocessing and save a new processed artifact.
4.  Validate the output without overwriting
    `data/processed/clean_data.parquet`.
5.  Create or load chronological train/validation/test splits.
6.  Inject anomalies separately in each split and validate class labels
    and traceability.
7.  Engineer features independently for each split.
8.  Handle missing values and recompute derived features.
9.  Fit the scaler on train only; transform validation and test with the
    same scaler.
10. Build 20-timestep sequences with targets aligned to the final
    timestep.
11. Train and select Model 1 using training and validation data.
12. Train and select Model 2 using training and validation data.
13. Evaluate selected models and save reports, confusion matrices, and
    configuration details.
14. Run the two-stage inference pipeline and benchmark it on deployment
    hardware.

### Artifacts worth saving with every model

For each model version, preserve:

-   Model file (`.keras`).
-   Ordered feature-name list.
-   Scaler and any other fitted preprocessing objects.
-   Sequence length and expected input shape.
-   Label-to-ID mapping.
-   Decision threshold for Model 1.
-   Training configuration, random seeds, and library versions.
-   Training history and model-selection metric.
-   Validation and test classification reports and confusion matrices.
-   A concise description of the data/injection version used.

Model weights alone are not sufficient to reproduce the predictions if
the feature order, scaler, sequence length, or label mapping changes.

------------------------------------------------------------------------

## Known limitations

1.  **Synthetic anomalies:** Injected anomalies provide controlled
    labels, but performance on injected data may not transfer to real
    sensor failures.
2.  **Stuck-sensor generalization:** This is the most important current
    weakness in Model 2 v2.
3.  **Possible split distribution shift:** The affected sensor and
    time-of-day distributions differ between train and validation.
4.  **Overfitting risk:** Model 2's best checkpoint occurred at epoch 4,
    despite training continuing to epoch 12.
5.  **Feature documentation:** The `XHINV`/`XHINT` naming mismatch
    should be resolved against authoritative sensor documentation.
6.  **Test set has been inspected:** Its current metrics are useful as a
    baseline, but future tuning should rely on validation.
7.  **Deployment latency is not yet end-to-end:** The fast XLA timing
    covers model inference, not the entire production pipeline.
8.  **Hardware configuration:** No GPU was visible to TensorFlow in the
    recorded environment.
9.  **Class and injection coverage:** Audit the stuck-sensor injection
    output to ensure all intended sensors and realistic regimes are
    represented. The diagnostic table showed seven affected sensors in
    the displayed output even though nine target sensors were intended;
    verify the full category counts rather than assuming complete
    coverage.

------------------------------------------------------------------------

## Recommended next steps

### Priority 1 --- Audit Sensor Atascado injection

-   Count stuck-sensor examples by split, affected sensor, hour/time
    period, stuck value, duration, and background conditions.
-   Verify that all nine intended target sensors can appear in the
    stuck-sensor class.
-   Compare the injection logic and feature distributions across train,
    validation, and test.
-   Inspect representative raw and injected sequences for each sensor.
-   Confirm that the label describes the same type of stuck behavior
    across every split.

### Priority 2 --- Improve training diversity

-   Add stuck-sensor patterns across sensors and
    environmental/time-of-day regimes.
-   Vary realistic stuck values and durations.
-   Avoid making the injected stuck class too similar to noise or
    correlation-deviation labels.
-   Keep injection rules and parameters documented and reproducible.
-   Prefer more representative examples over indiscriminate duplication
    of existing examples.

### Priority 3 --- Train a separate Model 2 v3

Keep `models/model2_v2/lstm_model2_best.keras` and all v2 reports
unchanged as the baseline. Train v3 as a separate experiment using the
same architecture initially, changing the injection/data coverage first
so that the effect is interpretable.

Select the candidate using validation macro-F1 and especially:

-   `Sensor Atascado` recall and F1.
-   Per-class precision/recall trade-offs.
-   Confusion between `Sensor Atascado`, `Ruido`, and
    `Desviacion de Correlacion`.
-   Overall macro-F1, not accuracy alone.

Only after selecting a candidate should a final test evaluation be
performed and reported, while acknowledging that the current test
baseline has already been inspected.

### Priority 4 --- Validate the full two-stage system

Once both models are stable, evaluate the end-to-end system:

1.  Model 1 predicts Normal or Anomaly.
2.  Only Model 1-positive sequences are passed to Model 2.
3.  Report binary detection performance, anomaly-type performance
    conditional on a true anomaly, and end-to-end type performance after
    accounting for Model 1 false negatives.
4.  Measure end-to-end latency on the intended hardware.

This matters because Model 2's standalone multiclass scores do not
include anomalies that Model 1 fails to detect.

------------------------------------------------------------------------

## Glossary

-   **Chronological split:** dividing time-series data by time order
    rather than randomly shuffling observations.
-   **Injected anomaly:** a deliberately modified or labeled example
    used to train and evaluate supervised detection.
-   **Sequence/window:** a fixed-length series of recent time steps used
    as one model input.
-   **Feature leakage:** information in the input that reveals the
    target or uses information unavailable at prediction time.
-   **Macro-F1:** the unweighted average of the F1-score for each class;
    useful when performance across all classes matters.
-   **Distribution shift:** a difference in the data distribution
    between training and evaluation sets.
-   **XLA:** TensorFlow's accelerated compilation path, which can reduce
    inference overhead after compilation but must be benchmarked in the
    target environment.

------------------------------------------------------------------------

## Project status summary

  -----------------------------------------------------------------------
  Component                           Status
  ----------------------------------- -----------------------------------
  Raw data loading and timestamp      Completed for the current dataset
  investigation                       

  Timestamp-aware preprocessing       Accepted output established

  Chronological train/validation/test Created
  data                                

  Anomaly injection and traceability  Implemented; further audit
                                      recommended for stuck-sensor
                                      coverage

  Feature engineering and             Implemented
  missing-value handling              

  Model 1 binary LSTM                 Trained and evaluated; strong
                                      baseline

  Model 2 v2 multiclass LSTM          Trained and evaluated

  Main Model 2 weakness               Low `Sensor Atascado` recall,
                                      especially on test

  XLA inference experiment            Completed; target-hardware
                                      validation still required

  Next major task                     Improve stuck-sensor injection
                                      diversity and train/evaluate Model
                                      2 v3
  -----------------------------------------------------------------------
