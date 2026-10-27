"""Frozen LightGBM evaluation on the physical LSTM test parquet."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.artifacts import file_sha256
from src.binary_postprocess import RULE_METHODS, binary_metrics
from .data import FEATURES, SENSORS, decorate_masks, stream_matrix
from .pipeline import _binary_predictions, load_run
from .settings import BASE_FEATURES, CLASSES, PROJECT, RUNS

PARQUET = PROJECT.parent / "Dataset" / "MismoTest" / "test_injected_physical.parquet"
RUN = RUNS / "20261013T084029635942Z"
OUTPUT = PROJECT / "data" / "common_physical_test_lightgbm_2020-12-20_29"
START, END = pd.Timestamp("2020-12-20"), pd.Timestamp("2020-12-30")
PLOT_START = pd.Timestamp("2020-12-21 09:09:46")
PLOT_END = PLOT_START + pd.Timedelta(hours=4)
CATEGORIES = {
    "Sensor Atascado": ("Stuck Sensor", "#E69F00", "D", 7, 28),
    "Ruido": ("Noise", "#56B4E9", "^", 6, 28),
    "Desviacion de Correlacion": ("Correlation Deviation", "#009E73", "s", 5, 24),
    "Datos Faltantes": ("Missing Data", "#CC79A7", "v", 4, 26),
    "Valores Fuera de Rango": ("Out of Range", "#D55E00", "P", 8, 30),
    "Contextual": ("Contextual", "#0072B2", "o", 3, 22),
}

def source():
    physical = pd.read_parquet(PARQUET)
    required = {"FECHA", "tipo_anomalia", *SENSORS}
    if missing := required - set(physical.columns):
        raise ValueError(f"Missing physical columns: {sorted(missing)}")
    dates = physical.FECHA
    if not pd.api.types.is_datetime64_any_dtype(dates) or dates.dt.tz is not None:
        raise ValueError("FECHA must contain naive local datetimes; inspect timezone before inference")
    if dates.isna().any() or not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("FECHA must be present, ordered and unique")
    if len(physical) != 31680:
        raise ValueError("Expected 31680 physical rows")
    daily = physical.groupby(dates.dt.normalize()).size()
    expected = pd.date_range("2020-12-19", "2020-12-29")
    if not daily.index.equals(expected) or not daily.eq(2880).all():
        raise ValueError("Expected 2880 rows on each of 11 days")
    for flag in ("sequence_unsafe", "has_unresolved_nan"):
        if flag in physical and physical[flag].fillna(False).any():
            raise ValueError(f"Unsafe physical rows: {flag}")
    if set(physical.tipo_anomalia.dropna()) - set(CLASSES):
        raise ValueError("Unknown truth class")
    return physical, file_sha256(PARQUET)

def provenance(digest):
    return {
        "input_sha256": digest,
        "run_manifest_sha256": file_sha256(RUN / "manifest.json"),
        "binary_manifest_sha256": file_sha256(RUN / "binary_manifest.json"),
        "feature_clock": "30 seconds by row from 2020-12-19 00:00:00; original FECHA retained",
        "warmup_rows": 360, "feature_order": list(FEATURES),
        "test_start": str(START), "test_end_exclusive": str(END),
    }

def evaluate(reuse=True):
    """Run once or reuse a hash-verified aligned trace."""
    physical, digest = source()
    prov = provenance(digest)
    trace_path, metrics_path = OUTPUT / "trace.parquet", OUTPUT / "metrics.json"
    binary, calibration, multiclass = load_run(RUN)
    if reuse and trace_path.exists() and metrics_path.exists():
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        if report.get("provenance") == prov and report.get("trace_sha256") == file_sha256(trace_path):
            trace = pd.read_parquet(trace_path)
            expected = physical.loc[(physical.FECHA >= START) & (physical.FECHA < END), "FECHA"].reset_index(drop=True)
            if len(trace) == 28800 and trace.FECHA.equals(expected):
                return trace, report, True
   # The source FECHA varies around 30 s. Do not round: floor(30 s) makes six
    # duplicate bins and six holes. The fixed clock is internal to features/rules.
    prepared = physical.loc[:, ["FECHA", "tipo_anomalia", *SENSORS]].copy()
    prepared["Fecha"] = pd.date_range("2020-12-19", periods=len(prepared), freq="30s")
    prepared = decorate_masks(prepared)
    warmup = prepared.loc[prepared.FECHA < START].iloc[-360:].reset_index(drop=True)
    test = prepared.loc[(prepared.FECHA >= START) & (prepared.FECHA < END)].reset_index(drop=True)
    if len(warmup) != 360 or len(test) != 28800:
        raise ValueError("Unexpected warmup or test coverage")
    matrix = stream_matrix(test, warmup, np.zeros(len(BASE_FEATURES), np.float32))
    if matrix.shape != (28800, 66) or not np.isfinite(matrix).all():
        raise ValueError("Feature matrix has wrong shape or nonfinite values")
    probability, detected, rules = _binary_predictions(binary, calibration, matrix, test, warmup)
    routed = np.flatnonzero(detected)
    predicted = np.full(len(test), "Normal", dtype=object)
    if len(routed):
        scores = np.asarray(multiclass.predict(matrix[routed], num_threads=1))
        if scores.shape != (len(routed), len(CLASSES)):
            raise ValueError("Wrong multiclass score shape")
        predicted[routed] = np.asarray(CLASSES, object)[scores.argmax(axis=1)]
    truth = test.tipo_anomalia.fillna("Normal").to_numpy()
    y = (truth != "Normal").astype(np.uint8)
    methods = [RULE_METHODS[int(code)] if code else
               ("lightgbm_debouncing" if hit else "normal")
               for code, hit in zip(rules, detected)]
    trace = pd.DataFrame({
        "FECHA": test.FECHA.to_numpy(), "PRAD": test.PRAD.to_numpy(),
        "verdad_binaria": y, "tipo_real": truth,
        "probabilidad_binaria": probability, "pred_binaria": detected,
        "metodo_deteccion_binaria": methods, "tipo_predicho": predicted,
    })
    if not trace.FECHA.equals(test.FECHA) or not trace.FECHA.is_unique:
        raise ValueError("Trace is not aligned to original FECHA")
    if not ((trace.pred_binaria.eq(0) & trace.tipo_predicho.eq("Normal")) |
            (trace.pred_binaria.eq(1) & trace.tipo_predicho.isin(CLASSES))).all():
        raise ValueError("Routing and type prediction disagree")
    labels = ("Normal",) + CLASSES
    report = {
        "provenance": prov, "rows": len(trace), "true_anomalies": int(y.sum()),
        "routed_to_multiclass": len(routed),
        "without_multiclass_prediction": len(trace)-len(routed),
        "binary": binary_metrics(y, detected, 10.0),
        "labels": labels,
        "end_to_end_macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "end_to_end_confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "end_to_end_report": classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0),
        "predicted_type_counts": trace.tipo_predicho.value_counts().to_dict(),
        "calibration": calibration.canonical(),
        "first_FECHA": str(trace.FECHA.iloc[0]), "last_FECHA": str(trace.FECHA.iloc[-1]),
        "nominal_clock_offset_seconds": {
            "minimum": float((test.FECHA-test.Fecha).dt.total_seconds().min()),
            "maximum": float((test.FECHA-test.Fecha).dt.total_seconds().max()),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trace.to_parquet(trace_path, index=False)
    report["trace_sha256"] = file_sha256(trace_path)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace, report, False

def plot_prad(trace):
    """Plot predicted classes on PRAD, retaining physical NaN and original FECHA."""
    window = trace.loc[(trace.FECHA >= PLOT_START) & (trace.FECHA < PLOT_END)].copy()
    if window.empty:
        raise ValueError("No rows in PRAD window")
    counts = window.loc[window.pred_binaria.eq(1), "tipo_predicho"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(window.FECHA, window.PRAD, color="#444444", linewidth=0.8, alpha=0.9, zorder=2, label="PRAD signal")
    for category, count in counts.items():
        label, color, marker, zorder, size = CATEGORIES[category]
        marked = window.loc[window.tipo_predicho.eq(category)]
        ax.scatter(marked.FECHA, marked.PRAD, s=size, c=color, marker=marker,
                   alpha=0.88, linewidths=0.3, edgecolors="white", zorder=zorder,
                   label=f"{label} (n={count:,})")
    ax.set_title("(b) PRAD sensor — LightGBM predicted anomaly categories during a 4-hour period at 21-12-2020",
                 loc="left", fontweight="bold", pad=10)
    ax.set_ylabel("Global Radiation [W m⁻²]")
    ax.set_xlabel("Date / Time (UTC+1)")
    ax.set_xlim(PLOT_START, PLOT_END)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major", length=4, width=0.8)
    ax.tick_params(which="minor", length=2, width=0.5)
    handles, names = ax.get_legend_handles_labels()
    items = [(h, n) for h, n in zip(handles, names) if n != "PRAD signal"]
    items += [(h, n) for h, n in zip(handles, names) if n == "PRAD signal"]
    legend = ax.legend(*zip(*items), title="Predicted anomaly type", loc="upper right",
                       framealpha=0.92, edgecolor="#BBBBBB", handletextpad=0.4,
                       labelspacing=0.4, borderpad=0.6, ncol=2)
    legend._legend_box.align = "left"
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    stem = OUTPUT / "fig_anomaly_detection_4h_lightgbm_same_physical_2020-12-21"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    absent = window.PRAD.isna()
    return {
        "window": f"[{PLOT_START}, {PLOT_END})", "rows": len(window),
        "first_FECHA": str(window.FECHA.iloc[0]), "last_FECHA": str(window.FECHA.iloc[-1]),
        "predicted_type_counts": counts.to_dict(),
        "without_multiclass_prediction": int(window.pred_binaria.eq(0).sum()),
        "missing_PRAD": int(absent.sum()),
        "routed_markers_without_PRAD": int((absent & window.pred_binaria.eq(1)).sum()),
        "PRAD_min": float(window.PRAD.min()), "PRAD_max": float(window.PRAD.max()),
        "figure_png": str(stem.with_suffix(".png")), "figure_pdf": str(stem.with_suffix(".pdf")),
    }

