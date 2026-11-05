"""Publication figures for aligned final predictions over one PRAD interval."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from shared.contract import ANOMALIES, FILES, ROOT


STYLES = {
    "Contextual": ("#0072B2", "o"), "Noise": ("#56B4E9", "^"),
    "Correlation deviation": ("#009E73", "s"), "Missing data": ("#CC79A7", "v"),
    "Out-of-range": ("#D55E00", "P"), "Stuck sensor": ("#E69F00", "D"),
}
DEFAULT_START = pd.Timestamp("2020-12-24 10:00:19")
DEFAULT_PREFIX = "prad_4h_all_classes"


def load_window(start: pd.Timestamp, hours: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = pd.read_parquet(FILES["test"], columns=["FECHA", "PRAD", "sensor_anomalia", "tipo_anomalia"])
    light = pd.read_parquet(ROOT / "results" / "traces" / "lightgbm_night_test.parquet")
    lstm = pd.read_parquet(ROOT / "results" / "traces" / "lstm_test.parquet")
    if not physical.FECHA.reset_index(drop=True).equals(light.timestamp.reset_index(drop=True)):
        raise ValueError("LightGBM trace is not aligned with physical PRAD")
    keys = ["timestamp", "ground_truth_type", "data_manifest_id"]
    if not light[keys].equals(lstm[keys]):
        raise ValueError("The two traces do not describe the same physical targets")
    selected = physical.FECHA.ge(start) & physical.FECHA.lt(start + pd.Timedelta(hours=hours))
    window = pd.DataFrame({
        "timestamp": physical.loc[selected, "FECHA"].reset_index(drop=True),
        "PRAD": physical.loc[selected, "PRAD"].reset_index(drop=True),
        "truth": light.loc[selected, "ground_truth_type"].reset_index(drop=True),
        "LightGBM": light.loc[selected, "final_prediction"].reset_index(drop=True),
        "LSTM": lstm.loc[selected, "final_prediction"].reset_index(drop=True),
    })
    if window.empty:
        raise ValueError("The selected interval has no test targets")
    return window, physical.loc[selected]


def _panel(ax, window: pd.DataFrame, column: str, title: str, start: pd.Timestamp,
           end: pd.Timestamp, *, legend: bool) -> None:
    ax.plot(window.timestamp, window.PRAD, color="#444444", linewidth=0.85,
            alpha=0.9, label="PRAD signal", zorder=2)
    for category, (color, marker) in STYLES.items():
        chosen = window[column].eq(category)
        if not chosen.any():
            continue
        present = chosen & window.PRAD.notna()
        if present.any():
            ax.scatter(window.loc[present, "timestamp"], window.loc[present, "PRAD"],
                       s=32 if category == "Stuck sensor" else 27, c=color, marker=marker,
                       alpha=0.9, edgecolors="white", linewidths=0.35,
                       zorder=6 if category == "Stuck sensor" else 5,
                       label=f"{category} (n={int(chosen.sum())})")
        absent = chosen & window.PRAD.isna()
        if absent.any():
            ax.scatter(window.loc[absent, "timestamp"], [0] * int(absent.sum()), s=80,
                       c=color, marker=marker, edgecolors="black", linewidths=1.3,
                       zorder=7, label=f"{category} (n={int(chosen.sum())})" if not present.any() else None)
    ax.set_title(title, loc="left", fontweight="bold", pad=9)
    ax.set_ylabel("PRAD [W m$^{-2}$]")
    ax.set_xlim(start, end)
    ax.set_ylim(-25, float(window.PRAD.max()) * 1.12)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
    if legend:
        handles, names = ax.get_legend_handles_labels()
        order = [i for i, name in enumerate(names) if name != "PRAD signal"]
        order += [i for i, name in enumerate(names) if name == "PRAD signal"]
        ax.legend([handles[i] for i in order], [names[i] for i in order],
                  loc="upper right", framealpha=0.94, ncol=2, fontsize=8,
                  title="System-wide label")


def make_figures(start: pd.Timestamp = DEFAULT_START, *, hours: float = 4,
                 output_dir: Path | None = None, prefix: str = DEFAULT_PREFIX) -> list[Path]:
    if hours <= 0:
        raise ValueError("hours must be positive")
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(hours=hours)
    window, physical = load_window(start, hours)
    if start == DEFAULT_START and hours == 4:
        for column in ("truth", "LightGBM", "LSTM"):
            if set(ANOMALIES) - set(window[column]):
                raise ValueError(f"Default publication window lacks a category in {column}")
    if output_dir is None:
        output_dir = ROOT / "results" / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{start:%Y-%m-%d}" if start == DEFAULT_START and hours == 4 else f"{start:%Y-%m-%dT%H%M%S}"
    stuck_sensors = sorted(physical.loc[physical.tipo_anomalia.eq("Sensor Atascado"),
                                      "sensor_anomalia"].dropna().unique())
    note = "Markers describe the full sensor observation. "
    if stuck_sensors:
        note += "Stuck events here affect " + " or ".join(stuck_sensors) + "; PRAD is the background signal."
    elif window.PRAD.isna().any():
        note += "A black outline at zero denotes missing PRAD; other zero readings are measured."
    else:
        note += "PRAD is the background signal."

    def save(fig, stem: str) -> list[Path]:
        files = []
        for extension in ("png", "pdf"):
            destination = output_dir / f"{stem}.{extension}"
            fig.savefig(destination, dpi=300, bbox_inches="tight")
            files.append(destination)
        plt.close(fig)
        return files

    created = []
    for model, short in (("LightGBM", "lightgbm"), ("LSTM", "lstm")):
        fig, ax = plt.subplots(figsize=(12, 5.5))
        _panel(ax, window, model, f"PRAD signal and {model} final predictions | {start:%d %b %Y}",
               start, end, legend=True)
        ax.set_xlabel(f"Source timestamp, {start:%d %b %Y} (time zone unspecified)")
        fig.text(0.5, 0.01, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        created.extend(save(fig, f"{prefix}_{short}_{suffix}"))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
    for ax, column, title in zip(axes, ("truth", "LightGBM", "LSTM"),
                                 ("Ground truth", "LightGBM final prediction", "LSTM final prediction")):
        _panel(ax, window, column, title, start, end, legend=False)
    axes[-1].set_xlabel(f"Source timestamp, {start:%d %b %Y} (time zone unspecified)")
    handles = [Line2D([0], [0], marker=marker, linestyle="None", markerfacecolor=color,
                      markeredgecolor="white", markersize=7, label=category)
               for category, (color, marker) in STYLES.items()
               if window[["truth", "LightGBM", "LSTM"]].eq(category).any().any()]
    handles.append(Line2D([0], [0], color="#444444", linewidth=1, label="PRAD signal"))
    fig.legend(handles=handles, labels=[item.get_label() for item in handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4, fontsize=9)
    fig.suptitle(f"Same {len(window)} test targets | {start:%d %b %Y, %H:%M:%S}–{end:%H:%M:%S}",
                 y=0.995, fontweight="bold")
    fig.text(0.5, 0.02, note + " Source time zone unspecified.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 0.92))
    created.extend(save(fig, f"{prefix}_paired_{suffix}"))
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=pd.Timestamp, default=DEFAULT_START)
    parser.add_argument("--hours", type=float, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    for path in make_figures(args.start, hours=args.hours,
                             output_dir=args.output_dir, prefix=args.prefix):
        print(path)


if __name__ == "__main__":
    main()
