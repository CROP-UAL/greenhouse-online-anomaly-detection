"""Verify the source CSV and the timestamp sequence of frozen splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shared.contract import ROOT, SPLITS, load_split, sha256

RAW = ROOT / "data_contract" / "snapshots" / "sensor_data_raw2.csv"


def audit_raw_lineage() -> dict[str, object]:
    # Match the source notebook's parsing and keep-first timestamp deduplication.
    raw = pd.read_csv(RAW, sep=";", low_memory=False)
    dates = pd.to_datetime(raw["Fecha"], dayfirst=True, errors="raise")
    ordered = raw.assign(FECHA=dates).sort_values("FECHA", kind="stable").reset_index(drop=True)
    clean = ordered.drop_duplicates(subset=["FECHA"], keep="first").reset_index(drop=True)
    frozen = pd.concat([load_split(name)[["FECHA", "timestamp_was_missing"]] for name in SPLITS], ignore_index=True)
    original = frozen.loc[~frozen.timestamp_was_missing, "FECHA"].reset_index(drop=True)
    matches = original.equals(clean.FECHA.reset_index(drop=True))
    report = {"raw_sha256": sha256(RAW), "raw_rows": len(raw), "duplicates_removed": len(raw) - len(clean),
              "original_rows_in_splits": len(original), "reconstructed_rows": int(frozen.timestamp_was_missing.sum()),
              "prepared_rows": len(frozen), "original_timestamp_sequence_matches": bool(matches)}
    if (report["raw_rows"], report["duplicates_removed"], report["reconstructed_rows"], report["prepared_rows"]) != (233381, 48, 20, 233353):
        raise ValueError(f"Unexpected raw lineage counts: {report}")
    if not matches:
        raise ValueError("Original timestamps in frozen splits differ from deduplicated CSV")
    return report


def rebuild_preinjection() -> tuple[pd.DataFrame, dict[str, object]]:
    """Replay the imported LSTM gap procedure without writing source files."""
    from approaches.lstm.src.data_loader import load_single_file
    from approaches.lstm.src.preprocessing import (PreprocessConfig, analyze_time_gaps, create_gap_grid,
                                   interpolate_short_gaps, handle_medium_gaps, handle_long_gaps,
                                   update_sequence_safety, check_physical_ranges)

    original = load_single_file(RAW)
    cleaned = original.drop_duplicates(subset=["FECHA"], keep="first").reset_index(drop=True)
    cfg = PreprocessConfig()
    gaps = analyze_time_gaps(cleaned, cfg=cfg)["gap_report"]
    grid = create_gap_grid(cleaned, gaps, expected_seconds=cfg.expected_seconds,
                           categories=["short_gap", "medium_gap", "long_gap"])
    prepared = interpolate_short_gaps(grid, gap_report=gaps, cfg=cfg)
    prepared = handle_medium_gaps(prepared, gaps)
    prepared = handle_long_gaps(prepared, gaps)
    prepared = update_sequence_safety(prepared)
    prepared, _ = check_physical_ranges(prepared)
    frozen = pd.concat([load_split(name) for name in SPLITS], ignore_index=True)
    if not prepared.FECHA.reset_index(drop=True).equals(frozen.FECHA.reset_index(drop=True)):
        raise ValueError("Rebuilt timestamps differ from frozen physical splits")
    if not prepared.timestamp_was_missing.reset_index(drop=True).equals(frozen.timestamp_was_missing.reset_index(drop=True)):
        raise ValueError("Reconstructed timestamp masks differ")
    normal = frozen.tipo_anomalia.isna().to_numpy()
    sensors = [column for column in original.columns if column != "FECHA"]
    differences = {}
    for sensor in sensors:
        left = prepared.loc[normal, sensor].to_numpy(dtype=float)
        right = frozen.loc[normal, sensor].to_numpy(dtype=float)
        bad = ~np.isclose(left, right, equal_nan=True, rtol=0, atol=1e-10)
        differences[sensor] = int(bad.sum())
    report = {"raw_rows": len(original), "deduplicated_rows": len(cleaned), "reconstructed_rows": len(prepared) - len(cleaned),
              "prepared_rows": len(prepared), "gap_episodes": int(gaps.gap_category.isin(["short_gap", "medium_gap", "long_gap"]).sum()),
              "normal_sensor_mismatches": differences}
    if any(differences.values()):
        raise ValueError(f"Normal physical values differ: {report}")
    return prepared, report
