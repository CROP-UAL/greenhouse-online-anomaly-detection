"""Immutable names and validation for the shared physical experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data_contract" / "snapshots"
SPLITS = ("train", "validation", "test")
FILES = {name: SNAPSHOTS / f"{name}_injected_physical.parquet" for name in SPLITS}
ROWS = {"train": 172_873, "validation": 28_800, "test": 31_680}
SENSORS = ("PCO2EXT", "PHEXT", "PRAD", "PRGINT", "PTEXT", "PVV", "XCO2I", "XHINV", "XTINV")
ANOMALIES = ("Contextual", "Noise", "Correlation deviation", "Missing data", "Out-of-range", "Stuck sensor")
CLASSES = ("Normal",) + ANOMALIES
SOURCE_TO_CANONICAL = {
    "Contextual": "Contextual",
    "Ruido": "Noise",
    "Desviacion de Correlacion": "Correlation deviation",
    "Datos Faltantes": "Missing data",
    "Valores Fuera de Rango": "Out-of-range",
    "Sensor Atascado": "Stuck sensor",
}
EXPECTED_SHA256 = {
    "train": "64f43cd15e121c8a147565aa691ebb3d6bf2d371ffd86824f2ab626dcba5e1a2",
    "validation": "044d3d23f7a06b98afed94cee741d8f206beb5a8319c70e1a2c56859689a7107",
    "test": "f6ad833659d0d379dc3a1677a8a5c0f9478536dadbb5b80bc6eadf82fa60d4f3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_label(value: object) -> str:
    if pd.isna(value) or value == "Normal":
        return "Normal"
    if value in ANOMALIES:
        return str(value)
    if value in SOURCE_TO_CANONICAL:
        return SOURCE_TO_CANONICAL[str(value)]
    raise ValueError(f"Unknown source label: {value!r}")


def manifest_id(hashes: dict[str, str]) -> str:
    payload = json.dumps({k: hashes[k] for k in SPLITS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_split(name: str, *, verify_hash: bool = True) -> pd.DataFrame:
    if name not in SPLITS:
        raise ValueError(f"Unknown split: {name}")
    path = FILES[name]
    if verify_hash and sha256(path) != EXPECTED_SHA256[name]:
        raise ValueError(f"Frozen input hash differs for {name}")
    frame = pd.read_parquet(path)
    required = {"FECHA", "tipo_anomalia", *SENSORS}
    if required - set(frame.columns):
        raise ValueError(f"{name}: missing {sorted(required - set(frame.columns))}")
    if len(frame) != ROWS[name]:
        raise ValueError(f"{name}: expected {ROWS[name]} rows; got {len(frame)}")
    dates = frame["FECHA"]
    if not pd.api.types.is_datetime64_ns_dtype(dates) or dates.isna().any() or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError(f"{name}: FECHA must be unique, ordered, naive datetime64[ns]")
    frame["tipo_canonico"] = frame.tipo_anomalia.map(canonical_label)
    return frame


def validate_splits() -> dict[str, object]:
    details: dict[str, object] = {}
    previous = None
    for name in SPLITS:
        frame = load_split(name)
        if previous is not None and frame.FECHA.iloc[0] <= previous:
            raise ValueError("Chronological splits overlap")
        previous = frame.FECHA.iloc[-1]
        details[name] = {"rows": len(frame), "first": str(frame.FECHA.iloc[0]), "last": str(previous),
                         "sha256": EXPECTED_SHA256[name], "labels": frame.tipo_canonico.value_counts().to_dict()}
    return {"schema_version": "physical-agroconnect-v1", "data_source": "Agroconnect",
            "manifest_id": manifest_id(EXPECTED_SHA256), "splits": details,
            "sensors": list(SENSORS), "classes": list(CLASSES), "timestamp_policy": "physical FECHA, original jitter retained"}
