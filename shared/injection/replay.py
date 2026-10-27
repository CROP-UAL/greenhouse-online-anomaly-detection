"""Replay the exact imported injection rows on rebuilt physical data.

This verifies and reproduces the frozen experimental input. It does not claim
to rerun the historical random event sampler from notebook 05.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from shared.contract import ROOT, SPLITS, ROWS, load_split, sha256
from shared.data_preparation.verify import rebuild_preinjection

PLAN = ROOT / "data_contract" / "manifests" / "injection_replay.parquet"
PLAN_MANIFEST = ROOT / "data_contract" / "manifests" / "injection_replay.json"
EXTRA = ("sensor_anomalia", "anomalia_id", "tipo_anomalia", "fuente_anomalia",
         "valor_original_anomalia", "valor_inyectado_anomalia")


def create_plan() -> dict:
    frozen = pd.concat([load_split(name) for name in SPLITS], ignore_index=True).drop(columns=["tipo_canonico"])
    targets = np.flatnonzero(frozen.tipo_anomalia.notna().to_numpy())
    plan = frozen.iloc[targets].copy()
    plan.insert(0, "row_index", targets.astype(np.int64))
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    plan.to_parquet(PLAN, index=False)
    metadata = {"schema_version": "frozen-injection-replay-v1", "source": "CIESOL 05_synthetic_anomaly_injection.ipynb",
                "historical_base_seed": 42, "replay_uses_random_number_generator": False,
                "row_count": len(plan), "split_rows": ROWS, "plan_sha256": sha256(PLAN),
                "meaning": "Full anomalous rows applied to independently rebuilt pre-injection physical data"}
    PLAN_MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def replay(*, verify: bool = True) -> tuple[dict[str, pd.DataFrame], dict]:
    metadata = json.loads(PLAN_MANIFEST.read_text(encoding="utf-8"))
    if sha256(PLAN) != metadata["plan_sha256"]:
        raise ValueError("Injection replay plan hash changed")
    base, lineage = rebuild_preinjection()
    frozen = pd.concat([load_split(name) for name in SPLITS], ignore_index=True).drop(columns=["tipo_canonico"])
    plan = pd.read_parquet(PLAN)
    if len(plan) != metadata["row_count"] or not plan.row_index.is_unique:
        raise ValueError("Injection replay plan has duplicate or missing row positions")
    if (plan.row_index < 0).any() or (plan.row_index >= len(base)).any():
        raise ValueError("Injection replay plan lies outside physical timeline")
    for column in EXTRA:
        if column.startswith("valor_"):
            base[column] = np.nan
        else:
            base[column] = None
    base = base.loc[:, frozen.columns]
    base.iloc[plan.row_index.to_numpy()] = plan.drop(columns=["row_index"]).to_numpy()
    if verify:
        assert_frame_equal(base.reset_index(drop=True), frozen.reset_index(drop=True), check_dtype=False)
    offsets = np.cumsum([0] + [ROWS[name] for name in SPLITS])
    splits = {name: base.iloc[offsets[i]:offsets[i + 1]].reset_index(drop=True)
              for i, name in enumerate(SPLITS)}
    return splits, {"replayed_rows": len(base), "injected_rows": len(plan), "plan_sha256": metadata["plan_sha256"],
                    "lineage": lineage, "exact_frozen_parity": bool(verify)}
