"""Copy ignored physical inputs and frozen models from a local source checkout.

Only metadata and this code are versioned. Source files are never changed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from shared.contract import ROOT, sha256, validate_splits
from shared.injection.replay import create_plan

INVENTORY = ROOT / "data_contract" / "manifests" / "imported_files.json"
LOCAL_PREFIXES = ("data_contract/snapshots/", "approaches/lightgbm/v8_online_2meses/reference_run/")


def hydrate(lightgbm_source: Path) -> dict:
    source_root = Path(lightgbm_source).resolve()
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    entries = [entry for entry in inventory["files"]
               if entry["repository"] == "lightgbm"
               and entry["destination"].startswith(LOCAL_PREFIXES)]
    if not entries:
        raise ValueError("Import inventory has no local physical/model inputs")
    missing = [entry["source"] for entry in entries if not (source_root / entry["source"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Source checkout lacks required ignored inputs: {missing}")
    for entry in entries:
        source = source_root / entry["source"]
        destination = ROOT / entry["destination"]
        if sha256(source) != entry["source_sha256"]:
            raise ValueError(f"Source hash changed: {source}")
        if destination.exists():
            if sha256(destination) != entry["destination_sha256"]:
                raise ValueError(f"Local input has unexpected hash: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != entry["destination_sha256"]:
            raise ValueError(f"Copied input hash changed: {destination}")
    physical = validate_splits()
    plan = create_plan()
    return {"copied_or_verified": len(entries), "data_manifest_id": physical["manifest_id"],
            "local_replay_plan_sha256": plan["plan_sha256"]}
