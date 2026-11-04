"""Prepare a local, hash-checked transfer bundle for Jetson measurements.

No data or model file is added to Git. The default invocation prints the
inventory; --build copies it to a new destination chosen by the user.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from shared.contract import FILES, ROOT, SPLITS, sha256, validate_splits


REFERENCE = ROOT / "approaches" / "lightgbm" / "v8_online_2meses" / "reference_run"
REFERENCE_NAMES = (
    "binary_calibration.json", "binary_manifest.json", "binary_model.txt",
    "binary_validation.json", "clock_audit.json", "manifest.json",
    "multiclass_model.txt", "multiclass_validation.json",
)
TRANSFER_PREFIXES = (
    "data_contract/snapshots/",
    "approaches/lightgbm/v8_online_2meses/reference_run/",
    "experiments/runs/lightgbm_night_v1/",
    "experiments/runs/lstm_causal_v1/",
    "experiments/runs/lstm_causal_v2/",
    "experiments/runs/edge_reference/",
)


def _transfer_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not any(value.startswith(prefix) for prefix in TRANSFER_PREFIXES):
        raise ValueError(f"Unsafe bundle path: {value}")
    return path


def _inside_root(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Run must be inside the repository: {path}") from exc


def inventory(lightgbm_run: Path, lstm_run: Path) -> dict:
    contract = validate_splits()
    lstm_parent = Path(lstm_run).parent.name
    if lstm_parent not in ("lstm_causal_v1", "lstm_causal_v2"):
        raise ValueError("LSTM run must be under lstm_causal_v1 or lstm_causal_v2")
    expected = {"lightgbm": (lightgbm_run, "lightgbm_night_v1", ("model1.txt", "model2.txt")),
                "lstm": (lstm_run, lstm_parent, ("model1.keras", "model2.keras"))}
    files: dict[str, Path] = {}
    for split in SPLITS:
        files[str(FILES[split].relative_to(ROOT)).replace("\\", "/")] = FILES[split]
    for name in REFERENCE_NAMES:
        path = REFERENCE / name
        files[str(path.relative_to(ROOT)).replace("\\", "/")] = path
    run_ids = {}
    for approach, (run, parent, names) in expected.items():
        relative = _inside_root(Path(run))
        if relative.parent.as_posix() != f"experiments/runs/{parent}":
            raise ValueError(f"{approach} run is outside experiments/runs/{parent}")
        manifest_path = ROOT / relative / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["data_manifest_id"] != contract["manifest_id"]:
            raise ValueError(f"{approach} run uses a different data manifest")
        if approach == "lstm" and manifest.get("schema_version") != lstm_parent.replace("_", "-"):
            raise ValueError("LSTM run directory and schema version disagree")
        for index, name in enumerate(names, 1):
            path = ROOT / relative / name
            if sha256(path) != manifest[f"model{index}_sha256"]:
                raise ValueError(f"{approach} {name} differs from its run manifest")
            files[(relative / name).as_posix()] = path
        files[(relative / "manifest.json").as_posix()] = manifest_path
        run_ids[approach] = relative.as_posix()
    traces = {}
    for approach, name in (("lightgbm", "lightgbm_night_test.parquet"), ("lstm", "lstm_test.parquet")):
        source = ROOT / "results" / "traces" / name
        if not source.is_file():
            raise FileNotFoundError(f"Run the paired quality notebook first: {source}")
        traces[approach] = pd.read_parquet(source, columns=["timestamp", "split", "ground_truth_binary",
                                                             "ground_truth_type", "episode_id", "data_manifest_id",
                                                             "model_version"])
        expected_version = ("lightgbm-night-context-v1-" if approach == "lightgbm" else f"lstm-causal-{lstm_parent[-2:]}-") + Path(run_ids[approach]).name
        if len(traces[approach]) != 31680 or set(traces[approach].model_version) != {expected_version}:
            raise ValueError(f"{approach} trace does not belong to the selected run")
        files[f"experiments/runs/edge_reference/{name}"] = source
    paired_columns = ["timestamp", "split", "ground_truth_binary", "ground_truth_type", "episode_id", "data_manifest_id"]
    if not traces["lightgbm"][paired_columns].equals(traces["lstm"][paired_columns]):
        raise ValueError("LightGBM and LSTM traces are not aligned")
    report_path = ROOT / "results" / "metrics" / "paired_test.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["paired_test_rows"] != 31680 or any(
        report[approach]["data_manifest_id"] != contract["manifest_id"] or
        report[approach]["model_version"] != traces[approach].model_version.iloc[0]
        for approach in expected
    ):
        raise ValueError("Paired quality report is missing or uses another data manifest")
    files["experiments/runs/edge_reference/paired_test.json"] = report_path
    records = []
    for destination, source in sorted(files.items()):
        records.append({"path": destination, "sha256": sha256(source), "bytes": source.stat().st_size})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    return {"schema_version": "edge-transfer-v1", "target_hardware": "NVIDIA Jetson Orin Nano",
            "source_commit": revision, "data_manifest_id": contract["manifest_id"],
            "runs": run_ids, "files": records, "total_bytes": sum(row["bytes"] for row in records)}


def build(destination: Path, lightgbm_run: Path, lstm_run: Path, *, copy: bool = False) -> dict:
    plan = inventory(lightgbm_run, lstm_run)
    if not copy:
        return plan
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Bundle destination already exists: {destination}")
    try:
        inside = destination.relative_to(ROOT.resolve())
    except ValueError:
        inside = None
    if inside is not None and inside.parts[:2] != ("experiments", "runs"):
        raise ValueError("Inside the repository, bundles must be placed under ignored experiments/runs/")
    destination.mkdir(parents=True)
    for record in plan["files"]:
        source_path = record["path"]
        if source_path.startswith("experiments/runs/edge_reference/"):
            name = Path(source_path).name
            source = ROOT / ("results/metrics" if name == "paired_test.json" else "results/traces") / name
        else:
            source = ROOT / source_path
        target = destination / source_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != record["sha256"]:
            raise ValueError(f"Copy changed: {source_path}")
    (destination / "bundle_manifest.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def verify(destination: Path) -> dict:
    destination = Path(destination)
    plan = json.loads((destination / "bundle_manifest.json").read_text(encoding="utf-8"))
    for record in plan["files"]:
        path = destination / _transfer_path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Bundle file changed: {record['path']}")
    return {"files_verified": len(plan["files"]), "data_manifest_id": plan["data_manifest_id"],
            "source_commit": plan["source_commit"]}


def install(destination: Path) -> dict:
    checked = verify(destination)
    plan = json.loads((Path(destination) / "bundle_manifest.json").read_text(encoding="utf-8"))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    if revision != plan["source_commit"]:
        raise ValueError("Jetson checkout must use the same commit as the verified training run")
    for record in plan["files"]:
        relative = _transfer_path(record["path"])
        source = Path(destination) / relative
        target = ROOT / relative
        if not target.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Bundle target escapes repository: {target}")
        if target.exists():
            if sha256(target) != record["sha256"]:
                raise ValueError(f"Local file differs; refusing to overwrite: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != record["sha256"]:
            raise ValueError(f"Installed file changed: {target}")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightgbm-run", type=Path)
    parser.add_argument("--lstm-run", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        result = install(args.destination)
    elif args.verify:
        result = verify(args.destination)
    else:
        if args.lightgbm_run is None or args.lstm_run is None:
            parser.error("--lightgbm-run and --lstm-run are required unless --verify is used")
        result = build(args.destination, args.lightgbm_run, args.lstm_run, copy=args.build)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
