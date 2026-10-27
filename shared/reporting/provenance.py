"""Inventory every imported source artifact with both source and target hashes."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from shared.contract import ROOT, sha256

PARENT = ROOT.parent
LIGHTGBM = PARENT / "Clasificaci-n-y-correcci-n"
LSTM = PARENT / "CIESOL_PROJECT"
OUTPUT = ROOT / "data_contract" / "manifests" / "imported_files.json"
ATTRIBUTION_CORRECTIONS = {
    "approaches/lstm/README_source.md",
    "approaches/lstm/src/feature_engineering.py",
    "approaches/lstm/src/config_injection.py",
    "approaches/lstm/src/preprocessing.py",
    "approaches/lstm/notebooks/04_gap_imputation_and_splitting.ipynb",
    "approaches/lstm/notebooks/10_model1_binary_lstm_training.ipynb",
}


def _source_path(relative: Path) -> tuple[Path, str, str, str]:
    text = relative.as_posix()
    if text.startswith("data_contract/snapshots/"):
        name = relative.name
        source = LIGHTGBM / "Dataset" / ("Datos2MesesLimpios" if name.endswith(".csv") else "MismoTest") / name
        return source, "lightgbm", "83399ba380124f9bc0754990f2afeb195df94d29", "copied"
    if text.startswith("approaches/lightgbm/v7_reference/"):
        name = relative.name
        section = "Antigua" if name.endswith(".ipynb") else "Test_2_anos"
        source = LIGHTGBM / "v7_palmerillas_vent_null" / section / name
        return source, "lightgbm", "83399ba380124f9bc0754990f2afeb195df94d29", "notebook outputs removed" if name.endswith(".ipynb") else "copied"
    if text.startswith("approaches/lightgbm/v8_online_2meses/"):
        suffix = Path(*relative.parts[3:])
        if suffix.parts[0] == "reference_run":
            suffix = Path("lstm_common_lightgbm/data/runs/20261021T120316012126Z") / Path(*suffix.parts[1:])
        source = LIGHTGBM / "v8_online_2meses" / suffix
        return source, "lightgbm", "83399ba380124f9bc0754990f2afeb195df94d29", "copied"
    if text.startswith("approaches/lstm/"):
        suffix = Path(*relative.parts[2:])
        if suffix.name == "README_source.md":
            suffix = Path("README.md")
        source = LSTM / suffix
        transform = ("notebook outputs removed" if source.suffix == ".ipynb"
                     else "relative package import adapted" if suffix.as_posix() == "src/preprocessing.py"
                     else "copied")
        return source, "lstm", "f3a5f500c7a4b45e9485653d79aa6b67857657f4", transform
    raise ValueError(relative)


def create_inventory() -> dict:
    candidates = list((ROOT / "data_contract" / "snapshots").glob("*"))
    candidates += list((ROOT / "approaches" / "lightgbm" / "v7_reference").rglob("*"))
    candidates += list((ROOT / "approaches" / "lightgbm" / "v8_online_2meses").rglob("*"))
    candidates += list((ROOT / "approaches" / "lstm" / "src").rglob("*"))
    candidates += list((ROOT / "approaches" / "lstm" / "notebooks").rglob("*"))
    candidates += [ROOT / "approaches" / "lstm" / "README_source.md"]
    files = []
    for path in sorted({p for p in candidates if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}):
        relative = path.relative_to(ROOT)
        source, repository, commit, transform = _source_path(relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        if relative.as_posix() in ATTRIBUTION_CORRECTIONS:
            transform += "; Agroconnect data attribution corrected"
        files.append({"destination": relative.as_posix(), "source": str(source.relative_to(LIGHTGBM if repository == "lightgbm" else LSTM)).replace("\\", "/"),
                      "repository": repository, "commit": commit, "source_sha256": sha256(source),
                      "destination_sha256": sha256(path), "bytes": path.stat().st_size, "transform": transform})
    result = {"schema_version": "source-import-v1", "import_date": date.today().isoformat(),
              "sources": {"lightgbm": "https://github.com/vfd003-ual/Clasificaci-n-y-correcci-n.git",
                          "lstm": "https://github.com/helqadiri03/CIESOL_PROJECT.git"},
              "files": files}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
