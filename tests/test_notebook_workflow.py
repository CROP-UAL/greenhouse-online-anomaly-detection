"""Keep the interactive workflow executable and free of exported results."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from shared.contract import ROOT
from shared.reporting.edge_bundle import _transfer_path, verify


def test_controlled_notebooks_have_clean_executable_cells() -> None:
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 3
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None
                compile("".join(cell["source"]), str(path), "exec")


@pytest.mark.parametrize("notebook_name", [
    "01_train_and_export.ipynb", "02_compare_quality.ipynb", "03_jetson_benchmark.ipynb"
])
def test_notebook_imports_shared_when_started_from_notebooks_directory(notebook_name: str) -> None:
    path = ROOT / "notebooks" / notebook_name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    bootstrap = "".join(first_code["source"]).split("from shared.contract import", 1)[0]
    script = bootstrap + "from shared.contract import ROOT\nassert Path.cwd().resolve() == ROOT\n"
    completed = subprocess.run([sys.executable, "-c", script], cwd=path.parent,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_edge_bundle_rejects_escaping_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe bundle path"):
        _transfer_path("experiments/runs/edge_reference/../../secret.csv")
    (tmp_path / "bundle_manifest.json").write_text(json.dumps({"files": [
        {"path": "../../secret.csv", "bytes": 0, "sha256": "0" * 64}
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe bundle path"):
        verify(tmp_path)


def test_edge_bundle_detects_changed_file(tmp_path: Path) -> None:
    relative = "experiments/runs/edge_reference/paired_test.json"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"verified")
    plan = {"files": [{"path": relative, "bytes": 8,
                       "sha256": hashlib.sha256(b"verified").hexdigest()}],
            "data_manifest_id": "test", "source_commit": "test"}
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(plan), encoding="utf-8")
    assert verify(tmp_path)["files_verified"] == 1
    target.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Bundle file changed"):
        verify(tmp_path)
