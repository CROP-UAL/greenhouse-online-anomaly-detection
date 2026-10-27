"""Versioned, hash-verified optimized artifact manifests."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "binary-edge-manifest-v1"

def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_manifest(directory: str | Path, payload: dict[str, Any]) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": MANIFEST_SCHEMA, **payload}
    artifacts = data.get("artifacts", {})
    data["hashes"] = {name: file_sha256(root/relative)
                      for name, relative in artifacts.items()
                      if (root/relative).is_file()}
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    data["manifest_hash"] = sha256(canonical).hexdigest()
    path = root/"manifest.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def load_manifest(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Unsupported artifact manifest schema")
    supplied_hash = data.pop("manifest_hash")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    if sha256(canonical).hexdigest() != supplied_hash:
        raise ValueError("Manifest hash mismatch")
    data["manifest_hash"] = supplied_hash
    if verify:
        for name, expected in data.get("hashes", {}).items():
            relative = data["artifacts"][name]
            if file_sha256(manifest_path.parent/relative) != expected:
                raise ValueError(f"Artifact hash mismatch: {name}")
    return data
