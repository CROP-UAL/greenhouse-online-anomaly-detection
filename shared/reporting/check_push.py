"""Reject pushes that could publish source histories, sensor data or artifacts.

Uses only the Python standard library and Git. Run ``--audit-ref main`` to
inspect a ref locally; the versioned pre-push hook supplies updates on stdin.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import PurePosixPath


BLOCKED_SUFFIXES = {
    ".csv", ".parquet", ".pkl", ".pickle", ".joblib", ".h5", ".hdf5",
    ".keras", ".onnx", ".tflite", ".npy", ".npz", ".png", ".jpg",
    ".jpeg", ".pdf", ".zip", ".7z", ".tar", ".gz",
}
BLOCKED_DIRECTORIES = {"dataset", "datasets", "results", "figures", "reports", "artifacts", "runs", "snapshots"}
ALLOWED_PLACEHOLDERS = {"results/README.md"}
BLOCKED_REF_PREFIXES = ("refs/source/", "refs/reconstructed/", "refs/reconstructed_safe/", "refs/archive/")
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
)
ZERO = "0" * 40


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def commits_to_audit(local_oid: str, remote_oid: str | None) -> list[str]:
    args = ["rev-list", local_oid]
    if remote_oid and remote_oid != ZERO:
        known = subprocess.run(["git", "cat-file", "-e", remote_oid], capture_output=True).returncode == 0
        if known:
            args.extend(["--not", remote_oid])
    return git(*args).decode("ascii").splitlines()


def audit_commits(commits: list[str]) -> tuple[int, int]:
    checked_blobs: set[str] = set()
    checked_notebooks: set[str] = set()
    paths_checked = 0
    for commit in commits:
        entries = git("ls-tree", "-r", "-z", commit).split(b"\0")
        for entry in entries:
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, blob_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8", errors="replace")
            parts = PurePosixPath(path).parts
            suffix = PurePosixPath(path).suffix.lower()
            if suffix in BLOCKED_SUFFIXES or (path not in ALLOWED_PLACEHOLDERS and
                    any(part.lower() in BLOCKED_DIRECTORIES for part in parts[:-1])):
                raise ValueError(f"blocked data or artifact path in {commit[:12]}: {path}")
            if kind != "blob" or mode == "120000":
                raise ValueError(f"unsupported Git object in {commit[:12]}: {path}")
            paths_checked += 1
            sha = blob_id
            if sha in checked_blobs and (suffix != ".ipynb" or sha in checked_notebooks):
                continue
            checked_blobs.add(sha)
            size = int(git("cat-file", "-s", sha))
            if size > 5_000_000:
                raise ValueError(f"blob over 5 MB requires review in {commit[:12]}: {path}")
            content = git("cat-file", "blob", sha)
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                raise ValueError(f"possible credential in {commit[:12]}: {path}")
            if suffix == ".ipynb":
                notebook = json.loads(content)
                for cell in notebook.get("cells", []):
                    if cell.get("outputs") or cell.get("execution_count") is not None or cell.get("attachments"):
                        raise ValueError(f"notebook contains outputs in {commit[:12]}: {path}")
                checked_notebooks.add(sha)
    return paths_checked, len(checked_blobs)


def audit_ref(ref: str, remote_oid: str | None = None) -> None:
    local_oid = git("rev-parse", "--verify", ref).decode("ascii").strip()
    for source in ("refs/source/lightgbm", "refs/source/lstm"):
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", source], capture_output=True).returncode == 0:
            if subprocess.run(["git", "merge-base", "--is-ancestor", source, local_oid], capture_output=True).returncode == 0:
                raise ValueError(f"{ref} includes an unfiltered source history")
    commits = commits_to_audit(local_oid, remote_oid)
    paths, blobs = audit_commits(commits)
    print(f"OK {ref}: {len(commits)} commits, {paths} paths, {blobs} distinct blobs checked")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-ref", help="Audit an entire local ref without pushing")
    parser.add_argument("--pre-push", action="store_true", help="Read Git pre-push update lines from stdin")
    args = parser.parse_args()
    try:
        if args.audit_ref:
            audit_ref(args.audit_ref)
        elif args.pre_push:
            updates = [line.split() for line in sys.stdin if line.strip()]
            if not updates:
                return 0
            for local_ref, local_oid, remote_ref, remote_oid in updates:
                if not remote_ref.startswith("refs/heads/") or any(local_ref.startswith(prefix) for prefix in BLOCKED_REF_PREFIXES):
                    raise ValueError(f"only reviewed branch pushes are allowed: {local_ref} -> {remote_ref}")
                if local_oid == ZERO:
                    raise ValueError(f"branch deletion requires a separate review: {remote_ref}")
                if remote_oid != ZERO and subprocess.run(
                    ["git", "cat-file", "-e", remote_oid], capture_output=True
                ).returncode == 0 and subprocess.run(
                    ["git", "merge-base", "--is-ancestor", remote_oid, local_oid], capture_output=True
                ).returncode != 0:
                    raise ValueError(f"non-fast-forward update requires a separate review: {remote_ref}")
                audit_ref(local_oid, remote_oid)
        else:
            parser.error("use --audit-ref or --pre-push")
    except (ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"PUSH BLOCKED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
