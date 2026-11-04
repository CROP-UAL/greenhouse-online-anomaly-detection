"""Reconstruct attributable source history without original content or account emails.

The new commits preserve parent relationships, author display names and dates.
Their trees contain only generic categories of work. Source paths, subjects,
emails, URLs, original hashes and file contents are kept out of the new DAG.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from shared.contract import ROOT


SOURCES = {"lightgbm": "refs/source/lightgbm", "lstm": "refs/source/lstm"}
EXCLUDED_PARTS = {
    "data", "dataset", "results", "reports", "figures", "plots", "models",
    "artifacts", "runs", "env", "venv", ".venv", "__pycache__", "outputs",
}
CODE_SUFFIXES = {".py", ".ipynb", ".md", ".rst", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
MARKER = re.compile(r"^history_(lightgbm|lstm)_[0-9]{4}\.json$")


def _git(*args: str, input_bytes: bytes | None = None, env: dict[str, str] | None = None) -> bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, input=input_bytes, capture_output=True,
                            check=True, env=env)
    return result.stdout.strip()


def safe_code_path(path: str) -> bool:
    """Expose only code and documentation path names, never their contents."""
    parts = Path(path).parts
    if not parts or any(part.lower() in EXCLUDED_PARTS for part in parts):
        return False
    name = parts[-1]
    return (Path(name).suffix.lower() in CODE_SUFFIXES or name == ".gitignore"
            or name in {"Dockerfile", "Makefile", "LICENSE"}
            or name.startswith("requirements") and name.endswith(".txt"))


@dataclass(frozen=True)
class OriginalCommit:
    sha: str
    parents: tuple[str, ...]
    author: str
    author_date: str
    work_area: str
    work_categories: tuple[str, ...]


def work_area(subject: str) -> str:
    """Classify a source subject into a fixed, non-sensitive activity label."""
    title = subject.casefold()
    rules = (
        (("jetson", "latencia", "hardware"), "edge benchmarking"),
        (("multiclase", "multiclass", "6 anomaly", "6-classes"), "multiclass classification"),
        (("inyeccion", "conjunto de datos", "date period"), "experimental data preparation"),
        (("falsas alarmas", "umbrales", "precision", "recall", "metricas", "resultados", "results"), "model evaluation"),
        (("binario", "binary"), "binary detection"),
        (("streaming", "online"), "online inference"),
        (("requisit", "dependenc", "entorno virtual", "docker", "kernel", "gitignore"), "development environment"),
        (("figure", "image", "grafica"), "visualization"),
        (("readme", "documentacion"), "documentation"),
        (("rename", "renamed", "files", "estructura"), "repository organization"),
    )
    for terms, label in rules:
        if any(term in title for term in terms):
            return label
    return "implementation and maintenance"


def inspect_source(name: str) -> list[OriginalCommit]:
    ref = SOURCES[name]
    revisions = _git("rev-list", "--reverse", "--topo-order", ref).decode("ascii").splitlines()
    commits = []
    seen = set()
    for sha in revisions:
        fields = _git("log", "-1", "--format=%H%x00%P%x00%an%x00%aI%x00%s", sha).decode("utf-8").split("\x00")
        if len(fields) != 5 or fields[0] != sha:
            raise ValueError(f"Could not parse source commit {sha}")
        parents = tuple(fields[1].split())
        if any(parent not in seen for parent in parents):
            raise ValueError(f"Source history is not topologically ordered at {sha}")
        paths = {p.decode("utf-8") for p in _git("diff-tree", "--root", "-m", "--no-commit-id",
                                                "--name-only", "-r", "-z", sha).split(b"\x00") if p}
        selected = (path for path in paths if safe_code_path(path))
        categories = set()
        for path in selected:
            suffix = Path(path).suffix.lower()
            if suffix == ".py":
                categories.add("Python code")
            elif suffix == ".ipynb":
                categories.add("notebooks")
            elif suffix in {".md", ".rst"}:
                categories.add("documentation")
            else:
                categories.add("configuration")
        commits.append(OriginalCommit(sha, parents, fields[2], fields[3], work_area(fields[4]),
                                      tuple(sorted(categories))))
        seen.add(sha)
    return commits


def _marker_blob(name: str, ordinal: int, commit: OriginalCommit) -> str:
    record = {"kind": "sanitized-source-contribution", "project": name,
              "sequence": ordinal, "author_display_name": commit.author,
              "author_date": commit.author_date, "work_area": commit.work_area,
              "work_categories": commit.work_categories,
              "content_policy": "generic attribution only; no source content or account email"}
    payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return _git("hash-object", "-w", "--stdin", input_bytes=payload).decode("ascii")


def _reachable_objects(ref: str) -> set[str]:
    return {line.split(b" ", 1)[0].decode("ascii")
            for line in _git("rev-list", "--objects", ref).splitlines()}


def build_source(name: str) -> dict[str, object]:
    ref = SOURCES[name]
    commits = inspect_source(name)
    rewritten: dict[str, str] = {}
    snapshots: dict[str, dict[str, str]] = {}
    for ordinal, commit in enumerate(commits, 1):
        files: dict[str, str] = {}
        for parent in commit.parents:
            files.update(snapshots[parent])
        marker_name = f"history_{name}_{ordinal:04d}.json"
        files[marker_name] = _marker_blob(name, ordinal, commit)
        tree_input = "".join(f"100644 blob {blob}\t{path}\n" for path, blob in sorted(files.items()))
        tree = _git("mktree", input_bytes=tree_input.encode("ascii")).decode("ascii")
        categories = ", ".join(commit.work_categories) or "historical contribution"
        message = (f"[sanitized {name}] {commit.work_area} ({ordinal:04d})\n\n"
                   f"Work categories: {categories}.\n"
                   "Reconstructed attribution; original content and account metadata excluded.\n")
        environment = os.environ.copy()
        environment.update({"GIT_AUTHOR_NAME": commit.author, "GIT_AUTHOR_EMAIL": "contributor@history.invalid",
                            "GIT_AUTHOR_DATE": commit.author_date,
                            "GIT_COMMITTER_NAME": "Source History Reconstruction",
                            "GIT_COMMITTER_EMAIL": "history-reconstruction@local.invalid",
                            "GIT_COMMITTER_DATE": commit.author_date})
        args = ["commit-tree", tree]
        for parent in commit.parents:
            args.extend(["-p", rewritten[parent]])
        new_sha = _git(*args, input_bytes=message.encode("utf-8"), env=environment).decode("ascii")
        rewritten[commit.sha] = new_sha
        snapshots[commit.sha] = files
    tip = rewritten[commits[-1].sha]
    expected = {f"history_{name}_{ordinal:04d}.json" for ordinal in range(1, len(commits) + 1)}
    actual = set(_git("ls-tree", "-r", "--name-only", tip).decode("utf-8").splitlines())
    if actual != expected or any(not MARKER.fullmatch(path) for path in actual):
        raise ValueError(f"Unexpected files in reconstructed {name} history")
    if int(_git("rev-list", "--count", tip)) != len(commits):
        raise ValueError(f"Reconstructed {name} history has the wrong number of commits")
    if _reachable_objects(ref) & _reachable_objects(tip):
        raise ValueError(f"Reconstructed {name} history reaches an original source object")
    new_ref = f"refs/reconstructed_safe_v2/{name}"
    current = _git("rev-parse", "--verify", "--quiet", new_ref).decode("ascii") if subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", new_ref], cwd=ROOT, capture_output=True).returncode == 0 else None
    if current is not None and current != tip:
        raise ValueError(f"Ref {new_ref} already points to a different history")
    _git("update-ref", new_ref, tip)
    return {"source": name, "reconstructed_tip": tip,
            "commit_count": len(commits), "metadata_files": len(actual)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Write reconstructed objects and refs")
    args = parser.parse_args()
    if args.build:
        print(json.dumps([build_source(name) for name in SOURCES], indent=2))
    else:
        print(json.dumps([{ "source": name, "commits": len(commits),
                            "authors": sorted({commit.author for commit in commits})}
                          for name in SOURCES for commits in [inspect_source(name)]], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
