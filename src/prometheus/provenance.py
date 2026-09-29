"""Provenance stamps: git commit, source hashes, library versions."""

from __future__ import annotations

import hashlib
import pathlib
import platform
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = pathlib.Path(__file__).resolve().parent


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def source_hashes() -> dict:
    return {p.name: sha256_file(p) for p in sorted(SRC.glob("*.py"))}


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except Exception:
        return None


def stamp() -> dict:
    import numpy
    import torch
    return {"git": git_commit(), "python": platform.python_version(), "numpy": numpy.__version__,
            "torch": torch.__version__, "sources": source_hashes()}
