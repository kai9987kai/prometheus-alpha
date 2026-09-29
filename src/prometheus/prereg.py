"""Preregistration lock.

``lock`` writes ``prereg/PREREGISTRATION.lock.json``: the SHA-256 of the preregistration and of
every source file that can change a confirmatory number. ``verify`` recomputes them. Results files
record the preregistration hash, and ``run`` labels its output CONFIRMATORY only while the lock
verifies (the Ghost in the Machine v4 rule).
"""

from __future__ import annotations

import datetime
import json

from .provenance import ROOT, SRC, sha256_file

PREREG = ROOT / "prereg" / "PREREGISTRATION.json"
LOCK = ROOT / "prereg" / "PREREGISTRATION.lock.json"
LOCKED_SOURCES = ["tissue.py", "body.py", "life.py", "experiments.py", "stats.py", "train.py"]


def _hashes() -> dict:
    return {"PREREGISTRATION.json": sha256_file(PREREG),
            **{f"src/prometheus/{n}": sha256_file(SRC / n) for n in LOCKED_SOURCES}}


def lock() -> dict:
    doc = {"locked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "sha256": _hashes()}
    LOCK.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def verify() -> tuple[bool, list[str]]:
    if not LOCK.exists():
        return False, ["no lock file"]
    want = json.loads(LOCK.read_text(encoding="utf-8"))["sha256"]
    have = _hashes()
    bad = [k for k in want if have.get(k) != want[k]]
    return not bad, bad
