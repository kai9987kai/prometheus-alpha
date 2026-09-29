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

LOCKED_SOURCES = ["tissue.py", "body.py", "life.py", "experiments.py", "stats.py", "train.py"]
VERSIONS = {  # version -> (preregistration, lock, extra locked sources)
    "1": ("PREREGISTRATION.json", "PREREGISTRATION.lock.json", []),
    "2": ("PREREGISTRATION_v2.json", "PREREGISTRATION_v2.lock.json", ["v2.py"]),
    "3": ("PREREGISTRATION_v3.json", "PREREGISTRATION_v3.lock.json", ["v2.py", "v3.py", "train_v3.py"]),
    "4": ("PREREGISTRATION_v4.json", "PREREGISTRATION_v4.lock.json", ["v2.py", "v3.py", "v4.py", "train_v4.py"]),
    "5": ("PREREGISTRATION_v5.json", "PREREGISTRATION_v5.lock.json", ["v5.py"]),
    "6": ("PREREGISTRATION_v6.json", "PREREGISTRATION_v6.lock.json", ["v2.py", "v6.py"]),
}
PREREG = ROOT / "prereg" / VERSIONS["1"][0]
LOCK = ROOT / "prereg" / VERSIONS["1"][1]


def _hashes(version: str = "1") -> dict:
    name, _, extra = VERSIONS[version]
    return {name: sha256_file(ROOT / "prereg" / name),
            **{f"src/prometheus/{n}": sha256_file(SRC / n) for n in LOCKED_SOURCES + extra}}


def lock(version: str = "1") -> dict:
    doc = {"locked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "sha256": _hashes(version)}
    (ROOT / "prereg" / VERSIONS[version][1]).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def verify(version: str = "1") -> tuple[bool, list[str]]:
    lk = ROOT / "prereg" / VERSIONS[version][1]
    if not lk.exists():
        return False, ["no lock file"]
    want = json.loads(lk.read_text(encoding="utf-8"))["sha256"]
    have = _hashes(version)
    bad = [k for k in want if have.get(k) != want[k]]
    return not bad, bad
