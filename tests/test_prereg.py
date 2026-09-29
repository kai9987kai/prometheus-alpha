"""The preregistration, its lock and the code agree."""

import json

from prometheus import experiments, prereg


def test_confirmatory_family_matches_the_preregistration():
    doc = json.loads(prereg.PREREG.read_text())
    assert list(doc["hypotheses"]) == list(experiments.CONFIRMATORY)
    assert set(doc["predictions"]) == set(experiments.CONFIRMATORY)
    for h in experiments.CONFIRMATORY:
        assert (h if h != "H4b" else "H4") in experiments.SEEDS     # H4b reuses H4's chimeras


def test_lock_verifies():
    ok, bad = prereg.verify()
    assert ok, f"changed since the preregistration was locked: {bad}"
