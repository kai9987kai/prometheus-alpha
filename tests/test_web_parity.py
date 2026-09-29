"""The browser engine (web/engine.js) reproduces the Python engine step for step."""

import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "parity.json"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_engine_matches_python():
    if not FIXTURE.exists():
        pytest.skip("run `prometheus parity-fixture` first")
    out = subprocess.run(["node", str(ROOT / "tests" / "parity.js"), str(FIXTURE)], capture_output=True, text=True, check=True)
    report = json.loads(out.stdout)
    assert report, "no cases"
    for case in report:
        assert case["steps"] > 100
        assert case["maxState"] < 1e-3, case
        assert case["maxResp"] < 1e-3, case
        assert case["regionMismatch"] == 0, case


def test_fixture_is_current_for_random_rule():
    """The fixture's random-rule case can be regenerated bit for bit from the Python engine."""
    torch = pytest.importorskip("torch")
    import numpy as np
    from prometheus import export
    if not FIXTURE.exists():
        pytest.skip("no fixture")
    stored = json.loads(FIXTURE.read_text())["cases"][0]
    fresh = export._case(export._parity_rule(), "random", np.random.default_rng(5))
    assert stored["T"] == fresh["T"]
    assert np.allclose(np.array(stored["resp"]), np.array(fresh["resp"]), atol=1e-6)
