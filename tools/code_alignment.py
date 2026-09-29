"""Exploratory (v0.4, after H23): how similar are the two S rules' compiled memory codes and decoders?
Writes results/exploratory_code_alignment.json."""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = {r: json.loads((ROOT / f"results/v2_{r}.json").read_text())["H12"]["patterns"] for r in ("S0", "S1")}
D = {r: json.loads((ROOT / f"results/v3_{r}.json").read_text())["H16"]["decoder"]["w"] for r in ("S0", "S1")}
corr = lambda a, b: float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
d0 = np.array(P["S0"]["A"]) - np.array(P["S0"]["B"])
d1 = np.array(P["S1"]["A"]) - np.array(P["S1"]["B"])
out = {"note": "exploratory, after the H23 result; Pearson correlations between S0 and S1",
       "memory_direction_all_sites": corr(d0, d1), "memory_direction_edge_12_15": corr(d0[:, 12:16], d1[:, 12:16]),
       "decoder_weights": corr(D["S0"], D["S1"])}
(ROOT / "results/exploratory_code_alignment.json").write_text(json.dumps(out, indent=1))
print(out)
