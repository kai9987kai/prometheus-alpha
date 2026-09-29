"""Build claims/claims.json from the results files. Every quoted number is a JSON pointer into a results
file, so `prometheus claims check` can verify it. Run: python tools/build_claims.py && prometheus claims render"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = {n: json.loads((ROOT / f"results/rule_{n}.json").read_text()) for n in ("E0", "E1", "S0", "S1")}
F = {n: f"results/rule_{n}.json" for n in R}
for n in ("E0", "E1", "S0", "S1"):
    R["ph" + n] = json.loads((ROOT / f"results/posthoc_{n}.json").read_text())
    F["ph" + n] = f"results/posthoc_{n}.json"
R["pilot"] = json.loads((ROOT / "results/pilot/S102_n32.json").read_text())
F["pilot"] = "results/pilot/S102_n32.json"


def ev(rule, pointer, decimals=2):
    cur = R[rule]
    for k in pointer.strip("/").split("/"):
        cur = cur[int(k)] if isinstance(cur, list) else cur[k]
    return {"file": F[rule], "pointer": pointer, "value": cur, "decimals": decimals}


def sup(rule, h):
    return R[rule]["verdicts"][h]["supported"]


def status(h, rules):
    s = [sup(r, h) for r in rules]
    return "replicated" if all(s) else "mixed" if any(s) else "not supported"


def fp(p):
    return "sci" if p < 9.9e-4 else 3


claims = []


def add(cid, claim, st, text, evidence, headline=False):
    claims.append({"id": cid, "claim": claim, "status": st, "headline": headline, "evidence_text": text, "evidence": evidence})


S, E, ALL = ("S0", "S1"), ("E0", "E1"), ("E0", "E1", "S0", "S1")

# ---- headline: selection makes the memory outlive its organ
e = {}
for r in S:
    e[f"{r}_m"] = ev(r, "/H2/test/mean")
    e[f"{r}_ret"] = ev(r, "/H2/retention")
    e[f"{r}_dz"] = ev(r, "/H2/test/d_z")
    e[f"{r}_p"] = ev(r, "/verdicts/H2/p_adj", fp(R[r]["verdicts"]["H2"]["p_adj"]))
add("H2-S", "Selected rules remember through complete decapitation: the regrown head, built from cells that never smelled the odours, knows which one predicted the shock",
    status("H2", S),
    "memory M in the regrown head (paired minus unpaired twin, 128 worms): S0 {S0_m} (retention {S0_ret} of the intact memory, d_z {S0_dz}, Holm p {S0_p}); S1 {S1_m} (retention {S1_ret}, d_z {S1_dz}, Holm p {S1_p}). Positive control: S rules were selected for one regeneration",
    e, headline=True)

# ---- headline: without selection the head forgets
e = {}
for r in E:
    e[f"{r}_m"] = ev(r, "/H2/test/mean", 3)
    e[f"{r}_lo"] = ev(r, "/H2/test/ci95/0", 3)
    e[f"{r}_hi"] = ev(r, "/H2/test/ci95/1", 3)
    e[f"{r}_int"] = ev(r, "/H2/intact_memory")
    e[f"{r}_iou"] = ev(r, "/H2/regen_iou", 3)
add("H2-E", "Emergent rules also remember through decapitation",
    status("H2", E),
    "as predicted, no: E rules regrow near-perfect heads (IoU E0 {E0_iou}, E1 {E1_iou}) that carry none of an intact memory of {E0_int} / {E1_int}: M after regeneration E0 {E0_m} [95% CI {E0_lo}, {E0_hi}], E1 {E1_m} [{E1_lo}, {E1_hi}], both intervals far inside the smallest effect of interest (0.10)",
    e, headline=True)

# ---- H3c hidden channels
e = {}
for r in S:
    e[f"{r}_m"] = ev(r, "/H3c/test/mean")
    e[f"{r}_f"] = ev(r, "/H3c/transferred_fraction")
    e[f"{r}_p"] = ev(r, "/verdicts/H3c/p_adj", fp(R[r]["verdicts"]["H3c"]["p_adj"]))
add("H3c", "The memory in the body is carried by its hidden (non-electrical) channels: writing only those into an untrained twin's headless body makes the twin's new head remember",
    status("H3c", S),
    "M in the implanted twin's regrown head: S0 {S0_m} ({S0_f} of the donor's own), Holm p {S0_p}; S1 {S1_m} ({S1_f}), Holm p {S1_p}. Predicted from pilot S102 ({pilot})",
    e | {"pilot": ev("pilot", "/H3c/transferred_fraction")}, headline=True)

# ---- H3a / H3b voltage
e = {}
for r in S:
    e[f"{r}_kept"] = ev(r, "/H3a/retained_fraction")
    e[f"{r}_tr"] = ev(r, "/H3b/transferred_fraction")
st = "not supported" if not any(sup(r, h) for r in S for h in ("H3a", "H3b")) else "mixed"
add("H3ab", "The memory is bioelectric: the body's voltage pattern is necessary and sufficient for it",
    st,
    "as predicted, no. Swapping the trained body's voltage for its twin's kept S0 {S0_kept} / S1 {S1_kept} of the memory (H3a); implanting the voltage alone into the twin transferred S0 {S0_tr} / S1 {S1_tr} (H3b). Gap-junction diffusion is a fixed law here, and the rules did not use it to hold the memory",
    e)

# ---- H4b chimera regrown
e = {}
for r in S:
    e[f"{r}_d"] = ev(r, "/H4b/dominance")
    e[f"{r}_p"] = ev(r, "/verdicts/H4b/p_adj", fp(R[r]["verdicts"]["H4b"]["p_adj"]))
add("H4b", "Memory chimeras: cut the head off a chimera whose head learned one odour and whose body learned the other, and the regrown head remembers the body's",
    status("H4b", S),
    "predicted from the pilot, and not borne out: dominance after regeneration (+1 = the head's memory, -1 = the body's) S0 {S0_d} (Holm p {S0_p}), S1 {S1_d} (Holm p {S1_p}), far weaker than the pilot's {pilot}. What the regrown head remembers is mostly the rule's favoured odour (see the post-hoc claim)",
    e | {"pilot": ev("pilot", "/H4b/dominance")})

# ---- H4 chimera intact
e = {}
for r in ALL:
    e[f"{r}_d"] = ev(r, "/H4/dominance")
e["S0_ab"] = ev("S0", "/H4/preference/A|B")
e["S0_ba"] = ev("S0", "/H4/preference/B|A")
e["S1_ab"] = ev("S1", "/H4/preference/A|B")
e["S1_ba"] = ev("S1", "/H4/preference/B|A")
sE, sS = status("H4", E), status("H4", S)
add("H4-E", "In an intact chimera of an emergent rule, the head's memory wins outright",
    sE, "dominance of the head's memory after 32 steps of healing: E0 {E0_d}, E1 {E1_d}", e | {})
claims[-1]["evidence"] = {k: v for k, v in e.items() if k.startswith("E")}
add("H4-S", "In an intact chimera of a selected rule, head and body do not split the memory evenly",
    sS, "dominance S0 {S0_d}, S1 {S1_d} (predicted from the pilot: an even split). The head leads, but each rule also favours one odour in a conflict: R(A) - R(B) of A-head|B-body and B-head|A-body grafts, S0 {S0_ab} and {S0_ba}, S1 {S1_ab} and {S1_ba}",
    {k: v for k, v in e.items() if k.startswith("S")})

# ---- H5 transfer
e = {}
for r in S:
    e[f"{r}_f"] = ev(r, "/H5/transfer_fraction")
    e[f"{r}_m"] = ev(r, "/H5/test/mean")
    e[f"{r}_p"] = ev(r, "/verdicts/H5/p_adj", fp(R[r]["verdicts"]["H5"]["p_adj"]))
    e[f"{r}_ta"] = ev("ph" + r, "/derived/transfer_fraction_A")
    e[f"{r}_tb"] = ev("ph" + r, "/derived/transfer_fraction_B")
add("H5", "A naive head grafted onto a trained body takes on the body's memory, with no amputation or regeneration",
    status("H5", S),
    "transfer (half the difference between naive heads on A- and B-trained bodies): S0 {S0_m} ({S0_f} of the uncontested memory, Holm p {S0_p}); S1 {S1_m} ({S1_f}, Holm p {S1_p}). Post hoc, it is all or none by odour: S0 passes on B ({S0_tb}) but not A ({S0_ta}); S1 passes on A ({S1_ta}) far more than B ({S1_tb})",
    e)

# ---- H6 Promethean
e = {}
for r in S:
    for k in range(4):
        e[f"{r}_c{k+1}"] = ev(r, f"/H6/cycles/{k}/mean")
    e[f"{r}_iou"] = ev(r, "/H6/cycles/3/anatomy/iou")
add("H6", "Promethean cycles: the memory survives a third successive decapitation (only one was ever trained)",
    status("H6", S),
    "M after cycles 1, 2, 3, 4: S0 {S0_c1}, {S0_c2}, {S0_c3}, {S0_c4}; S1 {S1_c1}, {S1_c2}, {S1_c3}, {S1_c4}. Body IoU after cycle 4: S0 {S0_iou}, S1 {S1_iou}",
    e, headline=True)

# ---- H7 fragment
e = {}
for r in S:
    e[f"{r}_m"] = ev(r, "/H7/test/mean")
    e[f"{r}_iou"] = ev(r, "/H7/regen_iou")
add("H7", "A trunk fragment, with head and tail both removed, regrows a head that remembers (never trained)",
    status("H7", S), "M: S0 {S0_m}, S1 {S1_m}; body IoU after regeneration {S0_iou}, {S1_iou}", e)

# ---- H1 learning
e = {}
for r in ALL:
    e[f"{r}_m"] = ev(r, "/H1/test/mean")
    e[f"{r}_dz"] = ev(r, "/H1/test/d_z", 1)
add("H1", "Worms learn which odour predicted the shock, in their state alone (the rule is frozen within a life)",
    status("H1", ALL),
    "M in intact worms (paired minus unpaired twin, 128 worms): E0 {E0_m} (d_z {E0_dz}), E1 {E1_m} ({E1_dz}), S0 {S0_m} ({S0_dz}), S1 {S1_m} ({S1_dz})",
    e)


# ---- post hoc: favoured odour
e = {}
for r in S:
    e[f"{r}_bi"] = ev("ph" + r, "/derived/chimera_cue_bias_intact")
    e[f"{r}_br"] = ev("ph" + r, "/derived/chimera_cue_bias_regrown")
    for c in ("A", "B"):
        e[f"{r}_{c}1"] = ev("ph" + r, f"/promethean_by_cue/cycles/0/{c}")
        e[f"{r}_{c}4"] = ev("ph" + r, f"/promethean_by_cue/cycles/3/{c}")
for r in E:
    e[f"{r}_br"] = ev("ph" + r, "/derived/chimera_cue_bias_regrown")
add("posthoc-attractor", "Post hoc: selection made one odour's memory an attractor and the other metastable. Conflicts, transfers and repeated regrowth all resolve toward the rule's favoured odour",
    "post hoc",
    "odour bias of conflicting chimeras (R(A) - R(B) averaged over both grafts; > 0 favours A) intact / regrown: S0 {S0_bi} / {S0_br}, S1 {S1_bi} / {S1_br}; E rules regrown {E0_br}, {E1_br}. Memory for each CS+ after Promethean cycle 1 and 4 (fresh cohort): S1 favoured A {S1_A1} to {S1_A4}, other B {S1_B1} to {S1_B4}; S0 favoured B {S0_B1} to {S0_B4}, other A {S0_A1} to {S0_A4}. Not preregistered; found by reading the H4, H4b and H5 grafts",
    e, headline=True)

# ---- SESOI saves
e, names = {}, []
for r in ALL:
    for h, v in R[r]["verdicts"].items():
        if v["reject"] and not v["sesoi_met"]:
            k = f"{r}_{h}"
            e[k] = ev(r, f"/verdicts/{h}/mean", 3)
            names.append(f"{r} {h} {{{k}}}")
add("SESOI", "Significant but trivial effects are not claims",
    "descriptive",
    f"{len(names)} of 40 confirmatory tests passed Holm but moved the response by less than the smallest effect of interest (0.10), so none is claimed: " + ", ".join(names),
    e)

# ---- locus exploratory
e = {}
for r in S:
    for g in ("voltage", "hidden", "identity+response", "all but voltage"):
        e[f"{r}_{g.replace(' ', '_').replace('+', '_')}"] = ev(r, f"/locus/implanted_fraction/{g}")
add("locus", "Exploratory: which channels, implanted alone into the twin's headless body, carry the memory",
    "exploratory",
    "fraction of the donor's memory: S0 voltage {S0_voltage}, hidden {S0_hidden}, identity+response {S0_identity_response}, all but voltage {S0_all_but_voltage}; S1 {S1_voltage}, {S1_hidden}, {S1_identity_response}, {S1_all_but_voltage}",
    e)

# ---- gap junctions exploratory
e = {}
for r in S:
    e[f"{r}_o"] = ev(r, "/gap_junctions/memory/open/cut")
    e[f"{r}_b"] = ev(r, "/gap_junctions/memory/blocked/cut")
add("gj", "Exploratory: blocking gap junctions for the delay between learning and the cut does not stop the memory reaching the body",
    "exploratory", "M in the regrown head, junctions open vs blocked: S0 {S0_o} vs {S0_b}, S1 {S1_o} vs {S1_b}", e)

# ---- calibration
e = {r: ev(r, "/calibration/false_positive_rate", 3) for r in ALL}
add("calibration", "The pipeline is calibrated: A/A comparisons produce false positives at about the nominal rate",
    "supported" if all(R[r]["calibration"]["false_positive_rate"] <= 0.1 for r in ALL) else "not supported",
    "false-positive rate at alpha 0.05 over 40 A/A tests of 128 worms each (nominal 0.05, i.e. 2 of 40): E0 {E0}, E1 {E1}, S0 {S0}, S1 {S1}", e)

# ---- anatomy
e = {}
for r in ALL:
    e[f"{r}_g"] = ev(r, "/anatomy/grown/iou")
    e[f"{r}_h"] = ev(r, "/anatomy/regen_head/iou")
    e[f"{r}_l"] = ev(r, "/anatomy/held_300_steps/iou")
add("anatomy", "All four rules grow the body from one founder cell and regrow it after decapitation",
    "descriptive",
    "body IoU with the target after 32 steps of growth / after head regeneration / 300 steps later: E0 {E0_g} / {E0_h} / {E0_l}; E1 {E1_g} / {E1_h} / {E1_l}; S0 {S0_g} / {S0_h} / {S0_l}; S1 {S1_g} / {S1_h} / {S1_l}",
    e)

ORDER = ["H2-S", "H2-E", "H3c", "H6", "posthoc-attractor", "H4b", "H3ab", "H4-E", "H4-S", "H5", "H7", "H1",
         "SESOI", "locus", "gj", "calibration", "anatomy"]
claims.sort(key=lambda c: ORDER.index(c["id"]))
doc = {"schema": 1, "note": "Built from results/rule_*.json. Failed and unsupported claims stay in the ledger.", "claims": claims}
(ROOT / "claims/claims.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
print(f"{len(claims)} claims")
for c in claims:
    print(f"- [{c['status']}] {c['claim']}")
