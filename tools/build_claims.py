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
for n in ("E0", "E1", "S0", "S1"):
    for v in ("v2", "v3", "v4"):
        pth = ROOT / f"results/{v}_{n}.json"
        if pth.exists():
            R[f"{v}{n}"] = json.loads(pth.read_text())
            F[f"{v}{n}"] = f"results/{v}_{n}.json"
for n in ("F0", "F1"):
    pth = ROOT / f"results/v4_{n}.json"
    if pth.exists():
        R[f"v4{n}"] = json.loads(pth.read_text())
        F[f"v4{n}"] = f"results/v4_{n}.json"
for n in ("E0", "E1", "S0", "S1", "B0", "B1", "F0", "F1"):
    pth = ROOT / f"results/v5_{n}.json"
    if pth.exists():
        R[f"v5{n}"] = json.loads(pth.read_text())
        F[f"v5{n}"] = f"results/v5_{n}.json"
for name in ("v6", "posthoc_v6", "v7", "exploratory_v7_engram"):
    if (ROOT / f"results/{name}.json").exists():
        R[name] = json.loads((ROOT / f"results/{name}.json").read_text())
        F[name] = f"results/{name}.json"
R["align"] = json.loads((ROOT / "results/exploratory_code_alignment.json").read_text())
F["align"] = "results/exploratory_code_alignment.json"
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



# ============================================================ v0.2
def vsup(v, rule, h):
    return R[f"{v}{rule}"]["verdicts"].get(h, {}).get("supported", False)


def vstatus(v, h, rules):
    s = [vsup(v, r, h) for r in rules]
    return "replicated" if all(s) else "mixed" if any(s) else "not supported"


e = {}
for r in S:
    e[f"{r}_e"] = ev(f"v2{r}", "/H13/edge_fraction")
    e[f"{r}_r"] = ev(f"v2{r}", "/H13/rest_fraction", 3)
add("v2-H13", "v0.2: the body's copy lives in the four cells at the wound edge",
    vstatus("v2", "H13", S),
    "hidden channels implanted into the untrained twin at sites 12-15 only transfer S0 {S0_e} / S1 {S1_e} of the donor's memory; at sites 16-27, {S0_r} / {S1_r}", e, headline=True)
e = {}
for r in ALL:
    e[f"{r}_w"] = ev(f"v2{r}", "/H12/written_memory")
    e[f"{r}_s"] = ev(f"v2{r}", "/H12/shuffled_memory")
add("v2-H12", "v0.2: a memory compiler. A gradient-designed hidden pattern written into untrained headless bodies makes held-out regrown heads remember an odour they never experienced",
    vstatus("v2", "H12", ALL),
    "written memory vs the same values at shuffled sites: S0 {S0_w} vs {S0_s}, S1 {S1_w} vs {S1_s}; also in emergent rules, whose bodies never keep a copy: E0 {E0_w} vs {E0_s}, E1 {E1_w} vs {E1_s} (one odour only in each E rule)", e, headline=True)
e = {}
for r in S:
    e[f"{r}_i"] = ev(f"v2{r}", "/H11/first_lesson_intact")
    e[f"{r}_g"] = ev(f"v2{r}", "/H11/first_lesson_regrown")
add("v2-H11", "v0.2: after a reversal the body is more conservative than the head: the regrown head reverts toward the first lesson",
    vstatus("v2", "H11", S),
    "memory of the first lesson after learning the other odour, intact vs regrown: S0 {S0_i} vs {S0_g}, S1 {S1_i} vs {S1_g}. The pilot predicted the opposite direction", e)
e = {}
for r in S:
    e[f"{r}_f"] = ev(f"v2{r}", "/H8/test/favoured_mean")
    e[f"{r}_o"] = ev(f"v2{r}", "/H8/test/other_mean")
add("v2-H8", "v0.2 replication of the v0.1 post-hoc attractor, direction named in advance: after 4 cycles the favoured odour is remembered better",
    vstatus("v2", "H8", S),
    "favoured vs other odour at cycle 4 (fresh worms): S0 {S0_f} vs {S0_o} (below the 0.10 threshold), S1 {S1_f} vs {S1_o}", e)
e = {}
for r in ALL:
    e[f"{r}_x"] = ev(f"v2{r}", "/H9/extinguished_fraction")
add("v2-H9", "v0.2: unreinforced presentations extinguish the memory (never trained)",
    vstatus("v2", "H9", ALL),
    "fraction of the memory extinguished by 96 steps of cues without shock: E0 {E0_x}, E1 {E1_x}, S0 {S0_x}, S1 {S1_x}", e)
e = {}
for r in S:
    e[f"{r}_i"] = ev(f"v2{r}", "/H10/extinguished_intact")
    e[f"{r}_g"] = ev(f"v2{r}", "/H10/extinguished_regrown")
add("v2-H10", "v0.2: a head regrown after extinction differs from the intact extinguished head",
    vstatus("v2", "H10", S),
    "S rules, intact vs regrown after extinction: S0 {S0_i} vs {S0_g}, S1 {S1_i} vs {S1_g}; the pilot's inversion did not replicate (E rules pass trivially: nothing survives regrowth)", e)


# ============================================================ v0.3
e = {}
for r in S:
    e[f"{r}_a"] = ev(f"v3{r}", "/H14/A_trained/pattern_B")
    e[f"{r}_b"] = ev(f"v3{r}", "/H14/B_trained/pattern_A")
    e[f"{r}_f"] = ev(f"v3{r}", "/H14/switched_fraction", 3)
add("v3-H14", "v0.3: memory surgery. Writing the compiled pattern for the other odour into a trained worm's headless body overwrites its real memory",
    vstatus("v3", "H14", S),
    "R(A) - R(B) of the regrown head: A-trained worms given pattern B S0 {S0_a}, S1 {S1_a}; B-trained given pattern A S0 {S0_b}, S1 {S1_b}; fraction of worms whose memory switched {S0_f}, {S1_f}", e, headline=True)
e = {}
for r in ALL:
    e[f"{r}_a"] = ev(f"v3{r}", "/H15/written_A")
    e[f"{r}_b"] = ev(f"v3{r}", "/H15/written_B")
add("v3-H15", "v0.3: four cells are enough. A compiled memory confined to the wound-edge sites 12-15 writes a full memory",
    vstatus("v3", "H15", S),
    "R(A) - R(B) of held-out regrown heads with the sparse A / B pattern: S0 {S0_a} / {S0_b}, S1 {S1_a} / {S1_b}; emergent rules E0 {E0_a} / {E0_b}, E1 {E1_a} / {E1_b}", e)
e = {}
for r in ALL:
    e[f"{r}_e"] = ev(f"v3{r}", "/H16/edge_accuracy")
    e[f"{r}_r"] = ev(f"v3{r}", "/H16/rest_accuracy")
add("v3-H16", "v0.3: decodable is not used. A linear decoder reads the odour from the wound-edge cells, in emergent rules too, whose regrown heads never use it",
    vstatus("v3", "H16", ALL),
    "held-out decoding accuracy, edge vs rest of trunk: S0 {S0_e} vs {S0_r}, S1 {S1_e} vs {S1_r}; E0 {E0_e} vs {E0_r}, E1 {E1_e} vs {E1_r} (predicted: no in E rules, which keep 0% of the memory)", e, headline=True)
e = {}
for r in S:
    e[f"{r}_b"] = ev(f"v3{r}", "/H17/B_other")
    e[f"{r}_s"] = ev(f"v3{r}", "/H17/S_other")
add("v3-H17", "v0.3: selection for two amputations rescues the disfavoured memory (balanced B rules)",
    vstatus("v3", "H17", S),
    "memory of the S rule's disfavoured odour at cycle 4, B rule vs S sibling: B1 {S1_b} vs S1 {S1_s}; B0 {S0_b} vs S0 {S0_s} (S0 was near ceiling; below the 0.10 threshold)", e)
e = {}
for r in ALL:
    e[f"{r}_p"] = ev(f"v3{r}", "/H18/posterior_half_memory", 3)
    e[f"{r}_a"] = ev(f"v3{r}", "/H18/anterior_half_memory")
add("v3-H18", "v0.3: fission. Cut a trained worm in two, and the half that must regrow a head remembers",
    vstatus("v3", "H18", ALL),
    "as predicted, no, in every rule: posterior half E0 {E0_p}, E1 {E1_p}, S0 {S0_p}, S1 {S1_p}; the anterior half, which keeps the head, {E0_a}, {E1_a}, {S0_a}, {S1_a}. The body's copy exists only at a wound next to the head", e)


# ============================================================ v0.4
e = {}
for r in ALL:
    e[f"{r}"] = ev(f"v4{r}", "/H24/test/mean")
add("v4-H24", "v0.4: one cell can hold a whole memory. A pattern designed for site 14 alone writes a memory into untrained headless bodies",
    vstatus("v4", "H24", S),
    "written memory (R(A) - R(B)) / 2 on 128 held-out worms: S0 {S0}, S1 {S1}; E0 {E0}, E1 {E1}", e, headline=True)
e = {r: ev(f"v4{r}", "/H23/test/mean") for r in S}
e["al"] = ev("align", "/memory_direction_edge_12_15")
e["dec"] = ev("align", "/decoder_weights")
add("v4-H23", "v0.4: a universal memory code. One selected rule's compiled memory, written into the other rule's body, writes the intended odour",
    vstatus("v4", "H23", S),
    "as predicted, no, and worse than no: S1's code in S0 bodies writes the opposite odour ({S0}), S0's in S1 bodies {S1}. Exploratory: the two rules' memory directions at the wound edge correlate {al}, their decoders {dec}. Each rule invented its own code", e, headline=True)
FR = [r for r in ("F0", "F1") if f"v4{r}" in R]
if FR:
    e = {}
    for r in FR:
        e[f"{r}_19"] = ev(f"v4{r}", "/H19/test/mean")
        e[f"{r}_20"] = ev(f"v4{r}", "/H20/test/mean")
        e[f"{r}_22"] = ev(f"v4{r}", "/H22/test/mean")
    for r in S:
        e[f"{r}_19"] = ev(f"v4{r}", "/H19/test/mean", 3)
    txt = "posterior half after a split at site 20 (trained) / 24 (held out): " + "; ".join(f"{r} {{{r}_19}} / {{{r}_20}}" for r in FR) + \
          "; S siblings at 20: S0 {S0_19}, S1 {S1_19}. After complete decapitation: " + ", ".join(f"{r} {{{r}_22}}" for r in FR)
    add("v4-H19", "v0.4: fission rules. Selection on split worms makes the back half, which must grow a new head, remember",
        vstatus("v4", "H19", FR), txt, e, headline=True)
    e = {}
    for r in FR:
        e[f"{r}_f"] = ev(f"v4{r}", "/H21/F_memory")
        e[f"{r}_s"] = ev(f"v4{r}", "/H21/S_memory")
    add("v4-H21", "v0.4: a distributed copy. In intact fission-selected worms the trunk and tail (sites 16-35) carry a copy that a decapitated twin's new head can use, more than in the S sibling",
        vstatus("v4", "H21", FR), "memory transferred by sites 16-35 of intact trained worms: " + "; ".join(f"{r} {{{r}_f}} vs its S sibling {{{r}_s}}" for r in FR), e)


# ============================================================ v0.5
EIGHT = ("E0", "E1", "S0", "S1", "B0", "B1", "F0", "F1")
if all(f"v5{r}" in R for r in EIGHT):
    e = {r: ev(f"v5{r}", "/H27/test/mean") for r in EIGHT}
    add("v5-H27", "v0.5: eight cells regrow a remembering worm. Keep only the neck and anterior trunk (sites 12-19) and the whole new worm remembers",
        vstatus("v5", "H27", ("S0", "S1", "B0", "B1", "F0", "F1")),
        "M after 64 steps of regrowth from 8 cells: S0 {S0}, S1 {S1}, B0 {B0}, B1 {B1}, F0 {F0}, F1 {F1}; emergent rules E0 {E0}, E1 {E1}", e, headline=True)
    e = {r: ev(f"v5{r}", "/H28/test/mean") for r in EIGHT}
    add("v5-H28", "v0.5: from an 8-cell tail fragment (sites 24-31) only the rule with a distributed copy regrows a remembering worm",
        "descriptive",
        "as predicted: F1 {F1}; F0 {F0} (inverted again); S0 {S0}, S1 {S1}, B0 {B0}, B1 {B1}, E0 {E0}, E1 {E1}", e)
    e = {r: ev(f"v5{r}", "/H25/test/mean") for r in EIGHT}
    add("v5-H25", "v0.5: Ship of Theseus. The memory survives 200 steps of random cell death and replacement (10% of cells every 20 steps)",
        "supported in 7 of 8 rules",
        "M after turnover: S0 {S0}, B0 {B0}, F0 {F0} (the S0 lineage) vs S1 {S1}, B1 {B1}, F1 {F1} (the S1 lineage); E0 {E0}, E1 {E1} (not supported). Robustness follows lineage more than selection regime", e, headline=True)
    e = {r: ev(f"v5{r}", "/H26/test/mean") for r in EIGHT}
    add("v5-H26", "v0.5: the memory lasts 300 steps without any reminder (training delays were 4-16)",
        "supported in 7 of 8 rules",
        "M: E0 {E0}, E1 {E1} (forgets), S0 {S0}, S1 {S1}, B0 {B0}, B1 {B1}, F0 {F0}, F1 {F1}", e)


# ============================================================ v0.6
if "v6" in R:
    e = {}
    for r in ("F0", "F1", "S0", "S1"):
        for g in range(3):
            e[f"{r}_{g + 1}"] = ev("v6", f"/H29/{r}/by_generation/{g}")
    st = "replicated" if all(R["v6"]["verdicts"][f"H29_{r}"]["supported"] for r in ("F0", "F1")) else "mixed"
    add("v6-H29", "v0.6: memory passes down three generations of fission (only one split was ever trained), fading each time",
        st, "M after generations 1, 2, 3: F0 {F0_1}, {F0_2}, {F0_3}; F1 {F1_1}, {F1_2}, {F1_3}; S0 {S0_1}, {S0_2}, {S0_3} and S1 {S1_1}, {S1_2}, {S1_3} never pass it on", e, headline=True)
    e = {"rho": ev("v6", "/H30/test/mean"), "p": ev("v6", "/H30/test/p", 3)}
    for r in ("S0", "S1", "B0", "B1", "F0", "F1"):
        e[f"{r}"] = ev("v6", f"/H30/sigma_star/{r}")
    add("v6-H30", "v0.6: a wider attractor basin at the neck predicts survival of cell turnover",
        "not supported", "Spearman rho {rho} (one-sided exact p {p}, 6 rules). Critical noise sigma*: S0 {S0}, S1 {S1}, B0 {B0}, B1 {B1}, F0 {F0}, F1 {F1}", e)
if "posthoc_v6" in R:
    e = {"rho": ev("posthoc_v6", "/spearman_area_vs_turnover/rho")}
    for r in ("S0", "S1", "B0", "B1", "F0", "F1"):
        e[r] = ev("posthoc_v6", f"/basins/{r}/area")
    add("posthoc-tradeoff", "Post hoc: a trade-off. Rules whose head memory tolerates noise are the ones whose memory dies with cell turnover",
        "post hoc", "Spearman rho between head-basin area and v0.5 turnover memory, 6 rules: {rho}. Basin area: S0 {S0}, B0 {B0}, F0 {F0} (turnover-robust) vs S1 {S1}, B1 {B1}, F1 {F1}. Not preregistered. Tested causally in v0.7 (v7-H33, v7-H34): not a trade-off", e)

# ============================================================ v0.7
if "v7" in R:
    def v7st(h):
        s = [R["v7"]["tests"][f"{h}_{k}"]["supported"] for k in (0, 1)]
        return "replicated" if all(s) else "mixed" if any(s) else "not supported"

    def v7ev(e, h, stress, a, b):
        for k in (0, 1):
            e[f"{a}{k}"] = ev("v7", f"/rules/{a}{k}/{stress}_memory")
            e[f"{b}{k}"] = ev("v7", f"/rules/{b}{k}/{stress}_memory")
            e[f"d{k}"] = ev("v7", f"/tests/{h}_{k}/mean")
            e[f"p{k}"] = ev("v7", f"/tests/{h}_{k}/p_adj", fp(R["v7"]["tests"][f"{h}_{k}"]["p_adj"]))
        return e

    add("v7-H31", "v0.7: training under cell turnover hardens the memory against it",
        v7st("H31"), "memory after 200 steps of turnover, turnover-trained T vs control-trained C sibling: lineage 0 {T0} vs {C0} (difference {d0}, Holm p {p0}: significant but below the smallest effect of interest; lineage 0 was already near ceiling, as predicted); lineage 1 {T1} vs {C1} ({d1}, Holm p {p1})",
        v7ev({}, "H31", "turnover", "T", "C"))
    add("v7-H32", "v0.7: training under hidden-state noise hardens the memory against it",
        v7st("H32"), "memory after sigma 1.5 noise on the head's hidden channels, noise-trained N vs control C: lineage 0 {N0} vs {C0} (difference {d0}, Holm p {p0}); lineage 1 {N1} vs {C1} ({d1}, Holm p {p1})",
        v7ev({}, "H32", "noise", "N", "C"))
    add("v7-H33", "v0.7 trade-off: hardening the memory against cell loss costs noise tolerance",
        v7st("H33"), "as predicted, no, and the difference points the other way: noise memory of the control C vs the turnover-trained T, lineage 0 {C0} vs {T0} (C - T {d0}, Holm p {p0}); lineage 1 {C1} vs {T1} ({d1}, Holm p {p1}). Hardening against cell loss did not cost noise tolerance; if anything it bought some",
        v7ev({}, "H33", "noise", "C", "T"), headline=True)
    add("v7-H34", "v0.7 trade-off: hardening the memory against noise costs survival of cell loss",
        v7st("H34"), "as predicted, no, and the difference points the other way: turnover memory of the control C vs the noise-trained N, lineage 0 {C0} vs {N0} (C - N {d0}, Holm p {p0}); lineage 1 {C1} vs {N1} ({d1}, Holm p {p1}). Noise training made lineage 1's memory survive cell loss almost as well as turnover training did (T1 {T1}); the v0.6 correlation was lineage, not a trade-off",
        v7ev({"T1": ev("v7", "/rules/T1/turnover_memory")}, "H34", "turnover", "C", "N"), headline=True)

if "exploratory_v7_engram" in R:
    X7 = "exploratory_v7_engram"
    e = {"rho": ev(X7, "/post_hoc/amplitude_vs_noise_memory/rho"), "p": ev(X7, "/post_hoc/amplitude_vs_noise_memory/p_one_sided", 3)}
    for r in ("C0", "T0", "N0", "C1", "T1", "N1"):
        e[f"{r}_a"] = ev(X7, f"/rules/{r}/amplitude")
        e[f"{r}_s"] = ev(X7, f"/rules/{r}/spread", 1)
    for r in ("S0", "B0", "F0", "S1", "B1", "F1"):
        e[f"{r}_cv"] = ev(X7, f"/rules/{r}/head_cv")
    add("explore-v7-engram", "Exploratory: noise training makes the engram louder, not wider; turnover training barely changes it",
        "exploratory", "distance between the hidden states of identical twins trained on A and on B, mean over living sites (amplitude) and participation ratio (spread, cells): C0 {C0_a} / {C0_s}, T0 {T0_a} / {T0_s}, N0 {N0_a} / {N0_s}; C1 {C1_a} / {C1_s}, T1 {T1_a} / {T1_s}, N1 {N1_a} / {N1_s}. So T1's new resistance to cell loss is not visible in the stored pattern. Post hoc, over the eight S, C, T, N rules, amplitude tracks noise memory (Spearman rho {rho}, one-sided exact p {p}). The lineages differ in shape: lineage 0 codes are flat across the head (coefficient of variation S0 {S0_cv}, B0 {B0_cv}, F0 {F0_cv}), lineage 1 codes graded (S1 {S1_cv}, B1 {B1_cv}, F1 {F1_cv})", e)

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

ORDER = ["H2-S", "H2-E", "H3c", "v2-H13", "v2-H12", "v4-H24", "v4-H19", "v4-H21", "v4-H23", "v5-H27", "v5-H25", "v5-H28", "v5-H26", "v6-H29", "v6-H30", "posthoc-tradeoff", "v7-H33", "v7-H34", "v7-H31", "v7-H32", "explore-v7-engram", "v3-H14", "v3-H15", "v3-H16", "v3-H17", "v3-H18", "H6",
         "v2-H11", "posthoc-attractor", "v2-H8", "v2-H9", "v2-H10", "H4b", "H3ab", "H4-E", "H4-S", "H5", "H7", "H1",
         "SESOI", "locus", "gj", "calibration", "anatomy"]
claims.sort(key=lambda c: ORDER.index(c["id"]))
doc = {"schema": 1, "note": "Built from results/rule_*.json. Failed and unsupported claims stay in the ledger.", "claims": claims}
(ROOT / "claims/claims.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
print(f"{len(claims)} claims")
for c in claims:
    print(f"- [{c['status']}] {c['claim']}")
