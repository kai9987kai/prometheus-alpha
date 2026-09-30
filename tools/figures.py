"""Paper figures, built from the results files (and, for Fig 1, one simulated worm per rule).

    python tools/figures.py        # writes paper/figures/*.png
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prometheus import body, experiments as X, life  # noqa: E402
from prometheus import tissue as T  # noqa: E402
from prometheus.train import load  # noqa: E402

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
REG = {0: "#eda100", 1: "#1baf7a", 2: "#4a3aa7"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
R = lambda name: json.loads((ROOT / "results" / name).read_text())


def fig1_kymograph():
    """One worm per rule: learn A, lose the head, regrow, lose it again. Top: body plan.
    Bottom: which odour each cell's hidden state encodes (v0.3 decoder)."""
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.6), sharex=True, sharey=True)
    for col, rule in enumerate(["E0", "S0"]):
        p = load(str(ROOT / f"weights/rule_{rule}.json"))["tensors"]
        dec = R(f"v3_{rule}.json")["H16"]["decoder"] if (ROOT / f"results/v3_{rule}.json").exists() else None
        base = X.cohort(7, 2, 0)                                  # two twins: CS+ = A and CS+ = B, same noise
        co = X.Cohort(base.seed, np.array([0, 1]), np.array([0, 0]), np.array([base.mask_seeds[0]] * 2))
        with torch.no_grad():
            w = X.Worms.found(p, co)
            frames, frames2, events = [], [], []
            tl = life.Timeline(2).free(X.GROW)
            tl.condition(co.csplus, "paired", np.random.default_rng(1)).free(X.DELAY)
            for seg in ("learn", "cut", "regen", "cut", "regen"):
                if seg == "cut":
                    w = w.cut("head")[0]
                    events.append(len(frames))
                    continue
                tl = tl if seg == "learn" else life.Timeline(2).free(X.REGEN)
                x0 = w.x
                for t in range(tl.t):
                    w = X.Worms(p, co, x0, w.t, w.masks) if t == 0 else w
                    w, _ = w.play(_one(tl, t))
                    frames.append(w.x[0].clone())
                    frames2.append(w.x.clone())
        lab = np.array([body.region(f[None])[0].numpy() for f in frames]).T.astype(float)
        img = np.full(lab.shape + (3,), 1.0)
        for k, c in REG.items():
            img[lab == k] = matplotlib.colors.to_rgb(c)
        axes[0, col].imshow(img, aspect="auto", interpolation="nearest")
        axes[0, col].set_title(f"{'emergent' if rule[0] == 'E' else 'selected'} rule {rule}: body plan", fontsize=9, loc="left", color=INK)
        if dec:
            wv, b = np.array(dec["w"]), dec["b"]
            hid = np.array([f.numpy() for f in frames2])                   # (t, 2 worms, C, L)
            code = np.einsum("twcl,c->twl", hid[:, :, T.HIDDEN0:], wv)
            diff = 0.5 * (code[:, 0] - code[:, 1]).T                      # odour information per cell
            alive = np.array([((f[:, T.ALPHA] > 0.1).all(0)).numpy() for f in frames2]).T
            diff = np.where(alive, diff, np.nan)
            m = np.nanpercentile(np.abs(diff), 99)
            im = axes[1, col].imshow(diff / m, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        axes[1, col].set_title("odour information per cell", fontsize=9, loc="left", color=INK)
        for ax in axes[:, col]:
            ax.grid(False)
            for e in events:
                ax.axvline(e, color=INK, lw=1)
            ax.axvline(X.GROW, color=INK2, lw=0.8, ls=":")
    for ax in axes[:, 0]:
        ax.set_ylabel("site (head at top)")
    fig.supxlabel("step (dotted: conditioning begins; solid: decapitation)", fontsize=9, color=INK2)
    fig.colorbar(im, ax=axes[1, :], shrink=0.8, label="A-trained minus B-trained twin\n(red A, blue B)")
    fig.savefig(OUT / "fig1_kymograph.png")


def _one(tl, t):
    """A timeline holding only step t of ``tl``."""
    s = life.Timeline(tl.B)
    s.cue, s.us, s.gj = [tl.cue[t]], [tl.us[t]], [tl.gj[t]]
    return s


def fig2_survival():
    rules = ["E0", "E1", "S0", "S1"]
    intact = [R(f"rule_{r}.json")["H1"]["test"]["mean"] for r in rules]
    regrown = [R(f"rule_{r}.json")["H2"]["test"]["mean"] for r in rules]
    lo = [R(f"rule_{r}.json")["H2"]["test"]["ci95"] for r in rules]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    xs = np.arange(len(rules))
    ax.bar(xs - 0.19, intact, 0.36, color=BLUE, label="intact (H1)")
    ax.bar(xs + 0.19, regrown, 0.36, color=ORANGE, label="after complete decapitation (H2)",
           yerr=np.array([[m - l[0], l[1] - m] for m, l in zip(regrown, lo)]).T, ecolor=INK2, capsize=2)
    ax.set_xticks(xs, rules)
    ax.set_ylabel("memory M (paired − unpaired twin)")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for x, v in zip(xs + 0.19, regrown):
        ax.text(x, max(v, 0) + 0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    fig.savefig(OUT / "fig2_survival.png")


def fig3_engram_map():
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for r, c in (("S0", BLUE), ("S1", ORANGE)):
        w = R(f"v2_{r}.json")["engram_map"]["windows"]
        xs = [f"{a}-{b - 1}" for a, b in (x["sites"] for x in w)]
        ax.plot(xs, [x["fraction"] for x in w], "-o", color=c, lw=2, ms=5, label=r)
    ax.set_xlabel("4-site window of the trunk receiving the implant (wound edge at left)")
    ax.set_ylabel("fraction of donor memory")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig3_engram_map.png")


def fig4_cycles():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True)
    for ax, r in zip(axes, ("S0", "S1")):
        cyc = R(f"v2_{r}.json")["long_cycles"]["cycles"]
        k = [c["cycle"] for c in cyc]
        ax.plot(k, [c["A"] for c in cyc], "-o", color=BLUE, lw=2, ms=4, label="CS+ = odour A")
        ax.plot(k, [c["B"] for c in cyc], "-o", color=ORANGE, lw=2, ms=4, label="CS+ = odour B")
        b = ROOT / f"results/cycles_B{r[1]}.json"
        if b.exists():
            cb = json.loads(b.read_text())["cycles"]
            other = "B" if r == "S1" else "A"
            ax.plot(k, [c[other] for c in cb], "--o", color=AQUA, lw=2, ms=4, label=f"B{r[1]} rule, odour {other}")
        ax.set_title(f"{r}: memory over repeated decapitation", fontsize=9, loc="left")
        ax.set_xlabel("Promethean cycle")
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[0].set_ylabel("memory M")
    fig.savefig(OUT / "fig4_cycles.png")


def fig5_compiled():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.4), sharey=True)
    for ax, (r, k) in zip(axes, (("S1", "A"), ("S1", "B"))):
        pat = np.array(R(f"v2_{r}.json")["H12"]["patterns"][k])[:, body.HEAD_END:body.BODY_END]
        m = np.abs(pat).max()
        im = ax.imshow(pat, aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest",
                       extent=[body.HEAD_END - 0.5, body.BODY_END - 0.5, 9.5, -0.5])
        ax.grid(False)
        ax.set_title(f"compiled memory, odour {k} ({r})", fontsize=9, loc="left")
        ax.set_xlabel("site")
    axes[0].set_ylabel("hidden channel")
    fig.colorbar(im, ax=axes, shrink=0.9, label="added value")
    fig.savefig(OUT / "fig5_compiled.png")




def fig6_fission():
    """Memory of the posterior half after fission at the trained site (20) and a held-out site (24)."""
    rules = ["S0", "S1", "F0", "F1"]
    a = [R(f"v4_{r}.json")["H19"]["test"]["mean"] for r in rules]
    b = [R(f"v4_{r}.json")["H20"]["test"]["mean"] for r in rules]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    xs = np.arange(len(rules))
    ax.bar(xs - 0.19, a, 0.36, color=BLUE, label="split at site 20 (trained in F)")
    ax.bar(xs + 0.19, b, 0.36, color=ORANGE, label="split at site 24 (held out)")
    for x, v in list(zip(xs - 0.19, a)) + list(zip(xs + 0.19, b)):
        ax.text(x, max(v, 0) + 0.03, f"{v:+.2f}", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(xs, rules)
    ax.set_ylim(-0.2, 1.15)
    ax.set_ylabel("memory M of the posterior half")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(OUT / "fig6_fission.png")


def fig7_robustness():
    rules = ["E0", "E1", "S0", "S1", "B0", "B1", "F0", "F1"]
    tests = [("H25", "cell turnover"), ("H26", "300-step delay"), ("H27", "8-cell neck\nfragment"), ("H28", "8-cell tail\nfragment")]
    M = np.array([[R(f"v5_{r}.json")[h]["test"]["mean"] for h, _ in tests] for r in rules])
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(tests)), [t for _, t in tests], fontsize=8)
    ax.set_yticks(range(len(rules)), rules)
    for i in range(len(rules)):
        for j in range(len(tests)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(M[i, j]) > 0.6 else INK)
    ax.set_title("memory M after each held-out challenge", fontsize=9, loc="left")
    fig.savefig(OUT / "fig7_robustness.png")


def fig8_generations_tradeoff():
    v6, ph = R("v6.json"), R("posthoc_v6.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))
    ax = axes[0]
    for r, c in (("F0", BLUE), ("F1", ORANGE), ("S0", AQUA)):
        y = v6["H29"][r]["by_generation"]
        ax.plot(range(1, len(y) + 1), y, "-o", color=c, lw=2, ms=5, label=r)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("generation of fission (posterior half each time)")
    ax.set_ylabel("memory M")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("memory across generations", fontsize=9, loc="left")
    ax = axes[1]
    rules = list(ph["basins"])
    xs = [ph["basins"][r]["area"] for r in rules]
    ys = [R(f"v5_{r}.json")["H25"]["test"]["mean"] for r in rules]
    ax.scatter(xs, ys, s=40, color=BLUE, zorder=3)
    for r, x, y in zip(rules, xs, ys):
        ax.annotate(r, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("head basin area (noise tolerance, post hoc)")
    ax.set_ylabel("memory after cell turnover")
    ax.set_title(f"a trade-off? rho = {ph['spearman_area_vs_turnover']['rho']:.2f} (post hoc)", fontsize=9, loc="left")
    fig.savefig(OUT / "fig8_generations_tradeoff.png")


def fig9_causal_tradeoff():
    """v0.7: where each sibling sits in the (turnover memory, noise memory) plane. A trade-off would move T
    right and down from C, and N up and left."""
    if not (ROOT / "results/v7.json").exists():
        return
    v7 = R("v7.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), sharex=True, sharey=True)
    col = {"S": INK2, "C": INK, "T": BLUE, "N": ORANGE}
    for s, ax in enumerate(axes):
        pt = {f: (v7["rules"][f"{f}{s}"]["turnover_memory"], v7["rules"][f"{f}{s}"]["noise_memory"]) for f in "SCTN"}
        for f in "TN":
            ax.annotate("", xy=pt[f], xytext=pt["C"], arrowprops=dict(arrowstyle="->", color=col[f], lw=1.8))
        ax.annotate("", xy=pt["C"], xytext=pt["S"], arrowprops=dict(arrowstyle="->", color=GRID, lw=1.2, ls="--"))
        for f, (x, y) in pt.items():
            ax.scatter([x], [y], s=46, color=col[f], zorder=3)
            ax.annotate(f"{f}{s}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8, color=col[f])
        ax.set_xlabel("memory after cell turnover")
        ax.set_title(f"lineage {s}", fontsize=9, loc="left")
    axes[0].set_ylabel("memory after head noise (sigma 1.5)")
    fig.savefig(OUT / "fig9_causal_tradeoff.png")


def fig10_engram_shape():
    """Exploratory: the engram field of intact worms (distance between the hidden states of identical
    twins trained on A and on B, per site) for each v0.7 sibling."""
    if not (ROOT / "results/exploratory_v7_engram.json").exists():
        return
    ex = R("exploratory_v7_engram.json")["rules"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8), sharey=True)
    col = {"S": INK2, "C": INK, "T": BLUE, "N": ORANGE}
    for s, ax in enumerate(axes):
        for f in "SCTN":
            e = ex[f"{f}{s}"]
            ax.plot(range(len(e["profile"])), e["profile"], "-", color=col[f], lw=2 if f != "S" else 1.2,
                    ls="--" if f == "S" else "-", label=f"{f}{s}  (head cv {e['head_cv']:.2f})")
        ax.axvspan(4, 11.5, color=REG[0], alpha=0.08, lw=0)
        ax.set_xlim(2, 30)
        ax.set_xlabel("site (head on the left)")
        ax.set_title(f"lineage {s}", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("A-trained vs B-trained twin,\nhidden-state distance")
    fig.savefig(OUT / "fig10_engram_shape.png")


if __name__ == "__main__":
    for f in (fig10_engram_shape, fig9_causal_tradeoff, fig8_generations_tradeoff, fig2_survival, fig3_engram_map, fig4_cycles, fig5_compiled, fig6_fission, fig7_robustness, fig1_kymograph):
        f()
        print("wrote", f.__name__)
