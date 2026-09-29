# Prometheus-α

**Does a regenerating body remember what its lost head learned? Synthetic worms grow from one cell, learn which odour predicts a shock, lose their heads, regrow them, and are asked again.**

Author/project lead: **Kai Piper** · Version **0.1.0** · 29 September 2026

> **Scientific status.** Everything here is a synthetic computational experiment on a neural cellular automaton. "Memory", "head", "odour" and "shock" name parts of a simulation. Nothing here is a model of a particular animal, and nothing here bears on any person or clinical condition.

Every day an eagle ate Prometheus's liver and every night it grew back. The name means *forethought*. This laboratory asks a question about regrowth and foresight that biology has raised and never answered: when an animal regrows the organ that learned something, does the new organ know it?

Planarian flatworms trained to find food in a textured, lit dish still show the habit after their heads are cut off and regrown ([Shomrat & Levin 2013](https://doi.org/10.1242/jeb.087809)); an August 2026 preprint reports the same with light-to-food conditioning ([bioRxiv 2026.08.10.741213](https://www.biorxiv.org/content/10.64898/2026.08.10.741213v1)). Moths remember what they learned as caterpillars through the dissolution of metamorphosis ([Blackiston, Silva Casey & Weiss 2008](https://doi.org/10.1371/journal.pone.0001736)). Where the memory waits while the brain is gone, and how the new brain reads it back, is unknown. McConnell's memory-transfer experiments of the 1960s asked the same question and could not be replicated.

Prometheus-α asks it of a synthetic body in which every cell's state can be read, copied, swapped and transplanted:

1. **A body that learns without changing its rule.** One small network, shared by every cell, is trained by backpropagation through whole lives. Within a life the rule is frozen, so a worm can only learn by changing its cells' *state*. That makes "where is the memory?" a question about places in the body.
2. **Only the head can smell.** A cell senses the odours only when it is more than half head tissue. Behaviour is read from the head. After complete decapitation no surviving cell can smell or behave.
3. **Pavlovian conditioning with the proper control.** One odour is paired with a shock. Every paired worm has an *explicitly unpaired twin*: the same odours, the same number of shocks at the same moments of its life, the same random update noise, the same founder; only the contingency differs. The memory measure is taken within the worm, so baseline shifts cancel.
4. **Emergent versus selected.** *E* rules are trained to learn and to regenerate, but never asked to remember through a regeneration. *S* rules are selected from their E sibling by also asking for memory after one amputation. Chimeras, voltage transplants, trunk fragments and repeated amputation are never trained in either.

### What v0.1 found (four rules, 128 worms per test, preregistered)

* **Without selection, the new head forgets.** The emergent rules learn (M = 0.80 and 0.73, where a full conditioned response is about 1). After complete decapitation they regrow near-perfect heads, and those heads carry none of the memory: M = 0.001 and −0.005, with both 95% intervals inside ±0.01. In chimeras the head's memory wins outright. A body that can rebuild a head does not, by default, keep a copy of what that head knew.
* **Selection makes the memory outlive its organ, and the ability generalises.** S rules were selected only to remember through one amputation, and some of those training cuts spared a few head cells. They keep the memory through complete decapitation (retention 0.98 and 1.02). They also keep it through removal of both head and tail (0.96, 0.96) and through a third successive decapitation (0.94, 0.73). Neither of those was ever trained.
* **The body's copy is not electrical.** Voltage is the channel gap junctions couple, and the one the bioelectric literature points to. It is neither necessary (swapping it for an untrained twin's voltage keeps 98–99% of the memory) nor sufficient (implanting it transfers 2–4%). The ten hidden channels, written alone into an untrained twin's headless body, transfer 1.00 and 0.98 of the memory. The twin's new head then remembers what it never learned.
* **A trained body teaches a naive head.** A naive head grafted onto a trained body takes on about half of the body's memory (0.51, 0.47) with no amputation at all.
* **What we got wrong, and what that revealed.** From the S102 pilot we predicted two things. First, that head and body would split an intact chimera's memory evenly: in fact the head leads (dominance 0.84, 0.53). Second, that a decapitated chimera would regrow the body's memory: this held only weakly in one rule (−0.12) and reversed in the other (+0.36), against the pilot's −0.66. Reading the grafts afterwards showed why. This part is post hoc, not preregistered. Selection made each rule's memory of one odour an **attractor** and its memory of the other **metastable**. Conflicts resolve toward the favoured odour whichever tissue carries it. Transfer passes on the favoured odour and hardly the other. Across repeated decapitations the favoured memory holds (S1: 0.96 after cycle 1, 0.96 after cycle 4) while the other erodes (0.97 to 0.32). The two S rules favour different odours: S0 favours B, S1 favours A. The E rules show no such bias.
* **Restraint.** Ten of the 40 confirmatory tests passed Holm with effects below 0.04, and none of them is claimed. Across 160 A/A tests the false-positive rate was 5 in 160.

Predictions made in the locked preregistration, against the outcomes:

| | hypothesis | predicted E / S | E0 | E1 | S0 | S1 |
|---|---|---|---|---|---|---|
| H1 | learns, intact | yes / yes | yes (+0.80) | yes (+0.73) | yes (+0.99) | yes (+0.93) |
| H2 | remembers after complete decapitation | no / yes | no (+0.00) | no (-0.00) | yes (+0.97) | yes (+0.95) |
| H3a | voltage is necessary | – / no | no (+0.00) | no (-0.00) | no (+0.02) | no (+0.01) |
| H3b | voltage is sufficient | – / no | no (+0.00) | no (-0.00) | no (+0.04) | no (+0.02) |
| H3c | hidden channels are sufficient | – / yes | no (+0.00) | no (-0.00) | yes (+0.99) | yes (+0.93) |
| H4 | intact chimera is not an even split | yes / no | yes (+0.99) | yes (+0.99) | yes (+0.84) ✗ | yes (+0.53) ✗ |
| H4b | decapitated chimera regrows the body's memory | no / yes | no (+0.00) | no (-0.01) | no (+0.36) ✗ | yes (-0.12) |
| H5 | a naive head takes the body's memory | no / yes | no (+0.00) | no (+0.00) | yes (+0.51) | yes (+0.47) |
| H6 | third Promethean cycle | no / yes | no (+0.00) | no (+0.00) | yes (+0.94) | yes (+0.73) |
| H7 | trunk fragment | no / yes | no (+0.00) | no (-0.00) | yes (+0.96) | yes (+0.96) |

31 of 34 preregistered predictions held (✗ marks a miss; – means no prediction was stated). The number in brackets is the tested mean. "yes" requires Holm-corrected significance and an effect of at least 0.10.

Every number in the results table is checked in CI against the results file it comes from (`prometheus claims check`). The hypotheses, sample sizes, seeds and decision rules were hash-locked in [`prereg/PREREGISTRATION.json`](prereg/PREREGISTRATION.json) before any confirmatory experiment; pilots and pre-lock design changes are disclosed there and in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## Built from eight earlier projects

| Earlier project | What Prometheus-α takes from it |
|---|---|
| [Morpheus](https://github.com/kai9987kai/morpheus) | the regenerating neural cellular automaton with a fixed gap-junction voltage law; "does a rewritten state survive re-amputation?" (Morpheus's compiled heads did not); the claims-ledger checker, statistics and JS/Python parity testing. Morpheus asked whether a *body plan* is remembered; this asks whether a *learned association* is |
| [Ghost in the Machine](https://github.com/kai9987kai/GhostInTheMachine) | the yoked-twin design (master vs a twin with matched input statistics), transplanting a signal between individuals to test where an effect lives, preregistration with a hash lock, A/A calibration, and a smallest effect of interest so a precise but trivial effect cannot pass |
| [GenesisEngine](https://github.com/kai9987kai/GenesisEngine) | development from a single founder cell; a deterministic seeded core separate from the renderer; paired seeds per intervention |
| [Supermix Expanse](https://github.com/kai9987kai/Supermix-expanse) / [v2](https://github.com/kai9987kai/Supermix-Expanse-v2) | the "evoked" lesson: the CNS core's gain was mostly a constant bias, so the readout here is a within-worm difference (CS+ minus CS-) that a bias cannot fake; zero-initialised action heads; hash-locked hypotheses with verdicts computed from recorded metrics |
| [Supermix Archimedes](https://github.com/kai9987kai/supermix-archimedes) | grafting between independently trained systems, made literal: chimeras are grafts of one worm's head onto another's body, and the question is which system's knowledge wins |
| [Supermix](https://github.com/kai9987kai/Supermix) | "the ruler moved": test generators, caps and cohorts are fixed and seeded, and every results file records the weights' and the preregistration's SHA-256 and the git commit |
| [FLY-DIAMOND-NEXUS](https://github.com/kai9987kai/FLY-DIAMOND-NEXUS) | a broadcast unconditioned signal to every cell, like dopamine to the mushroom body; paired-seed arms with bootstrap intervals; exact, seed-reproducible replay in the browser |

## What is new here

To my knowledge none of these has been done before:

* **A computational Shomrat–Levin experiment.** A learned association formed only by head tissue, followed by complete decapitation and regeneration, in a body whose every cell state is observable.
* **Emergent versus selected memory persistence.** The same rule, before and after selection for remembering through one regeneration, tested on held-out manipulations it never saw.
* **Memory chimeras.** The head of a worm that learned A grafted onto the body of a worm that learned B: whose memory does the animal express, and whose comes back after the chimera's head is cut off?
* **Memory transplant by state.** Copying one channel group of a trained, decapitated body (its voltage, or its hidden channels) and nothing else into an untrained twin's body, then asking the twin's regrown head.
* **Promethean cycles.** Repeated decapitation of the same worm, far beyond anything trained.

## How it works

```mermaid
flowchart LR
    subgraph cell["every cell, every step (one shared rule, 4,368 parameters)"]
      P["perceive self and 2 neighbours<br/>16 channels x (identity, gradient, Laplacian)"] --> N[MLP 51 → 64 → 16]
      S["odour A, odour B<br/>(sensed only if > half head)"] --> N
      U["shock (US)<br/>broadcast to every living cell"] --> N
      N --> A["change own 16 channels<br/>(each cell fires with p 0.5)"]
    end
    A --> G["gap-junction voltage diffusion<br/>(fixed law) and alive masking"]
    G --> B["behaviour = response channel<br/>read from head cells"]
```

**The worm.** 40 sites; the body occupies 32 of them (head 8, trunk 16, tail 8). Each cell has 16 channels: alive-ness, membrane voltage, three region identities, a response channel and 10 hidden channels with no target. A worm grows for 32 steps from one founder cell.

**A life in an experiment.** Grow 32 steps; conditioning, 48 steps (six 8-step slots in random order: two with the CS+, two with the CS-, two empty; the paired worm is shocked in the last 3 steps of each CS+, its unpaired twin in the middle of each empty slot); wait 12 steps; then the experiment's manipulation (decapitation, a graft, a voltage transplant); regenerate 32 steps; test each odour alone for 6 steps. The memory measure is `M = [R(CS+) - R(CS-)] of the worm - the same of its unpaired twin`.

**Training.** Exact backpropagation through lives of about 180 steps (PyTorch, CPU), batches of 32 random lives (paired, unpaired or naive; with or without a first wound; with or without a wound after conditioning). The loss scores anatomy at four checkpoints, the unconditioned response to the shock, a silent baseline, and the memory test. E rules score memory only when the body was not wounded after conditioning; S rules also score it after a head or tail cut.

## Results

<!-- claims:start (generated by `prometheus claims render`; edit claims/claims.json) -->

| claim | status | evidence |
|---|---|---|
| **Selected rules remember through complete decapitation: the regrown head, built from cells that never smelled the odours, knows which one predicted the shock** | **replicated** | memory M in the regrown head (paired minus unpaired twin, 128 worms): S0 0.97 (retention 0.98 of the intact memory, d_z 8.40, Holm p 0.001); S1 0.95 (retention 1.02, d_z 8.14, Holm p 0.001). Positive control: S rules were selected for one regeneration |
| **Emergent rules also remember through decapitation** | **not supported** | as predicted, no: E rules regrow near-perfect heads (IoU E0 0.995, E1 0.999) that carry none of an intact memory of 0.81 / 0.74: M after regeneration E0 0.001 [95% CI -0.000, 0.001], E1 -0.005 [-0.006, -0.004], both intervals far inside the smallest effect of interest (0.10) |
| **The memory in the body is carried by its hidden (non-electrical) channels: writing only those into an untrained twin's headless body makes the twin's new head remember** | **replicated** | M in the implanted twin's regrown head: S0 0.99 (1.00 of the donor's own), Holm p 0.001; S1 0.93 (0.98), Holm p 0.001. Predicted from pilot S102 (0.85) |
| **Promethean cycles: the memory survives a third successive decapitation (only one was ever trained)** | **replicated** | M after cycles 1, 2, 3, 4: S0 0.99, 0.99, 0.94, 0.90; S1 0.96, 0.89, 0.73, 0.63. Body IoU after cycle 4: S0 1.00, S1 0.97 |
| **Post hoc: selection made one odour's memory an attractor and the other metastable. Conflicts, transfers and repeated regrowth all resolve toward the rule's favoured odour** | **post hoc** | odour bias of conflicting chimeras (R(A) - R(B) averaged over both grafts; > 0 favours A) intact / regrown: S0 -0.13 / -0.49, S1 0.36 / 0.76; E rules regrown 0.01, -0.07. Memory for each CS+ after Promethean cycle 1 and 4 (fresh cohort): S1 favoured A 0.96 to 0.96, other B 0.97 to 0.32; S0 favoured B 0.95 to 0.96, other A 0.99 to 0.86. Not preregistered; found by reading the H4, H4b and H5 grafts |
| Memory chimeras: cut the head off a chimera whose head learned one odour and whose body learned the other, and the regrown head remembers the body's | **mixed** | predicted from the pilot, and not borne out: dominance after regeneration (+1 = the head's memory, -1 = the body's) S0 0.36 (Holm p 1.000), S1 -0.12 (Holm p 0.001), far weaker than the pilot's -0.66. What the regrown head remembers is mostly the rule's favoured odour (see the post-hoc claim) |
| The memory is bioelectric: the body's voltage pattern is necessary and sufficient for it | **not supported** | as predicted, no. Swapping the trained body's voltage for its twin's kept S0 0.98 / S1 0.99 of the memory (H3a); implanting the voltage alone into the twin transferred S0 0.04 / S1 0.02 (H3b). Gap-junction diffusion is a fixed law here, and the rules did not use it to hold the memory |
| In an intact chimera of an emergent rule, the head's memory wins outright | **replicated** | dominance of the head's memory after 32 steps of healing: E0 0.99, E1 1.01 |
| In an intact chimera of a selected rule, head and body do not split the memory evenly | **replicated** | dominance S0 0.84, S1 0.53 (predicted from the pilot: an even split). The head leads, but each rule also favours one odour in a conflict: R(A) - R(B) of A-head|B-body and B-head|A-body grafts, S0 0.71 and -0.97, S1 0.89 and -0.17 |
| A naive head grafted onto a trained body takes on the body's memory, with no amputation or regeneration | **replicated** | transfer (half the difference between naive heads on A- and B-trained bodies): S0 0.51 (0.51 of the uncontested memory, Holm p 0.001); S1 0.47 (0.48, Holm p 0.001). Post hoc, it is all or none by odour: S0 passes on B (1.00) but not A (0.03); S1 passes on A (0.78) far more than B (0.16) |
| A trunk fragment, with head and tail both removed, regrows a head that remembers (never trained) | **replicated** | M: S0 0.96, S1 0.96; body IoU after regeneration 1.00, 1.00 |
| Worms learn which odour predicted the shock, in their state alone (the rule is frozen within a life) | **replicated** | M in intact worms (paired minus unpaired twin, 128 worms): E0 0.80 (d_z 3.8), E1 0.73 (3.9), S0 0.99 (12.7), S1 0.93 (9.7) |
| Significant but trivial effects are not claims | **descriptive** | 10 of 40 confirmatory tests passed Holm but moved the response by less than the smallest effect of interest (0.10), so none is claimed: E0 H3a 0.000, E0 H3b 0.001, E0 H5 0.004, E0 H7 0.001, E1 H4b -0.006, E1 H5 0.003, S0 H3b 0.037, S0 H3a 0.020, S1 H3a 0.012, S1 H3b 0.023 |
| Exploratory: which channels, implanted alone into the twin's headless body, carry the memory | **exploratory** | fraction of the donor's memory: S0 voltage 0.05, hidden 0.99, identity+response 0.00, all but voltage 0.99; S1 0.02, 0.97, -0.00, 0.99 |
| Exploratory: blocking gap junctions for the delay between learning and the cut does not stop the memory reaching the body | **exploratory** | M in the regrown head, junctions open vs blocked: S0 0.99 vs 0.98, S1 0.96 vs 0.96 |
| The pipeline is calibrated: A/A comparisons produce false positives at about the nominal rate | **supported** | false-positive rate at alpha 0.05 over 40 A/A tests of 128 worms each (nominal 0.05, i.e. 2 of 40): E0 0.025, E1 0.000, S0 0.075, S1 0.025 |
| All four rules grow the body from one founder cell and regrow it after decapitation | **descriptive** | body IoU with the target after 32 steps of growth / after head regeneration / 300 steps later: E0 0.90 / 1.00 / 1.00; E1 0.91 / 1.00 / 1.00; S0 0.91 / 1.00 / 1.00; S1 0.90 / 1.00 / 0.95 |

<!-- claims:end -->

## Quick start

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
prometheus show --weights weights/rule_S0.json --cycles 2    # ASCII: grow, learn, lose the head, regrow, ask
python -m pytest -q                                           # engine invariants, JS/Python parity, statistics
prometheus prereg verify                                      # the preregistration lock still holds
prometheus claims check                                       # every quoted number against the results files
```

What `prometheus show` prints for one worm of each kind (H head, = trunk, t tail; response to the CS+ and CS-):

```text
selected rule S0                   |    HHHHHHHH================tttttttt    |
  conditioned (paired, CS+ = A)       response to CS+ +0.96   CS- -0.01   DI +0.97
  cycle 1: head removed            |              ==============tttttttt    |
  cycle 1: regenerated             |    HHHHHHHH================tttttttt    |
                                      response to CS+ +1.00   CS- -0.01   DI +1.01
emergent rule E0
  conditioned (paired, CS+ = A)       response to CS+ +0.96   CS- +0.03   DI +0.92
  cycle 1: regenerated             |    HHHHHHHH================tttttttt    |
                                      response to CS+ +0.15   CS- +0.12   DI +0.03
```

Open **`web/index.html`** in a browser for the interactive lab: condition a worm, cut its head off, regrow it, graft on a head from a worm that learned the other odour, block gap junctions, and run Promethean cycles, with a live kymograph of every channel. It needs no server or build step. `prometheus export-web` refreshes `web/data.js`.

Reproduce everything (about 90 minutes on a 4-core laptop CPU, one thread per run):

```bash
prometheus train --family E --seed 0 --iterations 3000 --out weights/rule_E0.json      # ~32 min
prometheus train --family E --seed 1 --iterations 3000 --out weights/rule_E1.json
prometheus train --family S --seed 0 --iterations 2000 --init weights/rule_E0.json --out weights/rule_S0.json   # selection, ~21 min
prometheus train --family S --seed 1 --iterations 2000 --init weights/rule_E1.json --out weights/rule_S1.json
for r in E0 E1 S0 S1; do prometheus run --weights weights/rule_$r.json --out results/rule_$r.json; done   # ~1 min each
python -m prometheus.posthoc                        # post-hoc per-odour analyses (not preregistered)
python tools/build_claims.py && prometheus claims render && prometheus export-web
```

Runs are deterministic for a given PyTorch version: every founder, schedule and update mask comes from a recorded seed. `prometheus run` labels its output CONFIRMATORY only while the preregistration lock verifies, and each results file records the weights' and the preregistration's SHA-256 and the git commit.

## Repository map

| Path | What |
|---|---|
| `src/prometheus/tissue.py` | the cell rule, perception, sensory gate, gap-junction physics, behaviour readout |
| `src/prometheus/body.py` | the target body plan, wounds, anatomy scores |
| `src/prometheus/life.py` | timelines: conditioning (paired, unpaired, naive), tests, wounds, the runner |
| `src/prometheus/train.py` | backpropagation through whole lives; the E and S objectives |
| `src/prometheus/experiments.py` | H1-H7 (with H3a-c and H4b), calibration, anatomy, the channel-locus and gap-junction explorations |
| `src/prometheus/posthoc.py` | post-hoc analyses written after the results were seen (per-odour survival, transfer, chimera bias, cycles) |
| `tools/build_claims.py` | builds `claims/claims.json` so that every quoted number is a pointer into a results file |
| `src/prometheus/stats.py` | paired bootstrap, sign-flip tests, Holm |
| `src/prometheus/claims.py`, `prereg.py` | the claims ledger and the preregistration lock |
| `prereg/` | hypotheses, seeds and decision rules, and their SHA-256 lock |
| `results/` | every results file the README quotes, with provenance; `results/pilot/` holds the disclosed pilots |
| `weights/` | the four confirmatory rules (`weights/pilot/` the pilot rules) |
| `web/` | the browser lab (`index.html`) and the JavaScript engine (`engine.js`), tested step for step against Python |

## Limitations

* Two rules per family. Whether a result holds for rules in general is tested only by its replication in the second rule.
* The body is one-dimensional and 32 cells long, the rule is small, and the "odours" and "shock" are two input lines and a broadcast scalar. The biology motivates the questions; it is not modelled.
* "Memory" here is operational: a within-worm difference in response to two cues, measured against an explicitly unpaired twin.
* S rules were selected to remember through one regeneration, with training cuts at the head/trunk boundary ±2 sites. Their survival of one decapitation is a positive control, not a discovery. What is held out is everything else: complete decapitation with a margin, fragments, chimeras, voltage transplants and repeated cycles.
* "Not electrical" is a statement about these rules. Gap-junction diffusion here is a fixed, passive law and the hidden channels are free, so the training had an easy non-electrical route. Real bioelectric memories involve active, bistable ion-channel dynamics that this model does not have. The result shows that a regenerating body *can* keep the copy outside its voltage, not that bodies do.
* The favoured-odour attractor is post hoc, seen in two rules that favour different odours. Why selection breaks the symmetry between two interchangeable odours, and whether a rule could be selected to hold both equally, are open questions for v0.2.
* Grown worms reach a body IoU of about 0.90 by step 32 and 1.00 by step 300. Conditioning begins while the head is still finishing its growth.

## Research context

* Shomrat, T. & Levin, M. (2013). An automated training paradigm reveals long-term memory in planarians and its persistence through head regeneration. *J Exp Biol* 216:3799–3810.
* Trained planaria retain memories through head regeneration (bioRxiv, August 2026, 10.64898/2026.08.10.741213).
* Blackiston, D. J., Silva Casey, E. & Weiss, M. R. (2008). Retention of memory through metamorphosis. *PLoS ONE* 3:e1736.
* Mordvintsev, A. et al. (2020). Growing neural cellular automata. *Distill*.
* Pezzulo, G. & Levin, M. (2016). Top-down models in biology. *J R Soc Interface* 13:20160555; Durant, F. et al. (2017). Long-term, stochastic editing of regenerative anatomy via targeting endogenous bioelectric gradients. *Biophys J* 112:2231–2243.
* Guichard, E. et al. (2025). EngramNCA: a neural cellular automaton model of memory transfer. arXiv:2504.11855 (morphological "genes" in private channels; no learning within a life, no regeneration test).
* Pio-Lopez, L., Hartl, B. & Levin, M. (2026). BraiNCA: brain-inspired neural cellular automata. arXiv:2604.01932.
* Rescorla, R. A. (1967). Pavlovian conditioning and its proper control procedures. *Psychol Rev* 74:71–80. Rescorla's caveat about the explicitly unpaired control is that its CS can become a safety signal; here both odours are equally unpaired in the twin, so any such inhibition cancels inside the twin's own CS+ minus CS- difference.

## License

MIT, see [LICENSE](LICENSE).
