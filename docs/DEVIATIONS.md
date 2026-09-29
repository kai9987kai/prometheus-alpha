# Deviations from the preregistration

Every departure from `prereg/PREREGISTRATION.json` made after it was hash-locked is listed here with its reason, before the result it could affect was seen where that is possible.

## Design changes made during piloting, before the lock (disclosed, not deviations)

These changes were made while piloting on pilot rules (seeds 100-102) and are listed in the preregistration's `disclosed_pilots`:

1. **Sensing gate.** Pilot rules sensed cues in proportion to `clamp(head, 0, 1)`. After a "complete" decapitation, trunk cells still carried a summed head weight of about 0.27, so a headless body could faintly smell. The gate became `clamp(2*head - 1, 0, 1)`: only cells that are more than half head sense or behave. Residual gated head weight after decapitation is now exactly 0.
2. **H3.** The first H3 blocked gap junctions from conditioning to the cut and tested an interaction. In the pilot the block halved learning itself (0.41 to 0.20), which swamped the question. H3 became two direct causal tests on the decapitated body (voltage swap: necessity; voltage implant: sufficiency). A gap-junction block restricted to the delay, after learning, became exploratory.
3. **S rules.** Trained from scratch, S pilots had not learned the association after 1,100 iterations while E pilots had. S rules are now selected from their E sibling (initialised from its weights, then trained with the S objective), so the E-versus-S contrast isolates what selection for remembering through regeneration adds.
4. **Anticipatory-CR loss.** An acquisition loss (respond to a CS+ before its US once it has been paired) was piloted (E101, S101) and dropped: neither had learned to discriminate after 850 iterations, when the plain objective had. The code remains with weight 0.
5. **Two hypotheses added from the S102 pilot.** H3c (the hidden channels are sufficient) and H4b (a decapitated chimera regrows the body's memory) were added to the confirmatory family after the S102 pilot showed them. On the confirmatory rules they are replications of pilot observations, and the preregistration says so.
6. **Region labels.** Region labels (and so the complete-amputation cut) use a cell's own alpha. The first version used the physics' alive mask, which includes the empty growth frontier; an empty frontier site labelled "head" made the cut remove the whole worm. Found by a unit test before any pilot result used it.

## After the lock

No confirmatory protocol, sample size, seed, test or decision rule was changed after the lock (`prometheus prereg verify` passes in CI). Things done after the lock, for the record:

1. **`cli.py` gained `--init`** (commit a703bd1) so S rules could be trained from the command line. `cli.py` is not a locked source. The E0 and E1 results files record the commit as `-dirty` for this reason; their locked-source hashes verify.
2. **Post-hoc analyses** (`src/prometheus/posthoc.py`, `results/posthoc_*.json`) were written after the confirmatory results showed that chimera and transfer outcomes depended on which odour was involved. They use fresh cohort seeds (2002, 2006) and the locked experiment code unchanged, and every claim drawn from them is labelled "post hoc" in the ledger.
3. **Three predictions failed.** H4 in both S rules (predicted an even split, found a head advantage) and H4b in S0 (predicted the body's memory, found the opposite sign). These are reported as misses, not re-described as successes.

## v0.2

1. **Exploratory crash, fixed and relocked.** The first confirmatory v0.2 run computed every confirmatory hypothesis (H8-H13) and then crashed in the exploratory 8-cycle analysis, which needs more update masks than the v0.1 experiment code allocates (420 steps), so no results file was written. `long_cycles` was changed to allocate its own masks (and to split memory by odour); no confirmatory function changed. The v0.2 lock was regenerated with that change, and the rerun was checked against the crashed run's logged confirmatory means and p-values (`logs/run2_*.crashed.log`, reproduced in `docs/v2_rerun_check.txt`), which it must reproduce exactly because the engine is deterministic.

## v0.6

1. **The lock was committed but not pushed before the run.** The v0.6 lock commit (458217d) was made before the confirmatory run, but its push was rejected: the owner had meanwhile added CODE_OF_CONDUCT.md and SECURITY.md to the branch on GitHub. The run went ahead in the same command before the rejection was noticed. The lock commit precedes the results in the history, and it was pushed immediately after, merged with the owner's commits (f19916f). No protocol was changed.
2. **A post-hoc analysis after H30 failed.** `src/prometheus/posthoc_v6.py` measures the basin in the heads of intact worms instead of at the neck of decapitated ones, and is labelled post hoc everywhere it is quoted.
