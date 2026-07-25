---
name: tournament-add-cycle
description: Add one or more new checkpoints to a versioned Elo tournament series in tournaments/<name>/ (e.g. scratch_vs_heads), reusing the previous version's games so only the new pairings are played. Use when the user asks to "add the newest cycle to the tournament", "extend the tournament", "run the next tournament version", or "add cycle N to our tournament".
---

# Add a checkpoint to a tournament series

Extends a versioned round-robin Elo tournament. The active series is
`scratch_vs_heads`, which tracks the fresh `models_9x9_scratch` run against the old best
(`models_9x9_heads/best.pt`). Each version `vN` adds newer checkpoint(s) and **reuses every
game already played in `vN-1`**, so only pairings involving the new checkpoint(s) are simulated.

`tournament_cpp.py --series` does all the path/config plumbing. Everything a follow-up run
needs is already recorded in the previous version's files, so it reads them back instead of
making you retype anything:

| read from | what it gives |
|---|---|
| `vN.csv` (`model` column) | the roster to carry over, and games/pair |
| `vN_matchups.csv` | the `--baseline` to reuse |
| `vN_matchups.csv.meta.json` | the rules config (sims/temp/c_puct/fpu/boardsize/walls/max_moves) |

## Steps

1. **Find the newest checkpoint** not already in the series:
   ```
   ls -1 runs/models_9x9_scratch/checkpoints/ | tail -5
   cut -d, -f2 runs/tournaments/scratch_vs_heads/v*.csv | grep -o 'cycle_[0-9]*' | sort -u | tail -5
   ```

2. **Run it** — one line, no hand-written paths:
   ```
   .venv/Scripts/python tournament_cpp.py --series scratch_vs_heads \
     --add runs/models_9x9_scratch/checkpoints/cycle_0350.pt
   ```
   Repeat `--add` to insert several at once. Long-running: run it in the background.

   Anything passed explicitly overrides what's inherited (e.g. `--sims 400`) — but don't,
   unless the user asked: a config that differs from the sidecar makes the baseline
   incompatible, and **every** pair gets replayed instead of reused.

3. **Verify baseline reuse before letting it grind.** The startup log must print:
   ```
   Series 'scratch_vs_heads': extending v7 -> v8  (14 model(s) inherited, 1 added)
   Baseline ...v7_matchups.csv: reused 91 pair(s) (45500 games) — 14 pair(s) left to simulate.
   ```
   - `reused` should be `C(old_roster_size, 2)`; `left` should be `old_roster_size × (models added)`.
   - If it says **"no fully-matching pairs found"**, or reuses fewer than expected, **STOP** —
     something made the config or roster names incompatible. Don't let it replay everything.

4. **Report the updated Elo table** from the new `vN+1.csv` (ranked), noting where the added
   cycle landed relative to the previous top scratch cycle.

5. **Report the added model's head-to-head** records. The summary CSV only has aggregate score
   across all opponents; per-opponent h2h must come from the matchups CSV:
   ```
   .venv/Scripts/python - "$NEW" "$OPP" runs/tournaments/scratch_vs_heads/vN+1_matchups.csv <<'PY'
   import csv, sys
   new, opp, path = sys.argv[1], sys.argv[2], sys.argv[3]
   s = n = w = d = l = 0.0
   for r in csv.DictReader(open(path, newline="")):
       a, b = r["agent_a"], r["agent_b"]
       if {a, b} != {new, opp}: continue
       sc = float(r["score_for_a"]) if a == new else 1.0 - float(r["score_for_a"])
       s += sc; n += 1
       w += sc == 1.0; d += sc == 0.5; l += sc == 0.0
   print(f"{new} vs {opp}: {s}/{int(n)} = {100*s/n:.1f}%  (W{int(w)} D{int(d)} L{int(l)})" if n
         else f"{new} vs {opp}: no games recorded")
   PY
   ```
   Opponents to report, in priority order:
   - **`models_9x9_heads\best.pt`** (the old best) — the canonical "has the scratch run
     surpassed the old model yet" number, and the milestone this series tracks (crossing 50%).
     Report it **only if it's in the roster**; not every future series will include it, so if
     it's absent say so briefly and skip it, don't error.
   - **The top 1–2 other models** in the new Elo table, so the h2h is anchored to the current
     strongest entries whether or not the old model is present.

   Note roster names use backslashes (`models_9x9_scratch\checkpoints\cycle_0321.pt`) — match
   them exactly as stored in the CSV.

## Notes

- **Starting a new series**: `--series <newname>` with an empty/absent directory starts at v1.
  There's nothing to inherit then, so pass the roster (`--extra`/`--add`) and full config
  explicitly that first time; every later run inherits it.
- **Paths**: series live in `runs/tournaments/<name>/` (override the root with `--series-dir`).
  Ad-hoc runs with no `--series` and no `--out` land in `runs/tournaments/adhoc/results.csv`.
- **Resume**: results are checkpointed after each completed pair to a `.progress.json` keyed
  by a roster/config signature, so a kill/crash loses at most the one in-flight pair. Re-run
  the *same* command to resume — it prints `Resuming from ... N/M pairs already complete`.
  (On a resume the baseline line may then say "no fully-matching pairs found"; that's expected
  and harmless — the progress file already absorbed those pairs.)
- **Don't delete checkpoints that are in a series roster.** Baseline reuse matches on model
  path; if a roster checkpoint is gone, its pairs can't be reused or replayed.
- Pairs are played sequentially (1–2 models hot at a time) — worth it only at 100s of games
  per pair, which is this series' regime. Not for small smoke scans.
- If `build_roster`, `_apply_series`, the `--baseline` matcher, or the `.meta.json` schema
  change, re-read `tournament_cpp.py` rather than trusting the details above.
