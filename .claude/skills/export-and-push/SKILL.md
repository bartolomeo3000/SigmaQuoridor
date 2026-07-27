---
name: export-and-push
description: Re-export ONNX models from the current best.pt checkpoints and push the model artifacts (ONNX files + best.pt + training_stats.csv) to origin, without touching self-play data or checkpoint history. Use when the user asks to "export onnx", "push the model", or "sync the model to the repo/frontend" after a training run.
---

# Export ONNX and push model artifacts

This repeats the exact procedure used to publish a freshly-trained model to both the
Python app and the JS/ONNX browser frontend (`docs/`), without dragging along the
large, fast-churning self-play data (`data_*/*.npz`) or per-cycle training checkpoints
(`models_*/checkpoints/*.pt`) — those are handled separately and should NOT be added here
unless the user explicitly asks for them in a given run.

## Steps

0. **Regression gate — ALWAYS run this first, before exporting anything.** The training
   loop has **no promotion gate**: `best.pt` is overwritten every cycle and just means
   "latest," not "strongest." A drifted/regressed checkpoint has already been pushed to
   the live site once this way. So before publishing, confirm the model about to be
   exported does **not regress** against the one currently live on the site.

   **First ask the user whether they've already checked this** (e.g. already ran a
   tournament vs the previous best). If they say yes, skip straight to step 1. Otherwise
   run at least a small head-to-head:

   ```
   # Recover the currently-LIVE model (the previously exported best.pt) from git,
   # so we compare against exactly what's on the site right now:
   git show HEAD:runs/models_9x9_heads/best.pt > /tmp/prev_exported_best.pt

   # Small regression tournament: new best.pt vs the previously-exported model.
   # (--dir . globs no checkpoints at repo root, so only the two --extra models play.)
   .venv/Scripts/python tournament_cpp.py --dir . \
     --extra runs/models_9x9_heads/best.pt \
     --extra /tmp/prev_exported_best.pt \
     --games 100 --temp 0.5 --sims 800 --boardsize 9 --walls 10 \
     --threads 7 --parallel 512 --max-batch 512
   ```

   - Proceed to export **only if the new model is at least on par** (roughly ≥50% score,
     i.e. no clear regression). If it's clearly worse, STOP and tell the user — the fix is
     usually to promote a stronger earlier checkpoint to `best.pt` instead of exporting the
     latest one, then re-run this gate.
   - Scale games up (200–300/pair) if the 100-game result is close/ambiguous.
   - Adjust `--parallel`/`--max-batch` down if RAM is tight while the user is on the machine.
   - `tournament_cpp.py` does not checkpoint partial results — if it's killed, it must be
     rerun from scratch.

1. **Export ONNX.**
   ```
   .venv/Scripts/python export_onnx.py
   ```
   This regenerates `docs/models*/best.onnx` and the curated checkpoint-history ONNX
   files (see `export_onnx.py`'s `CKPT_STEP`) from whatever `best.pt` currently exists
   in each lineage's model dir.

2. **Check what changed.**
   ```
   git status --short docs/ runs/models_9x9_heads/best.pt runs/models_9x9_heads/training_stats.csv runs/models_9x9/best.pt runs/models_9x9/training_stats.csv runs/models_7x7/best.pt runs/models_7x7/training_stats.csv
   ```
   Only lineages with an actual training update will show diffs — that's expected and fine.

3. **Watch for filename collisions across lineages.** `export_onnx.py` picks checkpoint
   ONNX files by cycle number modulo `CKPT_STEP`, and different lineages can produce the
   same filename (e.g. `cycle_0061.onnx`) for what is actually a *different* underlying
   network. Before committing, grep `docs/app.js` for any hardcoded picker entries
   that reference a filename you're about to overwrite:
   ```
   grep -n "cycle_0" docs/app.js
   ```
   If a modified/new file's name matches a picker entry that used to point at a
   *different* lineage's history, flag it to the user and ask how to proceed (accept the
   overwrite, rename to avoid collision, or skip that file) — don't decide silently, this
   happened once already (runs/models_9x9_heads export overwrote models_9x9-legacy picker
   entries) and the resolution depends on whether the user still wants that old history
   browsable.

4. **Stage only the model-artifact files** — never use `git add -A`/`.` for this. Stage
   exactly:
   - `docs/**/*.onnx` files that actually changed (from `git status` in step 2)
   - `models_<lineage>/best.pt` (the one(s) that changed)
   - `models_<lineage>/training_stats.csv` (the one(s) that changed)

   Do NOT stage:
   - `data_*/*.npz` (self-play data — large, handled elsewhere)
   - `models_*/checkpoints/*.pt` (per-cycle training checkpoints — large, not needed for
     serving)

5. **Commit** with a short message naming the lineage and cycle number reached, e.g.
   `export onnx from latest runs/models_9x9_heads best.pt (cycle N)`. If step 3 surfaced a
   collision that the user asked to accept, mention that in the commit body.

6. **Push** to the current branch's upstream (`git push origin <branch>` — this repo
   serves its frontend from `gh-pages`, so that's usually the branch in play; confirm
   with `git branch --show-current` if unsure).

## Notes

- This is a repeatable, low-risk push (model binaries + one CSV) — no source code
  changes are involved. Still, if `git status` after export shows anything unexpected
  (e.g. a lineage you didn't expect to have changed, or source files modified), stop and
  check with the user before committing.
- If `export_onnx.py`'s `EXPORTS` list or lineage defaults have changed since this skill
  was written, re-read the script rather than trusting the exact paths above.
