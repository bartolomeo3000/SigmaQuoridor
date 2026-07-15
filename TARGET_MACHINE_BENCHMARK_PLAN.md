# Target-machine benchmark plan (Ryzen 8c/8t + RTX 5070 Ti, 32GB RAM)

Everything below was designed/validated functionally on an M1 MacBook Air.
None of the throughput numbers from that machine should be trusted for
tuning — MPS has very different batching/latency characteristics than
CUDA. This doc is the checklist for re-establishing real numbers once
on the target PC.

## 0. Setup

```bash
git pull                                   # get cpp/, selfplay_cpp.py, etc.
pip install -r requirements.txt            # includes pybind11 now
python setup_cpp.py build_ext --inplace
```

CUDA-specific: confirm `torch.cuda.is_available()` is True before anything
else — `selfplay_cpp.py` silently falls back to cpu/mps otherwise.

## 1. Correctness (must pass before any benchmarking)

```bash
python test_cpp_parity.py --games 50
```
Expect: all three boardsizes (5x5, 7x7, 9x9) pass, no assertion errors.
This only depends on the engine, not the GPU — should be identical to
the Mac result (8915+ moves compared, all matching).

## 2. Engine-only raw speed (sanity check, no NN)

```bash
python -c "
import quoridor_cpp as q
d = q.random_playouts(num_games=5000, boardsize=7, walls=5, seed=1)
print(d)"
```
Expect: moves/s/core in the same ballpark as the M1 (~1.3M/s) or higher
(Ryzen single-core perf is comparable/better than M1 for this kind of
branchy integer code). This isolates "is the C++ build itself fine on
this machine" from anything GPU/threading related.

## 3. Thread count sweep

Question: does `--threads 8` (all cores) meaningfully beat `--threads 7`
(leave one for the Python inference loop + OS), or does it just make the
machine less responsive for the same throughput?

```bash
for t in 4 6 7 8; do
  echo "=== threads=$t ==="
  python selfplay_cpp.py --games 128 --sims 64 --threads $t \
      --parallel 128 --max-batch 256 --model models_7x7_v2/best.pt
done
```
Record: games/hour, evals/s, mean batch, and subjectively whether the
desktop feels sluggish during the run.

## 4. Batch/parallelism sweep (the big one)

Goal: find the `(parallel_games, leaf_batch, max_batch)` combination that
maximizes evals/s and keeps mean batch near `max_batch` for most of a run,
at the sim count you'll actually train with (likely 400-800).

Don't take `max_batch=256` (or even 512) for granted as a ceiling — that
was just what the M1 exploration happened to use, not a tuned value. The
RTX 5070 Ti has far more compute and memory (16GB) than the M1's
integrated GPU, and this model is small (filters=64, res=6), so bigger
batches likely amortize kernel-launch/dispatch overhead better and could
push evals/s meaningfully higher. Extend the grid with larger max_batch
(1024, 2048) and more parallel games (2048) if the smaller values in the
table below are still evals/s-scaling (i.e. haven't plateaued) — only
stop increasing once evals/s stops improving or GPU memory becomes a
constraint (watch `nvidia-smi` for both compute utilization and memory).

This sweep fixes `--leaf-batch 1` throughout — batching multiple leaves
per simulation trades search quality for speed (each batched leaf is
evaluated using slightly stale statistics from virtual loss instead of
the fully backed-up tree), so it's a different question from pure
engine/batching throughput and shouldn't be conflated with it here.
Parallel games alone (no leaf batching) already gives plenty of
batching opportunity for the GPU. See step 6 for the leaf_batch>1
quality-vs-speed tradeoff, evaluated properly (holdout/matchups, not
just throughput).

Grid to try (fix `--sims 800`, `--leaf-batch 1`, `--threads` = whatever
won step 3):

| parallel | max_batch |
|---|---|
| 128       | 256 |
| 256       | 256 |
| 256       | 512 |
| 512       | 512 |
| 1024      | 512 |
| 1024      | 1024 |
| 2048      | 1024 |
| 2048      | 2048 |

For each: run enough games that the steady-state (middle-of-run) evals/s
dominates the average — e.g. `--games` = 4-8x `--parallel` so the
tail-draining effect at the end doesn't skew the reported number much.
Watch the periodic progress lines (every 5s); take the **plateau** value
of evals/s and mean batch, not just the final printed summary (which
includes the tail).

Record for each combo: steady-state evals/s, steady-state mean batch,
final games/hour, peak GPU memory (`nvidia-smi` in another terminal).

## 4a. Transposition table cap sweep (`--tt-max-depth` / `--tt-max-entries`)

Naming convention for both flags (fixed after the M1 testing below,
so make sure you're on a build with this convention): **`0` = feature
fully disabled, `-1` (or any negative) = unlimited (no ceiling/no cap),
positive N = the actual depth ceiling / entry cap.** This applies
independently to `--tt-max-depth` (depth ceiling for cache eligibility)
and `--tt-max-entries` (hard cap on total cached entries across all
shards).

On the M1 (800 games, 7x7/5walls, sims=64) the shared TT's entry cap
showed a clear plateau: cap=50,000 caused active thrashing (constant
eviction of still-needed entries, ~553k evals, 39s), while cap=2,000,000
(the default) eliminated virtually all eviction at that scale (~257k
evals, ~23s — roughly 2x faster) and raising the cap further (5M, or -1
= unlimited) gave no additional benefit, because the actual working set
of distinct cacheable states at that scale never approached 2M.

The target machine will very likely need a **higher** cap than 2M to
reach the same "no eviction" plateau, because higher throughput (more
games/hour, more parallel games, more sims) means the self-play workload
accumulates a bigger distinct-state working set per unit wall-clock time
than the M1 ever could. Re-sweep on this machine rather than assuming the
M1's numbers transfer. Use the realistic target production params —
`--sims 800 --parallel <winner-from-step-4> --leaf-batch 1 \
--max-batch <winner>` (don't just default to 256/1024; use whatever
step 4 actually found, which may well be a larger max_batch/parallel
given the RTX 5070 Ti's extra headroom), not the 64-sims smoke-test
config used for the M1 numbers above. `--leaf-batch 1` is fixed here
deliberately — leaf batching is a separate quality-vs-speed question
(see step 6), not part of this throughput/TT-cap tuning track. Higher
sims means more of each simulated tree is TT-eligible per real move, and
more parallel games means a bigger working set, so the sweet spot found
at sims=64 will NOT transfer to a real training-cycle config:

```bash
for cap in 500000 2000000 5000000 10000000 20000000 -1; do
  echo "=== tt_max_entries=$cap ==="
  python selfplay_cpp.py --games 2000 --sims 800 --threads <winner> \
      --parallel <winner> --leaf-batch 1 --max-batch <winner> \
      --boardsize 7 --walls 5 --seed 42 \
      --model models_7x7_v2/best.pt --tt-max-depth -1 \
      --tt-max-entries $cap
done
```
Record: total time, games/hour, evals (lower evals at the same games/hour
= more cache hits = less eviction). Take the smallest cap where evals/
time stop improving as the "sweet spot" — going bigger than that buys
nothing, same as observed on the M1 at 2M. Also watch system RAM (32GB
available here vs far less on the M1) since a much bigger cap is
affordable if the sweep shows it's actually needed at this machine's
higher throughput. Win/loss/ply stats should stay bit-identical across
every cap value tested (same seed/model) — if they don't, something is
wrong (the eviction policy is designed to only ever affect speed, never
correctness).

Reference point only (NOT a target-machine prediction): a short M1 Air
sanity run at these exact realistic params (`tt_max_depth=-1`,
`tt_max_entries=20,000,000`, terminated early once steady-state was
established — games don't need to play to completion to measure
throughput) held steady at ~18,900-19,000 evals/s, mean batch ~255.4/256
(maxed), games/hour settling around ~44,000, with flat ~1.95-2.0GB RSS
memory throughout (no growth even with unlimited-depth caching). This
only confirms the pipeline is stable/correct at this scale on MPS — do
not use it as a CUDA throughput baseline.

## 5. bf16 / torch.compile

```bash
python selfplay_cpp.py --games 256 --sims 800 --threads <winner> \
    --parallel <winner> --max-batch <winner> \
    --model models_7x7_v2/best.pt --bf16
python selfplay_cpp.py ... --compile
python selfplay_cpp.py ... --bf16 --compile
```
Compare evals/s against the fp32 baseline from step 4. Expect bf16 alone
to give something in the 1.3-2x range; `--compile` adds compile-time
overhead up front (first batch will be slow) but should raise steady
state further. Confirm `--compile` doesn't break anything on a model
this small (compile overhead sometimes isn't worth it for tiny nets —
measure, don't assume).

## 6. Leaf batching: quality-vs-speed tradeoff (separate question from step 4)

Step 4 deliberately fixes `--leaf-batch 1` because batching leaves
degrades MCTS quality (batched leaves get evaluated against slightly
stale virtual-loss statistics instead of a fully backed-up tree) — it's
not a free speed knob like `parallel`/`max_batch`. Whether leaf batching
is worth it at all, and how many extra simulations would be needed to
compensate for the quality loss, is its own question and shouldn't be
decided from throughput numbers alone. Treat this step as open-ended
exploration, not a single sweep:

- **Throughput side**: measure games/hour at matched total simulation
  budget for a few `leaf_batch` values (1, 4, 8) at the winning
  `parallel`/`max_batch` from step 4, e.g.:
  ```bash
  for lb in 1 4 8; do
    python selfplay_cpp.py --games 512 --sims 800 --leaf-batch $lb \
        --threads <winner> --parallel <winner> --max-batch <winner> \
        --model models_7x7_v2/best.pt --out-dir /tmp/lb$lb
  done
  ```
- **Quality side** (the part that actually matters for the tradeoff):
  don't rely solely on soft proxies like policy entropy/value-distribution
  stats from the self-play npz files. Instead, evaluate quality directly:
  - Run each `leaf_batch` variant's resulting/trained checkpoint against
    a fixed holdout set (e.g. `_holdout_check.py` if applicable) and
    compare win-rate/loss metrics.
  - Or run direct matchups between models trained from leaf_batch=1 data
    vs leaf_batch=N data (see `tournament.py`/`_matchup.py`) to get a
    real win-rate delta, not just a proxy.
  - If leaf_batch>1 measurably hurts quality, test whether raising
    `--sims` for the batched variant (to compensate) closes the gap
    while still being faster in wall-clock terms than leaf_batch=1 at
    the same higher sim count.
- Only adopt `leaf_batch>1` for production self-play if the matchup/
  holdout evaluation shows it's quality-neutral (or the sims-compensated
  variant is both faster AND not worse) — a games/hour win alone is not
  sufficient justification.

## 7. End-to-end throughput vs the old pipeline (headline number)

Run the exact settings you'd use for a real training cycle and compare
directly against the old `train.py` self-play phase at the same
`--sims`/`--games`:

```bash
time python selfplay_cpp.py --games 200 --sims 800 \
    --threads <winner> --parallel <winner> --max-batch <winner> \
    --bf16 [--compile] --model models_7x7_v2/best.pt \
    --out-dir data_7x7_v2
```
Record final games/hour and compare to the ~400 games/hour baseline
measured previously on this same machine with the old multiprocessing
Python pipeline. This is the number that determines the actual wall-clock
speedup for training cycles.

## 8. Data compatibility check

Confirm the produced npz loads and trains identically to existing data:
```bash
python -c "
import numpy as np
d = np.load('data_7x7_v2/cycle_XXXX.npz')
print(d['states'].shape, d['policies'].shape, d['values'].shape)
print(d['states'].dtype, d['policies'].dtype, d['values'].dtype)
"
```
Then run a few `train.py` gradient steps against a directory containing
both old and new-format cycle files to confirm no shape/dtype mismatches.

## 9. Long-run stability

Once tuned, run something close to a real cycle's worth of games
(e.g. 2000+) unattended and confirm:
- no crashes / deadlocks over the full run
- memory stays flat (no leak across game resets) — watch RSS via `htop`
  or `ps` periodically
- GPU utilization (`nvidia-smi dmon`) stays high (>80%) through most of
  the run, not just the middle

## Deferred (explicitly out of scope for this round)

- Gumbel root-only MCTS modification — planned next, after infra is
  validated on real hardware.
