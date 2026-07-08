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

Grid to try (fix `--sims 800`, `--threads` = whatever won step 3):

| parallel | leaf_batch | max_batch |
|---|---|---|
| 128       | 8          | 256 |
| 256       | 8          | 256 |
| 256       | 4          | 512 |
| 512       | 4          | 512 |
| 512       | 2          | 512 |
| 1024      | 1          | 512 |
| 1024      | 2          | 512 |

For each: run enough games that the steady-state (middle-of-run) evals/s
dominates the average — e.g. `--games` = 4-8x `--parallel` so the
tail-draining effect at the end doesn't skew the reported number much.
Watch the periodic progress lines (every 5s); take the **plateau** value
of evals/s and mean batch, not just the final printed summary (which
includes the tail).

Record for each combo: steady-state evals/s, steady-state mean batch,
final games/hour, peak GPU memory (`nvidia-smi` in another terminal).

## 4a. Transposition table cap sweep (`--tt-max-depth` / `--tt-max-entries`)

On the M1 (800 games, 7x7/5walls, sims=64) the shared TT's entry cap
showed a clear plateau: cap=50,000 caused active thrashing (constant
eviction of still-needed entries, ~553k evals, 39s), while cap=2,000,000
(the default) eliminated virtually all eviction at that scale (~257k
evals, ~23s — roughly 2x faster) and raising the cap further (5M, or 0 =
uncapped) gave no additional benefit, because the actual working set of
distinct cacheable states at that scale never approached 2M.

The target machine will very likely need a **higher** cap than 2M to
reach the same "no eviction" plateau, because higher throughput (more
games/hour, more parallel games, more sims) means the self-play workload
accumulates a bigger distinct-state working set per unit wall-clock time
than the M1 ever could. Re-sweep on this machine rather than assuming the
M1's numbers transfer:

```bash
for cap in 500000 2000000 5000000 10000000 20000000 0; do
  echo "=== tt_max_entries=$cap ==="
  python selfplay_cpp.py --games 800 --sims 64 --threads <winner> \
      --parallel <winner> --boardsize 7 --walls 5 --seed 42 \
      --model models_7x7_v2/best.pt --tt-max-depth 200 \
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

## 6. Sequential vs batched MCTS quality/speed at matched batch size

This directly answers "does leaf_batch>1 hurt search quality, and is it
even needed for speed" — on the Mac, matched-batch leaf_batch=1 and
leaf_batch=8 landed in the same games/hour ballpark, but that was only
confirmed via projection (the full run wasn't let finish). Get a *real*
completed number here:

```bash
python selfplay_cpp.py --games 512 --sims 800 --leaf-batch 1 \
    --threads <winner> --parallel 512 --max-batch 512 \
    --model models_7x7_v2/best.pt --out-dir /tmp/lb1
python selfplay_cpp.py --games 512 --sims 800 --leaf-batch 8 \
    --threads <winner> --parallel 512 --max-batch 512 \
    --model models_7x7_v2/best.pt --out-dir /tmp/lb8
```
Compare: games/hour (should be close), and — if time allows — policy
entropy / value distribution stats between the two output npz files as a
proxy for search quality (higher-quality search should show slightly
lower policy entropy / sharper value predictions on average, though this
is a soft signal, not a strict test).

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
