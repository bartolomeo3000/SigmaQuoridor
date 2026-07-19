# Value + pawn head redesign — IMPLEMENTED 2026-07-19

Implemented as designed. Scope stayed confined to `dual_network.py` plus thin
CLI plumbing in `train.py`/`selfplay_cpp.py` — no C++ changes, no extension
rebuild (the C++ engine only ships the 8 input planes and receives
`(logits, value)`; confirmed by a live `selfplay_cpp.py` smoke run against the
new net).

## What actually shipped

- `DualNetwork` takes `value_head="pooled"|"legacy"` and `pawn_head="local"|"legacy"`
  (defaults: the new variants — sections 1/2 below, built exactly as specced).
- `_infer_arch`/`load_model` detect the variant from state-dict keys and add a
  `boardsize_marker` buffer (only on pooled-head nets) so boardsize inference
  no longer depends on `value_fc1`. Verified: `models_9x9/best.pt` (cycle 359,
  pre-redesign) still loads as `legacy`/`legacy` unchanged.
- `dual_network.warm_start_from_legacy(old_path, value_head, pawn_head)` — the
  partial-load helper from section 4/Option A. Copies every tensor whose key
  and shape are unchanged (trunk + wall sub-heads), leaves the rest freshly
  initialized. On the real net: **161/174 tensors copied**, 13 fresh (the two
  changed heads + the new marker). New param count: 2,909,869 vs 2,878,575
  legacy (**+1.09%** — "heads should add little", confirmed).
- `init_head_redesign.py` — one-off script wrapping the above: warm-starts a
  **new lineage** `models_9x9_heads/` (`best.pt` + `checkpoints/cycle_0000.pt`,
  the latter built with a fresh Adam/`MultiStepLR` so `train.py --resume`
  picks it up at cycle 0). `models_9x9`/`data_9x9_fix` is untouched.
- `train.py`/`selfplay_cpp.py` gained `--value-head`/`--pawn-head` flags
  (same "only affects fresh construction" caveat as `--filters`/`--res`/
  `--gpool-every` already had — resuming requires matching flags, exactly the
  pre-existing pattern, no new sharp edge). `_eval_worker`'s challenger
  reconstruction and the pure-Python self-play worker path were updated too.
- `test_head_redesign.py` — regression suite: old-checkpoint load, fresh-net
  round-trip (incl. `boardsize_marker`), 4-way forward-shape check across head
  variants, and the warm-start correctness checks above. All pass, alongside
  unchanged `test_cpp_parity.py` and `test_canon_consistency.py`.

## A/B result — 2026-07-19/20: new heads win decisively; now the default lineage

Rather than a fresh self-play buffer, `models_9x9_heads` was warm-trained
directly on the existing `data_9x9_fix` canonical-frame buffer (no new
self-play needed — the 8-plane input encoding is architecture-agnostic, so the
existing policy/value targets are valid supervised data for any head shape):

1. `--train-only` on `data_9x9_fix` for a handful of cycles (default
   `recency_decay=0.92`, `buffer_cycles=30`) — loss converged to legacy
   parity (~1.13) within 3 cycles, confirming the warm-started trunk only
   needed the two changed heads to catch up.
2. Then `--recency-decay 1.0 --buffer-cycles 50` (added as new CLI flags,
   see below) for several more cycles to get a less recency-skewed fit over
   the full buffer, since repeatedly resampling a *static* buffer with the
   default recency bias over many cycles over-weights the same subset every
   time (unlike normal self-play, where the favoured window shifts as fresh
   cycles keep arriving) — loss rose transiently (broader distribution) then
   re-converged below legacy parity (1.1229 vs legacy's 1.1338).
3. Finally 2 more cycles back on default settings (recency-biased toward the
   newest/highest-quality data) to sharpen the fit before evaluating.

**Tournament** (`tournament_cpp.py`, 1024 games, 800 sims, `--boardsize 9
--walls 10` — required explicitly, the script's own defaults are the old 7x7
lineage's 7/5): `models_9x9_heads/best.pt` vs `models_9x9/best.pt` (cycle
359, legacy heads) — **1549.7 vs 1450.3 Elo, 63.9% score (633W/43D/348L)**.
Decisive win.

**`train.py` gained two more CLI flags** during this process (both default to
the pre-existing hardcoded constants, so unrelated runs are unaffected):
`--recency-decay F` (overrides `BUFFER_RECENCY_DECAY`) and `--buffer-cycles N`
(overrides `BUFFER_CYCLES`). Also fixed a pre-existing (unrelated) bug found
along the way: `_Tee.write` crashed the whole run with `UnicodeEncodeError`
when the Windows console couldn't encode a `→` in a log line — now degrades
gracefully to the console while the UTF-8 log file keeps the exact text.

**`models_9x9_heads` is now the default lineage** — updated in:
- `train.py` (`MODEL_DIR`), `cpp_train_loop.py` (`MODEL_PATH`) — `DATA_DIR`
  stays `data_9x9_fix` on purpose (still valid, no reason to fork the buffer).
- `app.py` (`MODEL_DIR`, serving default).
- `export_onnx.py` (9x9 lineage source path + checkpoint-export source dir;
  output paths under `docs/models_9x9/` are unchanged, so the browser
  frontend's `defaultModel`/checkpoint-picker entries keep working with zero
  JS changes — confirmed `docs/mcts_worker.js` only reads the `policy_logits`/
  `value` output tensors generically, no architecture-specific assumptions).
- Re-exported `docs/models_9x9/best.onnx` from the new net and verified
  numerically against the PyTorch model via `onnx.reference.ReferenceEvaluator`
  (no `onnxruntime` install needed) — max diff 2.5e-6 (policy), 1.5e-8 (value),
  pure fp32 noise. Fully compatible with both the Python app and the JS/ONNX
  browser frontend.
- `models_9x9`/its checkpoints are left untouched as the frozen legacy
  reference (also what `test_head_redesign.py`/`test_canon_consistency.py`
  test against, deliberately).

Not touched: `docs/index.html`'s hardcoded legacy-cycle picker entries (e.g.
"Cycle 339", "Cycle 321", ...) still point at already-exported, still-valid
legacy onnx files — nothing broke, they just represent the old lineage's
history. Only "best (latest)" needed to (and does) point at the new net.

---

Original design notes follow (kept for rationale/reference).

## Current state (as of 2026-07-19)

Live nets are already **128 filters / 10 residual blocks / gpool every 3** — the
earlier "scale up / enable gpool" suggestions are done. What remains are the two
**heads**, both of which throw away information:

- **Value head** ([dual_network.py:195-199](dual_network.py#L195-L199),
  forward [239-242](dual_network.py#L239-L242)):
  `Conv2d(F→1) → BN → ReLU → flatten(N²) → Linear(N²→64) → ReLU → Linear(64→1) → tanh`.
  The whole trunk is squeezed through **one** spatial channel before the FC.
  Quoridor value hinges on global race margin + wall budgets; one channel is
  very tight, and the flatten makes it board-size-specific.
- **Pawn sub-head** ([dual_network.py:183](dual_network.py#L183), forward
  [224-225](dual_network.py#L224-L225)): `global-avg-pool(F) → Linear(F→8)`.
  Global average pool discards *where the pawn is*, yet pawn-move value is local
  (jumps depend on the opponent being adjacent; step value depends on the
  pawn's own surroundings).

Both heads output in the current player's canonical frame — unchanged by this
plan. The canonicalization boundary (record/serve vflip) is independent of head
internals; see the `policy-canonical-frame` memory / `test_canon_consistency.py`.

---

## 1. Value head — KataGo-style pooled head

Replace the single-channel flatten with a small conv + **global pooling**
(mean **and** max over the board) feeding the FC. This gives the value FC direct
access to global statistics and makes it board-size-agnostic.

New modules (constructor), `C_v = 32`, `H = 128`:

```python
self.value_conv = nn.Conv2d(filters, C_v, kernel_size=1, bias=False)
self.value_bn   = nn.BatchNorm2d(C_v)
self.value_pool_fc = nn.Linear(2 * C_v, H)   # NEW key -> variant marker
self.value_fc2     = nn.Linear(H, 1)
```

Forward:

```python
v = F.relu(self.value_bn(self.value_conv(x)))         # (B, C_v, N, N)
v = torch.cat([v.mean(dim=(2, 3)), v.amax(dim=(2, 3))], dim=1)  # (B, 2*C_v)
v = F.relu(self.value_pool_fc(v))                     # (B, H)
v = torch.tanh(self.value_fc2(v))                     # (B, 1)
```

Rationale: mean captures aggregate race/wall state, max captures the single most
decisive cell (e.g. a chokepoint). `2*C_v = 64` inputs to the FC vs 1 channel
today. Cheap (C_v=32 1×1 conv + a 64→128 FC).

Tunables: `C_v ∈ {16,32,48}`, `H ∈ {64,128}`. Optionally add a small "value
bonus" auxiliary later (see plies-to-end idea in the broader review) — out of
scope here.

---

## 2. Pawn sub-head — gather trunk features at the pawn cells

Extract trunk features **at the current player's pawn cell and the opponent's
pawn cell** (via the input one-hot planes 0 and 1), concatenate with the global
mean, and feed an MLP.

New modules (`H_p = 64`):

```python
self.policy_pawn_fc1 = nn.Linear(3 * filters, H_p)  # NEW keys -> variant marker
self.policy_pawn_fc2 = nn.Linear(H_p, 8)
```

Forward (needs the original input `x_in` — capture it at the top of `forward`
before the trunk overwrites `x`):

```python
my_mask  = x_in[:, 0:1]          # (B,1,N,N) one-hot, my pawn      (canonical POV)
opp_mask = x_in[:, 1:2]          # (B,1,N,N) one-hot, opponent pawn
feat_my   = (trunk * my_mask ).sum(dim=(2, 3))   # (B,F) trunk features at my cell
feat_opp  = (trunk * opp_mask).sum(dim=(2, 3))   # (B,F) at opponent cell
feat_glob = trunk.mean(dim=(2, 3))               # (B,F) global context
p_pawn = torch.cat([feat_my, feat_opp, feat_glob], dim=1)   # (B, 3F)
p_pawn = self.policy_pawn_fc2(F.relu(self.policy_pawn_fc1(p_pawn)))  # (B, 8)
```

The mask-multiply-and-sum is an exact, differentiable gather at the one-hot cell
(no coordinate bookkeeping, works batched). Wall sub-heads (h/v 1×1 convs) are
unchanged. Concatenation order into the full policy vector
(`[pawn(8), h(W²), v(W²)]`) is unchanged.

Rationale: `feat_my` gives local step/wall context, `feat_opp` gives the jump/
diagonal context (only relevant when adjacent), `feat_glob` keeps the race
context the old avg-pool provided. Legality is still handled by masking, so the
head only has to rank among legal directions.

Tunables: drop `feat_opp` (→ 2F) if it doesn't help; `H_p ∈ {32,64}`.

---

## 3. Backward-compatible loading (REQUIRED — do it this way)

Mirror the existing self-describing pattern (`gpool_every` is already inferred
from the presence of `residuals.*.gpool_fc.weight`, [dual_network.py:381-385](dual_network.py#L381-L385)).

### 3a. Detect head variants in `_infer_arch`
Use the **presence of the new keys** as the variant marker:

```python
value_head = "pooled" if "value_pool_fc.weight" in sd else "legacy"
pawn_head  = "local"  if "policy_pawn_fc1.weight" in sd else "legacy"
```

Return these alongside `(filters, num_residual, boardsize, gpool_every)` and pass
them to the `DualNetwork` constructor so **each checkpoint builds the head
architecture it was saved with**. Old checkpoints → legacy heads (load fine);
new checkpoints → new heads. `load_state_dict` stays `strict=True` because the
constructed modules match the saved keys exactly.

### 3b. Fix boardsize inference (the sharp edge)
`boardsize` is currently derived **only** from `value_fc1.weight.shape[1] == N²`
([dual_network.py:380](dual_network.py#L380)). The pooled value head removes
`value_fc1`, so that inference path disappears for new nets. Two options:

- **Preferred:** add `self.register_buffer("boardsize_marker", torch.tensor(N))`
  in the constructor. It serializes into the state-dict. `_infer_arch` reads it
  when present, else falls back to the `value_fc1` path for legacy checkpoints.
  Robust and self-documenting; only new-variant nets carry it, so no key clash
  with old checkpoints.
- Alternative: derive `N` from `boardsize_marker` OR keep a legacy `value_fc1`
  around unused (ugly — don't).

### 3c. Constructor flags
Add `value_head="pooled"|"legacy"` and `pawn_head="local"|"legacy"` params
(default to the **new** variants for fresh nets). Fresh-net creation sites pick
new defaults automatically:
`train.py` `DualNetwork(...)`, `selfplay_cpp.py:181` `DualNetwork(...)`,
`dual_network.py:init_fresh_model`, and any smoke-test constructions.

### 3d. `save_model` unchanged
Still saves `state_dict()`; the new buffer + new keys ride along automatically.

---

## 4. Interaction with the canonicalization-fix retrain (decision)

The canon fix was designed to **continue from `best.pt`** (keep the frame-
invariant value head + trunk; only the policy head re-aligns). A value-head
redesign is **incompatible with keeping the old value head** — new head shapes
mean those weights can't transfer. So decide:

- **Option A — sequence them.** First run the canon-fix retrain by continuing
  from `best.pt` (cheap, keeps value head, gets a strong *canonical* net).
  Later, do the head redesign as a **separate experiment**: warm-start the
  **trunk + wall sub-heads** from that net and reinit the two changed heads
  (needs a small partial-load helper: `load_state_dict(sd, strict=False)` after
  filtering to matching keys). Cleanest attribution — you'll know which change
  bought what.
- **Option B — bundle now.** Since the canon fix already forces a policy-head
  re-alignment and a fresh data buffer, adopt the new heads in the *same* fresh
  run. You lose the "keep the value head" benefit (value head trains from
  scratch), but it's one retrain instead of two. Trunk can still be
  warm-started from `best.pt` via the partial-load helper.

Recommendation: **A** — get the canonical net solid first (it's the higher-
confidence win and preserves the value head), then A/B-test the head redesign
against it so you can measure the head change in isolation.

---

## 5. Testing checklist (before trusting a run)

- **Old-checkpoint load:** `load_model("models_9x9/best.pt")` still works and
  `_infer_arch` returns legacy heads + correct boardsize (9). `app.py` serves it
  unchanged.
- **New net round-trip:** create a pooled/local net, `save_model` → `load_model`,
  assert identical outputs on a fixed input (arch inferred correctly).
- **Shape/parity:** forward on a `(B,8,9,9)` batch returns `(B,136)` + `(B,1)`;
  `test_cpp_parity.py` unaffected (no C++ change); `test_canon_consistency.py`
  still passes (head internals don't touch the canonical frame boundary).
- **Param count sanity:** log new vs old param counts; heads should add little.
- **Partial warm-start helper** (if used): assert trunk/wall keys copied,
  changed-head keys freshly initialized.

## Files touched (when implemented)
`dual_network.py` only: `DualNetwork.__init__`/`forward`, `_infer_arch`,
constructor call sites in `train.py` / `selfplay_cpp.py`. No `cpp/*`, no rebuild.
