# Value + pawn head redesign — implementation plan (NOT YET IMPLEMENTED)

Design doc only. Nothing here is built yet. Scope is confined to
`dual_network.py` — **no C++ changes, no extension rebuild** (the C++ engine only
ships the 8 input planes and receives `(logits, value)`).

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
