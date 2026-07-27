# FiLM Implementation Audit
## Reference: ethanjperez/film (film_gen.py, filmed.py)
## Date: 2026-07-27

| Check | Verdict | One-line finding |
|---|---|---|
| 1. Gamma residual | PASS | γ starts at 1 (weight=0, bias=1 init) — mathematically equivalent to `1 + proj(cond)` under gradient descent with `weight_decay=0.0` (confirmed in all configs), though the residual is baked into the bias rather than made explicit. |
| 2. Beta initialization | PASS | `beta_proj` weight and bias are both zero-initialized — β=0 at init, matching the reference exactly. |
| 3. Application order relative to normalization | PARTIAL | Q-FiLM is applied before the cosine-attention L2-normalize on Qh; per-channel magnitude information from γ is attenuated by that later normalize, though channel-wise directional changes survive. |
| 4. Per-channel vs per-token application | PASS | γ/β have shape `(B, 1, C)` and broadcast over the M query positions — modulation is per feature channel, identical across tokens, matching the reference. |
| 5. V-FiLM | PASS | Explicit `1.0 + gamma_v_proj(...)` residual, both projections zero-initialized, and applied to V after `norm_kv` — matches the reference's post-norm application pattern. |
| 6. Instruction encoder fidelity | PASS (simpler than reference) | Mean-pools the frozen encoder's last hidden state over real (non-pad) tokens into a single `(B, C)` vector — a valid, simpler substitute for the reference's GRU-over-sequence + last-token-hidden-state approach. |

---

## CHECK 1: Gamma residual

**Verdict: PASS** (functionally equivalent to the reference, with a caveat)

`aadp/models/film.py:46-53`:
```python
self.gamma_proj = nn.Linear(cond_dim, target_dim, bias=True)
self.beta_proj = nn.Linear(cond_dim, target_dim, bias=True)

# Identity initialisation: γ=1, β=0 at training start
nn.init.zeros_(self.gamma_proj.weight)
nn.init.ones_(self.gamma_proj.bias)
nn.init.zeros_(self.beta_proj.weight)
nn.init.zeros_(self.beta_proj.bias)
```

`aadp/models/film.py:67-69`:
```python
gamma = self.gamma_proj(cond).unsqueeze(1)  # (B, 1, target_dim)
beta = self.beta_proj(cond).unsqueeze(1)    # (B, 1, target_dim)
return x * gamma + beta
```

ICTC does **not** compute `gamma_applied = proj_output + 1` as an explicit forward-pass residual. Instead it bakes the "+1" into `gamma_proj.bias`'s initial value (bias init to `1`, weight init to `0`), so `gamma_proj(cond) = W@cond + 1` from the first forward pass, with no separate residual term.

This is functionally equivalent to the reference's explicit-residual form under gradient descent: adding a constant offset to a bias term does not change its gradient, so `d(loss)/d(bias)` is identical whether the "+1" is a fixed residual added after a bias initialized to 0, or baked into the bias's initial value directly. Both start at γ=1 and receive the same training signal.

**Caveat:** this equivalence breaks if weight decay is applied to biases — the reference's explicit-residual form regularizes γ toward 1 (identity) under weight decay, while ICTC's baked-in form would regularize `gamma_proj.bias` toward 0 (γ→0, i.e. toward *zeroing* the query, not identity). Checked `configs/*.yaml`: every config in this repo sets `weight_decay: 0.0`, so this discrepancy is not currently live, but it would become a real bug if weight decay were ever enabled without also excluding `gamma_proj.bias` from decay (or switching to an explicit residual).

## CHECK 2: Beta initialization

**Verdict: PASS**

Same block as above (`aadp/models/film.py:52-53`):
```python
nn.init.zeros_(self.beta_proj.weight)
nn.init.zeros_(self.beta_proj.bias)
```

`beta_proj` outputs exactly 0 for any input at init (zero weight AND zero bias — not default kaiming/xavier init). This matches the reference exactly; there is no random-shift instability from β at step 0.

## CHECK 3: Application order relative to normalization

**Verdict: PARTIAL**

`aadp/models/projector/stage2.py:182-188` (Q-FiLM):
```python
q = self.depth_queries.unsqueeze(0).expand(B, -1, -1)  # (B, M, C)
q = self.film(q, etext)                    # (B, M, C)
...
kv = self.norm_kv(kv)
```
There is no `norm_q` at all — it was removed earlier this session (see the `__init__` comment at `stage2.py:126-131`: found "essentially untrained... yet still the smallest-gradient submodule," a bottleneck rather than a collapse point). So the literal question ("is FiLM applied before or after norm_q?") has no norm_q to compare against, and `norm_kv` only touches the key/value side, not Q. In that narrow sense there's no LayerNorm undoing Q-FiLM.

However, a normalization *does* touch Q downstream, introduced by the cosine-attention fix — `stage2.py:221-232`:
```python
Qh = Qp.view(B, self._num_tokens, self._num_heads, head_dim).transpose(1, 2)  # (B,H,M,hd)
...
Qh = F.normalize(Qh + 1e-8, dim=-1)
```
`Qp` is the linear projection of `q`, which was already FiLM-modulated. `F.normalize(..., dim=-1)` L2-normalizes each head's query vector to unit norm, applied **after** FiLM. Since γ is a per-channel (diagonal) scale rather than a uniform scalar, `x*γ+β` generally changes *direction* as well as magnitude, so this normalize does not fully erase FiLM's effect — but it does discard any purely-magnitude-based signal γ was encoding (e.g. a uniform up/down-scale of Q), which the reference's post-batchnorm placement was designed to avoid entirely. This is the same class of bug the check is probing for, just via a different normalization than `norm_q` — worth being aware of even though it isn't a full erasure like a true pre-FiLM `LayerNorm(q)` would be.

## CHECK 4: Per-channel vs per-token application

**Verdict: PASS**

`aadp/models/film.py:67-69`:
```python
gamma = self.gamma_proj(cond).unsqueeze(1)  # (B, 1, target_dim)
beta = self.beta_proj(cond).unsqueeze(1)    # (B, 1, target_dim)
return x * gamma + beta
```
`gamma`/`beta` have shape `(B, 1, C)` — one scale/shift value per feature channel, broadcast identically across all M query positions (the `1` in dim 1). This is per-channel, matching the reference (`out[:,:,gs[i]] = ...` — scoped to the channel/module dim, broadcast over spatial/token positions). It is not the weaker per-token form (`B, M, 1`).

## CHECK 5: V-FiLM (recently added)

**Verdict: PASS** on all three sub-checks

`stage2.py:106-111` (init):
```python
self.gamma_v_proj = nn.Linear(cond_dim, embed_dim)
self.beta_v_proj = nn.Linear(cond_dim, embed_dim)
nn.init.zeros_(self.gamma_v_proj.weight)
nn.init.zeros_(self.gamma_v_proj.bias)   # gamma_v = 1.0 + 0 = 1.0 at init
nn.init.zeros_(self.beta_v_proj.weight)
nn.init.zeros_(self.beta_v_proj.bias)    # beta_v = 0 at init
```

`stage2.py:186-216` (forward, order relative to `norm_kv`):
```python
kv = self.norm_kv(kv)                       # Step 7
...
Vp = F.linear(kv, Wv, bv)   # (B, N, C)      # derived from normed kv

gamma_v = 1.0 + self.gamma_v_proj(etext)   # (B, C)
beta_v = self.beta_v_proj(etext)            # (B, C)
Vp = gamma_v.unsqueeze(1) * Vp + beta_v.unsqueeze(1)   # (B, N, C)
```

- **(a) Residual +1 present:** Yes, explicitly (`gamma_v = 1.0 + self.gamma_v_proj(etext)`) — unlike Check 1's Q-FiLM, this is the literal residual form the reference uses, not a baked-in bias.
- **(b) Zero-initialized:** Yes, both `gamma_v_proj` and `beta_v_proj` have weight AND bias zeroed, so `gamma_v=1.0, beta_v=0.0` for any input at init — identical to no V-FiLM until trained.
- **(c) Order relative to norm_kv:** Applied after. `kv` is normalized first (`norm_kv`), then projected to `Vp` via the (post-norm) `kv`, and only then is V-FiLM's affine transform applied to `Vp`. This matches the reference's "FiLM after normalization" placement — the normalization cannot undo a transform that hasn't happened yet at the point it runs.

## CHECK 6: Instruction encoder fidelity

**Verdict: PASS (simpler than reference, not incorrect)**

`aadp/data/instruction_encoder.py:92-98`:
```python
def _pool(self, last_hidden, attention_mask):
    if self.pooling == "mean":
        # Average over real (non-padding) token positions only.
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
```
`InstructionEncoder.__init__` defaults `pooling="mean"` (`instruction_encoder.py:24`), and none of `configs/ctclip_stage1.yaml`, `configs/ctclip_stage2_final.yaml`, or `configs/narval_smoke_stage2.yaml` override it — so training uses mean-pooling over the frozen BioMedLM's last hidden state, masked to exclude padding tokens.

This is option **(b)** from the check's list: mean pool over all real instruction tokens into a single `(B, cond_dim)` vector — not CLS (option a, `last_hidden[:, 0, :]`, also implemented but unused by default) and not the raw per-token sequence (option c). It is architecturally simpler than the reference's GRU-over-sequence-extract-last-hidden-state approach (no recurrence, no explicit "last real token" gather — though `pooling="last"` exists in the same file and would give a closer analogue), but it is a single, well-defined contextual vector and not a source of the mode-collapse symptom under investigation.

---

## PRIORITY FIXES

Ordered by likely impact on the mode-collapse / instruction-signal problem. Only genuine FAILs/PARTIALs are listed — everything else above is a PASS.

1. **Check 3 (PARTIAL) — cosine-attention's `F.normalize(Qh, dim=-1)` runs after Q-FiLM and discards any purely-magnitude-based signal γ encodes.** Likely low-to-moderate impact given `gamma_proj` is per-channel (direction survives), but it is the one place in the pipeline where a normalization sits downstream of FiLM on the query path, which is exactly the pattern the reference architecture avoids by applying FiLM after normalization instead of before. If cross-attention still under-differentiates by instruction after the V-FiLM fix's full training results are confirmed, this is the next place to look — e.g. by testing whether making `gamma_proj`'s output a scalar-per-batch (rather than diagonal) changes anything, which would confirm whether direction-only conditioning is carrying enough signal on its own.
2. **Check 1 (PASS with caveat) — not a current bug, but a latent one.** If `weight_decay` is ever set above `0.0` in a future config without excluding `gamma_proj.bias`/`beta_proj.bias` (and now `gamma_v_proj`/`beta_v_proj`'s biases too, for consistency) from decay, Q-FiLM's γ would regularize toward 0 (killing the query) instead of toward 1 (identity), unlike V-FiLM's explicit-residual form which decays safely toward identity. Cheap preventative fix: switch `FiLMLayer.forward` to the explicit `gamma = 1.0 + self.gamma_proj(cond)` form (bias init 0) to match V-FiLM's already-correct pattern, removing the dependency on `weight_decay=0.0` staying true forever.
