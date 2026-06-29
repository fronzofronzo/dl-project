"""T2 — Input-conditioned image-space directions + dynamic weight gate Φ (owner: person 2).

The training twin of the no-training Solution A. Solution A failed for three
MEASURED reasons (results/findings_2026-06-19.md):
  1. text axes live in the wrong cone (modality gap) -> cos(image, text dir) ~ 0
  2. dynamic weights from cos(v_ref, d) were ~constant (broken, not dynamic)
  3. CLIP text directions are entangled (Heavy_Makeup ~ female)

T2 LEARNS, directly in image space, both the directions and the weights, so all
three bugs disappear by construction. Closes the four CLAY limits:

  P1 sign        -> a positive constraint ADDS d_i, a negative SUBTRACTS it
  P2 weighting   -> per-condition magnitude from an MLP w = f(v_ref, attr_id)
  P3 interaction -> correlation-aware regularizer on D (learned, structure-matched)
  P4 dynamic     -> BOTH the magnitude AND the DIRECTION depend on the actual v_ref

DESIGN HISTORY (results/solution_a_t2.md): the first T2 used a single GLOBAL
direction per attribute (d_i fixed for every face), only the magnitude saw v_ref.
That rigid-axis bottleneck made T2 lose to the plain MLP baseline on composed /
entangled queries (+Chubby&-Young 0.159 vs 0.534) while still winning on single
directional attributes (+Smiling 0.198 vs 0.087). Fix (I1): the direction itself
is now input-conditioned via a small per-attribute low-rank correction predicted
from v_ref, so "add smiling" can point differently on different faces — strictly
more expressive than the rigid axis, still per-attribute interpretable, and
ablatable (cond_dir=False recovers the rigid T2 exactly).

Implements the shared Φ contract from src/solution_b/phi.py:

    forward(v_ref, cond_col, cond_sign, cond_mask) -> v_q

CLIP stays frozen; only Φ is trained. The residual on v_ref preserves identity.

Composition (ambient):
    v_q = normalize( v_ref + scale * Σ_i  w_i · s_i · d̂_i(v_ref) )
with d̂_i(v_ref) = normalize( D[col_i] + U[col_i]ᵀ·dir_net(v_ref, attr_emb[col_i]) ),
     w_i        = softplus( weight_net(v_ref, attr_emb[col_i]) ).

NOTE: pad vs attribute-0 — cond_col==0 is BOTH padding and attribute 0
(5_o_Clock_Shadow). The truth is cond_mask; never use cond_col==0 as "is pad".
Padded conditions are zeroed via cond_mask AND carry cond_sign==0, so they add
nothing to the sum (permutation-invariant, mask-honest by construction).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ATTR = 40
DIM = 512


class T2Phi(nn.Module):
    """Input-conditioned learned-direction fusion module Φ_T2.

    Pipeline (forward):
      (a) per-attribute base direction + v_ref-conditioned low-rank correction (I1)
          d̂ = normalize(D[col] + U[col]ᵀ·dir_net(v_ref, attr_emb[col]))
      (b) dynamic per-condition magnitude   w = softplus(weight_net(v_ref, attr_emb[col]))
      (c) signed, masked composition        edit = scale * Σ w·s·d̂
      (d) residual + identity anchor         v_q = v_ref + edit  -> normalize
    """

    def __init__(self, n_attr=N_ATTR, dim=DIM, w_emb=64, w_hidden=256, rank=16,
                 residual="ambient", cond_dir=True):
        """
        n_attr     number of CelebA attributes (direction-dictionary rows)
        dim        CLIP feature dim (v_ref / v_q / directions), = 512
        w_emb      width of the attribute embedding fed to both heads
        w_hidden   hidden width of the MLP heads
        rank       low-rank dimension of the input-conditioned direction correction (I1)
        residual   "ambient" | "tangent"  (tangent = Exp map at v_ref, ablation)
        cond_dir   I1 toggle: True = directions depend on v_ref (low-rank correction),
                   False = rigid global directions (the original T2, for ablation)
        """
        super().__init__()
        self.dim = dim
        self.residual = residual
        self.cond_dir = cond_dir
        self.rank = rank

        # (a) base learned direction dictionary, IMAGE space (warm-startable, I3)
        self.D = nn.Parameter(torch.randn(n_attr, dim) * 0.02)

        # shared attribute embedding feeding both heads
        self.attr_emb = nn.Embedding(n_attr, w_emb)

        # (b) dynamic-magnitude head: f(v_ref, attr_id) -> non-negative scalar weight
        self.weight_net = nn.Sequential(
            nn.Linear(dim + w_emb, w_hidden),
            nn.ReLU(),
            nn.Linear(w_hidden, 1),
        )

        # (I1) input-conditioned direction correction: per-attribute low-rank basis U,
        # whose coordinates are predicted from (v_ref, attr_emb). delta = coords·U.
        # Init small so the correction starts ~0 (d ~= base D) and the net grows it.
        if cond_dir:
            self.U = nn.Parameter(torch.randn(n_attr, rank, dim) * (0.02 / rank ** 0.5))
            self.dir_net = nn.Sequential(
                nn.Linear(dim + w_emb, w_hidden),
                nn.ReLU(),
                nn.Linear(w_hidden, rank),
            )

        # learned global edit magnitude (mirrors Solution A's alpha; here trainable)
        self.log_scale = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------ #
    # components
    # ------------------------------------------------------------------ #
    def _feat(self, v_ref, cond_col):
        """Shared head input [B, C, dim + w_emb] = (v_ref broadcast || attr_emb)."""
        C = cond_col.shape[1]
        emb = self.attr_emb(cond_col)                       # [B, C, w_emb]
        vref = v_ref.unsqueeze(1).expand(-1, C, -1)         # [B, C, dim]
        return torch.cat([vref, emb], dim=-1)               # [B, C, dim + w_emb]

    def _weights(self, x):
        """(b) per-condition non-negative magnitude w = softplus(MLP). [B, C] >= 0."""
        return F.softplus(self.weight_net(x)).squeeze(-1)   # [B, C]

    def _directions(self, cond_col, x):
        """(a) unit directions, OPTIONALLY conditioned on v_ref (I1). -> [B, C, dim].

        rigid (cond_dir=False): d̂ = normalize(D[col]) — global, same per face.
        I1    (cond_dir=True) : d̂ = normalize(D[col] + Σ_r coords_r · U[col, r]),
              coords = dir_net(v_ref, attr_emb) -> the direction REORIENTS per input,
              killing the rigid-axis bottleneck (B1) while keeping per-attribute D
              interpretable and ablatable.
        """
        d = self.D[cond_col]                                # [B, C, dim] base
        if self.cond_dir:
            coords = self.dir_net(x)                        # [B, C, rank]
            U = self.U[cond_col]                            # [B, C, rank, dim]
            delta = (coords.unsqueeze(-1) * U).sum(dim=2)   # [B, C, dim]
            d = d + delta
        return F.normalize(d, dim=-1)                       # [B, C, dim] unit

    def _compose(self, dirs, cond_sign, cond_mask, w):
        """(c) signed, masked, weighted sum of unit directions -> edit [B, dim].

        Polarity is explicit: +1 adds d̂, -1 subtracts it (P1). Padded columns are
        zeroed by cond_mask, so the count of conditions is variable and the result
        is permutation-invariant.
        """
        coeff = w * cond_sign * cond_mask.to(w.dtype)       # [B, C] signed, padded->0
        edit = (coeff.unsqueeze(-1) * dirs).sum(dim=1)      # [B, dim]
        return self.log_scale.exp() * edit

    def _apply_residual(self, v_ref, edit):
        """(d) compose v_q. The `+ v_ref` residual preserves identity.

          ambient: v_q = v_ref + edit
          tangent: geodesic step from v_ref along the tangent-projected edit
                   (Exp map anchored at v_ref); findings A: tangent ~= ambient.

        Returns un-normalized v_q; forward() applies the final L2-normalize.
        """
        if self.residual == "ambient":
            return v_ref + edit

        # tangent: anchor μ = v_ref per row, drop the radial part, Exp back
        vr = F.normalize(v_ref, dim=1)
        step = edit - (edit * vr).sum(dim=1, keepdim=True) * vr     # tangent component
        r = step.norm(dim=1, keepdim=True).clamp_min(1e-7)
        return vr * torch.cos(r) + (step / r) * torch.sin(r)

    def ortho_reg(self, target=None):
        """Decorrelation regularizer on the base dictionary D (CLAY limit P3).

        Mean squared off-diagonal of (Gram(D̂) − target), where Gram is the
        unit-direction cosine matrix.

          target=None : push every off-diagonal to 0 (plain orthogonality — the
                        original T2 reg). MEASURED to over-decorrelate: findings
                        show correlated same-sign attributes carry real signal, so
                        forcing all pairs apart hurts (+Eyeglasses&+Smiling −0.023).
          target=C    : push the direction geometry toward the empirical attribute
                        CORRELATION matrix C (from train labels). Attributes that
        
                        genuinely co-occur are ALLOWED to correlate; only truly
                        independent / anti-correlated pairs are pushed apart. This
                        is the conflict-aware version (I2) — structure-matched, not
                        a blanket identity.

        Added to the loss by the train loop.
        """
        Dn = F.normalize(self.D, dim=1)                     # [n_attr, dim]
        gram = Dn @ Dn.t()                                  # [n_attr, n_attr] in [-1,1]
        diff = gram if target is None else gram - target.to(gram)
        off = diff - torch.diag(torch.diagonal(diff))       # zero the diagonal
        n = gram.shape[0]
        return off.pow(2).sum() / (n * (n - 1))

    @torch.no_grad()
    def load_directions(self, axes):
        """Hybrid warm start (I3): copy image-space probe axes into the base D.

        `axes` is a [n_attr, dim] tensor row-aligned to attribute columns, built
        from the TRAIN split as d_img(attr) = mean(F[label=1]) − mean(F[label=0])
        (image-image difference -> modality-gap-free, findings §3). Injects real
        attribute structure instead of starting D from random·0.02; the net then
        only has to refine, not discover, the directions.
        """
        self.D.copy_(axes.to(self.D).float())

    # ------------------------------------------------------------------ #
    # contract
    # ------------------------------------------------------------------ #
    def forward(self, v_ref, cond_col, cond_sign, cond_mask):
        """Φ contract: compose L2-normalized query v_q [B, 512] from v_ref + signed
        conditions. Query-side only; CLIP/DB untouched.

          v_ref     [B, 512]  reference image features (L2-norm, frozen CLIP)
          cond_col  [B, C]    long, attribute column per condition (0 = pad)
          cond_sign [B, C]    float, +1 additive / -1 subtractive / 0 pad
          cond_mask [B, C]    bool, True for real conditions
          -> v_q    [B, 512]  composed query, L2-normalized
        """
        x = self._feat(v_ref, cond_col)                       # [B, C, dim + w_emb]
        w = self._weights(x)                                  # (b) [B, C]
        dirs = self._directions(cond_col, x)                  # (a) [B, C, dim] (I1)
        edit = self._compose(dirs, cond_sign, cond_mask, w)   # (c) [B, dim]
        v_q = self._apply_residual(v_ref, edit)               # (d) residual
        return F.normalize(v_q, dim=1)


# --------------------------------------------------------------------------- #
# Smoke test: python -m src.solution_a.t2_directions
# Shape + invariants (mirror of the T1 smoke):
#   - output [B, 512], L2-normalized (v_q.norm(dim=1) ~ 1)
#   - permutation-invariance: shuffling the condition order leaves v_q unchanged
#   - mask honesty: extra padded columns leave v_q unchanged
#   - empty edit: cond_mask all False -> v_q ~ normalize(v_ref)
#   - I1 actually conditions: direction changes with v_ref (cond_dir=True)
#   - ortho_reg (both targetless and correlation-matched): finite scalar
#   - backward: finite, non-zero grads on every parameter
# --------------------------------------------------------------------------- #
def _smoke():
    torch.manual_seed(0)
    B, C = 4, 3
    phi = T2Phi()
    phi.eval()

    v_ref     = F.normalize(torch.randn(B, DIM), dim=1)
    cond_col  = torch.randint(0, N_ATTR, (B, C))
    cond_sign = torch.where(torch.rand(B, C) > 0.5, 1.0, -1.0)
    cond_mask = torch.ones(B, C, dtype=torch.bool)

    # 1. output shape + L2-norm
    v_q = phi(v_ref, cond_col, cond_sign, cond_mask)
    assert v_q.shape == (B, DIM), f"shape {v_q.shape}"
    assert torch.allclose(v_q.norm(dim=1), torch.ones(B), atol=1e-5), "not L2-normalized"
    print("  ok  output shape and L2-norm")

    # 2. permutation-invariance: shuffling condition order must not change v_q
    perm = torch.randperm(C)
    v_q_perm = phi(v_ref, cond_col[:, perm], cond_sign[:, perm], cond_mask[:, perm])
    assert torch.allclose(v_q, v_q_perm, atol=1e-5), "not permutation-invariant"
    print("  ok  permutation-invariance")

    # 3. mask honesty: adding padded columns must not change v_q
    col_pad  = torch.cat([cond_col,  torch.zeros(B, 2, dtype=torch.long)], dim=1)
    sign_pad = torch.cat([cond_sign, torch.zeros(B, 2)], dim=1)
    mask_pad = torch.cat([cond_mask, torch.zeros(B, 2, dtype=torch.bool)], dim=1)
    v_q_pad = phi(v_ref, col_pad, sign_pad, mask_pad)
    assert torch.allclose(v_q, v_q_pad, atol=1e-5), "mask not honored"
    print("  ok  mask honesty (extra padded columns ignored)")

    # 4. empty edit: all conditions masked -> v_q ~ normalize(v_ref)
    mask_empty = torch.zeros(B, C, dtype=torch.bool)
    v_q_empty = phi(v_ref, cond_col, cond_sign, mask_empty)
    assert torch.allclose(v_q_empty, F.normalize(v_ref, dim=1), atol=1e-5), \
        "empty-condition case: v_q should equal normalize(v_ref)"
    print("  ok  empty edit (v_q == normalize(v_ref) when no conditions)")

    # 5. I1 conditions on v_ref: same constraint, different reference -> different dir
    v_ref2 = F.normalize(torch.randn(B, DIM), dim=1)
    one_col = cond_col[:, :1]
    x1 = phi._feat(v_ref,  one_col); x2 = phi._feat(v_ref2, one_col)
    d1 = phi._directions(one_col, x1); d2 = phi._directions(one_col, x2)
    assert not torch.allclose(d1, d2, atol=1e-4), "I1 inactive: direction ignores v_ref"
    rigid = T2Phi(cond_dir=False)
    xr = rigid._feat(v_ref, one_col)
    dr1 = rigid._directions(one_col, xr)
    dr2 = rigid._directions(one_col, rigid._feat(v_ref2, one_col))
    assert torch.allclose(dr1, dr2, atol=1e-6), "rigid ablation should ignore v_ref"
    print("  ok  I1 input-conditioned directions (rigid ablation static)")

    # 6. ortho_reg: finite non-negative scalar, both targetless and correlation-matched
    reg0 = phi.ortho_reg()
    target = torch.eye(N_ATTR)                              # dummy correlation target
    regc = phi.ortho_reg(target=target)
    for r in (reg0, regc):
        assert r.ndim == 0 and torch.isfinite(r) and r >= 0, f"bad ortho_reg {r}"
    print(f"  ok  ortho_reg (targetless={reg0.item():.4f}, corr-target={regc.item():.4f})")

    # 7. backward: finite loss (+ortho), non-zero gradients on all parameters
    phi.train()
    v_q = phi(v_ref, cond_col, cond_sign, cond_mask)
    loss = (1 - (v_q * F.normalize(v_ref, dim=1)).sum(dim=1)).mean() + 0.1 * phi.ortho_reg()
    loss.backward()
    for name, p in phi.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0, \
            f"bad grad on {name}"
    n_params = sum(p.numel() for p in phi.parameters())
    print(f"  ok  backward (loss={loss.item():.4f}, all grads finite/non-zero, "
          f"{n_params:,} params)")

    print("smoke test passed.")


if __name__ == "__main__":
    _smoke()
