"""T2 — Learned image-space directions + dynamic weight gate Φ (owner: person 2).

The training twin of the no-training Solution A. Solution A failed for three
MEASURED reasons (results/findings_2026-06-19.md):
  1. text axes live in the wrong cone (modality gap) -> cos(image, text dir) ~ 0
  2. dynamic weights from cos(v_ref, d) were ~constant (broken, not dynamic)
  3. CLIP text directions are entangled (Heavy_Makeup ~ female)

T2 LEARNS, directly in image space, both the directions and the weights, so all
three bugs disappear by construction. Closes the four CLAY limits:

  P1 sign        -> a positive constraint ADDS d_i, a negative SUBTRACTS it
  P2 weighting   -> per-condition weight from an MLP w = f(v_ref, attr_id)
  P3 interaction -> orthogonality regularizer on D (learned Gram–Schmidt)
  P4 dynamic     -> weights computed from the ACTUAL v_ref, per input

Implements the shared Φ contract from src/solution_b/phi.py so it drops into the
shared train.py / run.py loop unchanged:

    forward(v_ref, cond_col, cond_sign, cond_mask) -> v_q

CLIP stays frozen; only Φ is trained. The residual on v_ref preserves identity.

Composition (ambient):
    v_q = normalize( v_ref + scale * Σ_i  w_i · s_i · d̂_i )
with d̂_i = D[col_i] / ‖D[col_i]‖, w_i = softplus(MLP(v_ref, attr_emb[col_i])).

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
    """Learned-direction fusion module Φ_T2.

    Pipeline (forward):
      (a) pick + unit-normalize learned directions   d̂ = normalize(D[col])
      (b) dynamic per-condition weight               w = softplus(MLP(v_ref, attr_emb[col]))
      (c) signed, masked composition                 edit = scale * Σ w·s·d̂
      (d) residual + identity anchor                  v_q = v_ref + edit  -> normalize
    """

    def __init__(self, n_attr=N_ATTR, dim=DIM, w_emb=64, w_hidden=256,
                 residual="ambient"):
        """
        n_attr     number of CelebA attributes (direction-dictionary rows)
        dim        CLIP feature dim (v_ref / v_q / directions), = 512
        w_emb      width of the attribute embedding fed to the weight MLP
        w_hidden   hidden width of the dynamic-weight MLP
        residual   "ambient" | "tangent"  (tangent = Exp map at v_ref, ablation)
        """
        super().__init__()
        self.dim = dim
        self.residual = residual

        # (a) learned direction dictionary, IMAGE space (not text prompts)
        self.D = nn.Parameter(torch.randn(n_attr, dim) * 0.02)

        # (b) dynamic-weight head: f(v_ref, attr_id) -> scalar weight per condition
        self.attr_emb = nn.Embedding(n_attr, w_emb)
        self.weight_net = nn.Sequential(
            nn.Linear(dim + w_emb, w_hidden),
            nn.ReLU(),
            nn.Linear(w_hidden, 1),
        )

        # learned global edit magnitude (mirrors Solution A's alpha; here trainable)
        self.log_scale = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------ #
    # components
    # ------------------------------------------------------------------ #
    def _weights(self, v_ref, cond_col):
        """(b) per-condition non-negative weight w = softplus(MLP(v_ref, attr_emb)).

          v_ref    [B, dim]
          cond_col [B, C] long
        Returns w [B, C] >= 0. Dynamic (depends on v_ref) -> P2 + P4.
        """
        B, C = cond_col.shape
        emb = self.attr_emb(cond_col)                       # [B, C, w_emb]
        vref = v_ref.unsqueeze(1).expand(-1, C, -1)         # [B, C, dim]
        x = torch.cat([vref, emb], dim=-1)                  # [B, C, dim + w_emb]
        return F.softplus(self.weight_net(x)).squeeze(-1)   # [B, C]

    def _compose(self, v_ref, cond_col, cond_sign, cond_mask, w):
        """(c) signed, masked, weighted sum of unit directions -> edit [B, dim].

        Polarity is explicit: +1 adds d̂, -1 subtracts it (P1). Padded columns are
        zeroed by cond_mask, so the count of conditions is variable and the result
        is permutation-invariant.
        """
        dirs = F.normalize(self.D[cond_col], dim=-1)        # [B, C, dim] unit dirs
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

    def ortho_reg(self):
        """Orthogonality regularizer on D (the learned Gram–Schmidt, fixes P3).

        Mean squared off-diagonal of the unit-direction Gram matrix. Pushes
        correlated/opposite axes apart (e.g. Blond_Hair vs Black_Hair share the
        hair-colour axis) so composed constraints stop double-counting/cancelling.
        Added to the loss by the train loop (see losses.py note).
        """
        Dn = F.normalize(self.D, dim=1)                     # [n_attr, dim]
        gram = Dn @ Dn.t()                                  # [n_attr, n_attr]
        off = gram - torch.diag(torch.diagonal(gram))       # zero the diagonal
        n = gram.shape[0]
        return off.pow(2).sum() / (n * (n - 1))

    @torch.no_grad()
    def load_directions(self, axes):
        """Optional warm start: copy contrastive text axes into D (rows by column
        index). `axes` is a [n_attr, dim] tensor row-aligned to attribute columns.
        Convergence trick from next_steps.md; off by default (image-space-from-
        scratch is the whole point of T2 — it avoids the text cone)."""
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
        w = self._weights(v_ref, cond_col)                            # (b) [B, C]
        edit = self._compose(v_ref, cond_col, cond_sign, cond_mask, w)  # (a)+(c) [B, dim]
        v_q = self._apply_residual(v_ref, edit)                       # (d) residual
        return F.normalize(v_q, dim=1)


# --------------------------------------------------------------------------- #
# Smoke test: python -m src.solution_a.t2_directions
# Shape + invariants (mirror of the T1 smoke):
#   - output [B, 512], L2-normalized (v_q.norm(dim=1) ~ 1)
#   - permutation-invariance: shuffling the condition order leaves v_q unchanged
#   - mask honesty: extra padded columns leave v_q unchanged
#   - empty edit: cond_mask all False -> v_q ~ normalize(v_ref)
#   - ortho_reg: finite scalar, contributes a gradient to D
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

    # 5. ortho_reg: finite non-negative scalar
    reg = phi.ortho_reg()
    assert reg.ndim == 0 and torch.isfinite(reg) and reg >= 0, f"bad ortho_reg {reg}"
    print(f"  ok  ortho_reg (= {reg.item():.4f}, finite scalar)")

    # 6. backward: finite loss (+ortho), non-zero gradients on all parameters
    phi.train()
    v_q = phi(v_ref, cond_col, cond_sign, cond_mask)
    loss = (1 - (v_q * F.normalize(v_ref, dim=1)).sum(dim=1)).mean() + 0.1 * phi.ortho_reg()
    loss.backward()
    for name, p in phi.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0, \
            f"bad grad on {name}"
    print(f"  ok  backward (loss={loss.item():.4f}, all grads finite and non-zero)")

    print("smoke test passed.")


if __name__ == "__main__":
    _smoke()
