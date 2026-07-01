"""T1T2Phi — Cross-attention weighted direction adapter (Solution B hybrid).

Combines T2's interpretable per-attribute direction dictionary with T1's
cross-attention mechanism for computing interaction-aware condition weights.

T2's weight_net computes w_i = softplus(MLP(v_ref, attr_emb[i])) independently
per condition — it never sees the other conditions in the set (CLAY limit P3).

Fix: replace weight_net with a self-attention block where each condition token
attends to the others (with v_ref injected into every token), producing
per-condition weights that are both v_ref-conditioned and interaction-aware.

Pipeline (forward):
  (a) d̂_i(v_ref)  T2 I1: unit direction per attribute, conditioned on v_ref
                   d̂_i = normalize(D[col_i] + U[col_i]ᵀ·dir_net(v_ref, emb[col_i]))
  (b) t_i          condition token: attr_emb_attn[col_i] + sign_emb[sign_i] + q_proj(v_ref)
  (c) ctx_i        self-attention over {t_1,...,t_C}: each condition sees the others (P3)
  (d) w_i          per-condition scalar weight: softplus(w_proj(ctx_i))
  (e) edit         signed weighted sum: log_scale.exp() · Σ w_i · s_i · d̂_i
  (f) v_q          normalize(v_ref + edit)

Inherited from T2 (unchanged):
  - I1 input-conditioned directions (cond_dir=True)
  - I2 correlation-aware ortho_reg on D
  - I3 warm-start via load_directions()
  - I4 hybrid text-axis conditioning via set_text_axes()

See results/t1t2_design.md for the full design rationale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ATTR = 40
DIM = 512


class T1T2Phi(nn.Module):
    """T2 direction dictionary + T1 self-attention for condition weighting."""

    def __init__(self, n_attr=N_ATTR, dim=DIM,
                 # T2 direction params
                 w_emb=64, w_hidden=256, rank=16,
                 residual="ambient", cond_dir=True, hybrid=False,
                 # T1 attention params for weighting
                 d_model=256, n_heads=4):
        super().__init__()
        self.dim = dim
        self.residual = residual
        self.cond_dir = cond_dir
        self.hybrid = hybrid

        # ── T2 direction components ──────────────────────────────────────────
        self.D = nn.Parameter(torch.randn(n_attr, dim) * 0.02)
        self.attr_emb = nn.Embedding(n_attr, w_emb)        # feeds dir_net

        if cond_dir:
            self.U = nn.Parameter(torch.randn(n_attr, rank, dim) * (0.02 / rank ** 0.5))
            self.dir_net = nn.Sequential(
                nn.Linear(dim + w_emb, w_hidden), nn.ReLU(),
                nn.Linear(w_hidden, rank),
            )

        self.log_scale = nn.Parameter(torch.zeros(()))

        if hybrid:
            self.text_proj = nn.Linear(dim, dim, bias=False)
            self.text_gate = nn.Parameter(torch.zeros(()))
            self.register_buffer("text_axes", torch.zeros(n_attr, dim))

        # ── T1 attention components (replaces weight_net) ────────────────────
        self.attr_emb_attn = nn.Embedding(n_attr, d_model)  # separate from dir's attr_emb
        self.sign_emb = nn.Embedding(2, d_model)             # idx 0 → negative, 1 → positive
        self.q_proj = nn.Linear(dim, d_model)                # v_ref → d_model for injection
        self.cond_sa = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.w_proj = nn.Linear(d_model, 1)                  # ctx_i → scalar weight

    # ── T2 direction helpers ──────────────────────────────────────────────────

    def _feat(self, v_ref, cond_col):
        """Shared input for dir_net: [B, C, dim + w_emb]."""
        C = cond_col.shape[1]
        emb = self.attr_emb(cond_col)                        # [B, C, w_emb]
        vref = v_ref.unsqueeze(1).expand(-1, C, -1)         # [B, C, dim]
        return torch.cat([vref, emb], dim=-1)

    def _directions(self, cond_col, x):
        """(a) Unit directions per attribute, optionally v_ref-conditioned (I1).

        rigid (cond_dir=False): d̂ = normalize(D[col])
        I1   (cond_dir=True) : d̂ = normalize(D[col] + Σ_r coords_r · U[col,r])
        """
        d = self.D[cond_col]                                 # [B, C, dim]
        if self.cond_dir:
            coords = self.dir_net(x)                         # [B, C, rank]
            U = self.U[cond_col]                             # [B, C, rank, dim]
            d = d + (coords.unsqueeze(-1) * U).sum(dim=2)   # [B, C, dim]
        if self.hybrid:
            t = self.text_proj(self.text_axes[cond_col])     # [B, C, dim]
            d = d + self.text_gate * t
        return F.normalize(d, dim=-1)                        # [B, C, dim] unit

    # ── T1 attention weighting ────────────────────────────────────────────────

    def _attention_weights(self, v_ref, cond_col, cond_sign, cond_mask):
        """(b-d) Interaction-aware per-condition scalar weights.

        Each condition token is initialised as attr_emb + sign_emb, then v_ref
        is injected additively so every token is reference-aware. Self-attention
        lets the conditions see each other (P3). The resulting context vectors
        are projected to per-condition scalar weights.

        Returns w [B, C], padded positions are zero.
        """
        sign_idx = (cond_sign > 0).long()                   # +1→1, -1/0→0
        tokens = (self.attr_emb_attn(cond_col)              # [B, C, d_model]
                  + self.sign_emb(sign_idx))
        tokens = tokens * cond_mask.unsqueeze(-1)            # zero padded

        vr = self.q_proj(v_ref).unsqueeze(1)                # [B, 1, d_model]
        tokens = tokens + vr                                 # inject v_ref into every token

        ctx, _ = self.cond_sa(tokens, tokens, tokens,
                               key_padding_mask=~cond_mask)  # [B, C, d_model]
        ctx = torch.nan_to_num(ctx, nan=0.0)                 # guard all-masked rows

        w = F.softplus(self.w_proj(ctx).squeeze(-1))        # [B, C]
        return w * cond_mask.float()                         # zero padded

    # ── composition + residual ────────────────────────────────────────────────

    def _compose(self, dirs, cond_sign, cond_mask, w):
        """(e) Signed, masked, weighted sum of unit directions -> edit [B, dim]."""
        coeff = w * cond_sign * cond_mask.to(w.dtype)       # [B, C] signed, padded→0
        edit = (coeff.unsqueeze(-1) * dirs).sum(dim=1)      # [B, dim]
        return self.log_scale.exp() * edit

    def _apply_residual(self, v_ref, edit):
        """(f) v_q = v_ref + edit (ambient) or Exp-map geodesic (tangent)."""
        if self.residual == "ambient":
            return v_ref + edit
        vr = F.normalize(v_ref, dim=1)
        step = edit - (edit * vr).sum(dim=1, keepdim=True) * vr
        r = step.norm(dim=1, keepdim=True).clamp_min(1e-7)
        return vr * torch.cos(r) + (step / r) * torch.sin(r)

    # ── orthogonality regularizer (T2 I2, unchanged) ─────────────────────────

    def ortho_reg(self, target=None):
        """Decorrelation regularizer on the base dictionary D (CLAY limit P3).

        target=None  → push all off-diagonal to 0 (plain orthogonality)
        target=C     → push toward the empirical attribute correlation matrix C
        """
        Dn = F.normalize(self.D, dim=1)
        gram = Dn @ Dn.t()
        diff = gram if target is None else gram - target.to(gram)
        off = diff - torch.diag(torch.diagonal(diff))
        n = gram.shape[0]
        return off.pow(2).sum() / (n * (n - 1))

    # ── warm-start helpers (T2 I3 / I4, unchanged) ───────────────────────────

    @torch.no_grad()
    def load_directions(self, axes):
        """I3: warm-start D from image-space probe axes [n_attr, dim]."""
        self.D.copy_(axes.to(self.D).float())

    @torch.no_grad()
    def set_text_axes(self, axes):
        """Hybrid: store frozen CLIP text axes [n_attr, dim]."""
        if not self.hybrid:
            raise RuntimeError("set_text_axes called but hybrid=False")
        self.text_axes.copy_(axes.to(self.text_axes).float())

    # ── Φ contract ────────────────────────────────────────────────────────────

    def forward(self, v_ref, cond_col, cond_sign, cond_mask):
        """Φ contract: v_ref + signed conditions -> L2-normalized v_q [B, 512]."""
        x = self._feat(v_ref, cond_col)                      # [B, C, dim+w_emb]
        dirs = self._directions(cond_col, x)                 # (a) [B, C, dim]
        w = self._attention_weights(v_ref, cond_col,         # (b-d) [B, C]
                                    cond_sign, cond_mask)
        edit = self._compose(dirs, cond_sign, cond_mask, w)  # (e) [B, dim]
        v_q = self._apply_residual(v_ref, edit)              # (f)
        return F.normalize(v_q, dim=1)


# ── Smoke test: python -m src.solution_b.t1t2_phi ────────────────────────────

def _smoke():
    torch.manual_seed(0)
    B, C = 4, 3
    phi = T1T2Phi(hybrid=False)
    phi.eval()

    v_ref     = F.normalize(torch.randn(B, DIM), dim=1)
    cond_col  = torch.randint(0, N_ATTR, (B, C))
    cond_sign = torch.where(torch.rand(B, C) > 0.5, 1.0, -1.0)
    cond_mask = torch.ones(B, C, dtype=torch.bool)

    # 1. shape + L2-norm
    v_q = phi(v_ref, cond_col, cond_sign, cond_mask)
    assert v_q.shape == (B, DIM), f"shape {v_q.shape}"
    assert torch.allclose(v_q.norm(dim=1), torch.ones(B), atol=1e-5), "not L2-normalized"
    print("  ok  output shape and L2-norm")

    # 2. permutation-invariance
    perm = torch.randperm(C)
    v_q_p = phi(v_ref, cond_col[:, perm], cond_sign[:, perm], cond_mask[:, perm])
    assert torch.allclose(v_q, v_q_p, atol=1e-5), "not permutation-invariant"
    print("  ok  permutation-invariance")

    # 3. mask honesty
    col_pad  = torch.cat([cond_col,  torch.zeros(B, 2, dtype=torch.long)], dim=1)
    sign_pad = torch.cat([cond_sign, torch.zeros(B, 2)], dim=1)
    mask_pad = torch.cat([cond_mask, torch.zeros(B, 2, dtype=torch.bool)], dim=1)
    assert torch.allclose(v_q, phi(v_ref, col_pad, sign_pad, mask_pad), atol=1e-5), \
        "mask not honored"
    print("  ok  mask honesty (extra padded columns ignored)")

    # 4. empty edit -> v_q == normalize(v_ref)
    mask_empty = torch.zeros(B, C, dtype=torch.bool)
    v_q_empty = phi(v_ref, cond_col, cond_sign, mask_empty)
    assert torch.allclose(v_q_empty, F.normalize(v_ref, dim=1), atol=1e-4), \
        "empty-condition case failed"
    print("  ok  empty edit (v_q ≈ normalize(v_ref) when no conditions)")

    # 5. attention weights differ per condition (not all equal)
    w = phi._attention_weights(v_ref, cond_col, cond_sign, cond_mask)
    assert not torch.allclose(w[:, 0], w[:, 1], atol=1e-4), \
        "all conditions got identical weights — attention not working"
    print("  ok  per-condition weights differ (attention is active)")

    # 6. ortho_reg finite
    reg = phi.ortho_reg()
    assert reg.ndim == 0 and torch.isfinite(reg) and reg >= 0
    print(f"  ok  ortho_reg = {reg.item():.4f}")

    # 7. backward: finite, non-zero grads on all params
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
