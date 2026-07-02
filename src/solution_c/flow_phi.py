"""Solution C — Φ-Flow: conditional flow matching on the CLIP hypersphere.

WHY A FLOW. T1 and T2 are ONE-SHOT editors: whatever the module, the edit is a
single Δ added to v_ref, so antagonistic-but-correlated constraints get resolved
by SUMMING nearly antiparallel vectors — they cancel. Measured: the query
`+Wearing_Lipstick, -Heavy_Makeup, +Smiling` scores 0.000 for EVERY method
(naive, CLAY, Solution A, T1, T2-hybrid, ensemble), and the one-shot unified
T1T2Phi is a documented negative result. Φ-Flow reframes composition as a
TRAJECTORY: a learned velocity field u_θ(v_t, t | conditions) on the unit sphere
S^{d-1}, integrated from v_ref for N Euler steps. Re-evaluating the field at the
CURRENT point lets the model SEQUENCE edits (+Lipstick first, then −Heavy_Makeup
from the new point) instead of summing them once.

CLAY limits:
  P1 sign        -> signed dictionary readout (+1 adds a direction, −1 subtracts)
  P2 weighting   -> per-condition, per-STEP weights w_i(v_t, t, all conditions)
  P3 interaction -> self-attention across condition tokens at every step;
                    conflicts resolved sequentially along the path, not summed once
  P4 dynamic     -> the field is a function of the current point v_t: Φ is
                    literally a dynamical system conditioned on the reference

WARM START (de-risk). The direction/weight branch is name-compatible with T2Phi
(src/solution_a/t2_directions.py): D, U, attr_emb, dir_net, weight_net,
log_scale and the hybrid text bridge load directly from
results/phi_t2_hybrid.pt. The NEW heads (interaction correction w_corr, free
residual free_proj) are ZERO-initialized, so at init the velocity field is
exactly the tangent projection of T2-hybrid's edit field: the flow starts from
the strongest existing model and training can only reshape it. One Euler step ≈
T2 — the one-shot editors are the single-step special case of the flow.

HYBRID GUIDANCE (inference-time, optional). Classifier guidance from 40 frozen
linear attribute probes (src/solution_c/probes.py): integration adds
λ · ∇_v Σ_i log p(constraint_i | v_t), tangent-projected. Query-side only; the
frozen DB is never touched. CLIP stays frozen throughout; only Φ trains.

Φ contract (drops into src/solution_b/run.eval_phi unchanged):

    forward(v_ref, cond_col, cond_sign, cond_mask) -> v_q  [B, 512] L2-normalized

NOTE: pad vs attribute-0 — cond_col==0 is BOTH padding and attribute 0
(5_o_Clock_Shadow). The truth is cond_mask; never use cond_col==0 as "is pad".
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ATTR = 40
DIM = 512


# --------------------------------------------------------------------------- #
# spherical helpers (batched; the scalar versions live in src/common/geometry)
# --------------------------------------------------------------------------- #
def slerp_velocity(v0, v1, t):
    """Point and velocity of the geodesic slerp path at time t (CFM targets).

      x_t      = [sin((1−t)θ)·v0 + sin(tθ)·v1] / sin θ,   θ = arccos(v0·v1)
      dx_t/dt  = θ·[−cos((1−t)θ)·v0 + cos(tθ)·v1] / sin θ

    dx_t/dt is tangent to the sphere at x_t with constant norm θ (check: at t=0
    the dot with v0 is θ(−cosθ + cosθ)/sinθ = 0). Near-coincident endpoints
    (θ < 1e-3; the cos clamp alone already forces θ ≳ 4.5e-4) collapse to
    x_t = v0, velocity 0.

    v0, v1: [B, d] unit vectors.  t: [B, 1] in [0, 1].  -> (x_t [B,d], u [B,d])
    """
    cos = (v0 * v1).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos)                                  # [B, 1]
    sin = torch.sin(theta).clamp_min(1e-7)
    x_t = (torch.sin((1 - t) * theta) * v0 + torch.sin(t * theta) * v1) / sin
    u = theta * (-torch.cos((1 - t) * theta) * v0 + torch.cos(t * theta) * v1) / sin
    small = theta < 1e-3
    x_t = torch.where(small, v0, F.normalize(x_t, dim=-1))
    u = torch.where(small, torch.zeros_like(u), u)
    return x_t, u


def exp_step(v, step):
    """Geodesic retraction: move from unit v along tangent vector `step`.

      Exp_v(s) = cos(‖s‖)·v + sin(‖s‖)·s/‖s‖     (stays on S^{d-1})
    """
    r = step.norm(dim=-1, keepdim=True)
    moved = torch.cos(r) * v + torch.sin(r) * step / r.clamp_min(1e-8)
    out = torch.where(r > 1e-8, moved, v)
    return F.normalize(out, dim=-1)


def tangent_project(u, v):
    """Drop the radial component: u − (u·v)·v, tangent at unit v."""
    return u - (u * v).sum(-1, keepdim=True) * v


class FlowPhi(nn.Module):
    """Velocity-field fusion module Φ_flow.

    velocity(v_t, t, conds):
      (a) T2 direction branch (warm-startable, names match T2Phi exactly)
            d̂_i(v_t) = normalize(D[c_i] + U[c_i]ᵀ·dir_net(v_t, attr_emb[c_i])
                                  [+ text_gate·text_proj(text_axes[c_i])])
      (b) base per-condition magnitude (T2's independent head, on the CURRENT v_t)
            w_base_i = weight_net(v_t, attr_emb[c_i])
      (c) interaction/time correction (NEW, zero-init -> inert at warm start)
            tok_i = attr_emb_attn[c_i] + sign_emb[s_i] + inj(v_t, fourier(t))
            ctx   = self_attention(tok)            # every condition sees the others
            w_i   = softplus(w_base_i + w_corr(ctx_i))
      (d) signed readout + optional free residual (zero-init), tangent-projected
            u = Π_{T_{v_t}}[ exp(log_scale)·Σ_i w_i·s_i·d̂_i + free_proj(mean ctx) ]

    forward() integrates N Euler steps of `velocity` from v_ref (plus optional
    probe guidance) and returns the endpoint — the composed query v_q.
    """

    def __init__(self, n_attr=N_ATTR, dim=DIM, w_emb=64, w_hidden=256, rank=16,
                 d_model=256, n_heads=4, t_freqs=8, hybrid=True,
                 free_residual=True, n_steps=8, guidance=0.0, horizon=1.0):
        """
        n_attr, dim     CelebA attributes / CLIP feature dim
        w_emb, w_hidden, rank   T2 direction-branch sizes (MUST match the T2
                        checkpoint for the warm start to load)
        d_model, n_heads  width/heads of the interaction encoder (new branch)
        t_freqs         Fourier frequencies for the time embedding (2·t_freqs dims)
        hybrid          keep T2's frozen text-axis bridge (needed to load the
                        phi_t2_hybrid.pt checkpoint; harmless otherwise)
        free_residual   zero-init free velocity term outside the dictionary span
        n_steps         default Euler steps at inference (mutable attribute)
        guidance        default probe-guidance λ at inference (mutable attribute)
        horizon         default integration horizon T ∈ (0, 1]: integrate to time
                        T instead of 1 — a continuous edit-strength dial (T=1
                        reaches the CFM endpoint, smaller T stays closer to
                        v_ref; mutable attribute, swept at inference)
        """
        super().__init__()
        self.dim = dim
        self.hybrid = hybrid
        self.free_residual = free_residual
        self.n_steps = n_steps
        self.guidance = guidance
        self.horizon = horizon
        self.t_freqs = t_freqs

        # ---- (a)+(b) T2-compatible branch: names MUST mirror T2Phi ---------- #
        self.D = nn.Parameter(torch.randn(n_attr, dim) * 0.02)
        self.attr_emb = nn.Embedding(n_attr, w_emb)
        self.weight_net = nn.Sequential(
            nn.Linear(dim + w_emb, w_hidden), nn.ReLU(), nn.Linear(w_hidden, 1))
        self.U = nn.Parameter(torch.randn(n_attr, rank, dim) * (0.02 / rank ** 0.5))
        self.dir_net = nn.Sequential(
            nn.Linear(dim + w_emb, w_hidden), nn.ReLU(), nn.Linear(w_hidden, rank))
        if hybrid:
            self.text_proj = nn.Linear(dim, dim, bias=False)
            self.text_gate = nn.Parameter(torch.zeros(()))
            self.register_buffer("text_axes", torch.zeros(n_attr, dim))
        self.log_scale = nn.Parameter(torch.zeros(()))

        # ---- (c) interaction/time correction (new, inert at init) ----------- #
        self.attr_emb_attn = nn.Embedding(n_attr, d_model)
        self.sign_emb = nn.Embedding(2, d_model)             # 0 -> negative, 1 -> positive
        self.inj = nn.Linear(dim + 2 * t_freqs, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=w_hidden, batch_first=True)
        self.w_corr = nn.Linear(d_model, 1)
        nn.init.zeros_(self.w_corr.weight); nn.init.zeros_(self.w_corr.bias)

        # ---- (d) free residual velocity (new, inert at init) ---------------- #
        if free_residual:
            self.free_proj = nn.Linear(d_model, dim)
            nn.init.zeros_(self.free_proj.weight); nn.init.zeros_(self.free_proj.bias)

        # ---- probe guidance (frozen buffers, set via set_probes) ------------ #
        self.register_buffer("probe_W", torch.zeros(n_attr, dim))
        self.register_buffer("probe_b", torch.zeros(n_attr))

    # ------------------------------------------------------------------ #
    # warm start & probes
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def warm_start_from_t2(self, t2_state):
        """Copy a trained T2Phi state dict into the direction/weight branch.

        The correction heads stay zero, so right after this call
        velocity(v, t) == Π_tangent(T2 edit field at v) for every t. Raises if
        the checkpoint carries keys this model cannot place (a real mismatch),
        while OUR new heads being absent from the checkpoint is expected.
        """
        res = self.load_state_dict(t2_state, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f"T2 checkpoint has unloadable keys: {res.unexpected_keys}")
        loaded = set(t2_state) - set(res.missing_keys)
        return sorted(loaded)

    @torch.no_grad()
    def set_probes(self, W, b):
        """Store frozen linear attribute probes (from probes.py) for guidance."""
        self.probe_W.copy_(W.to(self.probe_W).float())
        self.probe_b.copy_(b.to(self.probe_b).float())

    def has_probes(self):
        return bool(self.probe_W.abs().sum() > 0)

    # ------------------------------------------------------------------ #
    # velocity field
    # ------------------------------------------------------------------ #
    def _fourier_t(self, t):
        """Time embedding: [sin(2π·2^k·t), cos(2π·2^k·t)], k = 0..t_freqs-1."""
        freqs = 2.0 ** torch.arange(self.t_freqs, device=t.device, dtype=t.dtype)
        ang = 2 * math.pi * t.unsqueeze(-1) * freqs          # [B, F]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def _feat(self, v, cond_col):
        """T2's shared head input [B, C, dim + w_emb] = (v broadcast || attr_emb)."""
        C = cond_col.shape[1]
        emb = self.attr_emb(cond_col)
        return torch.cat([v.unsqueeze(1).expand(-1, C, -1), emb], dim=-1)

    def _directions(self, cond_col, x):
        """T2's input-conditioned unit directions, evaluated at the CURRENT v_t."""
        d = self.D[cond_col]                                 # [B, C, dim]
        coords = self.dir_net(x)                             # [B, C, rank]
        d = d + (coords.unsqueeze(-1) * self.U[cond_col]).sum(dim=2)
        if self.hybrid:
            d = d + self.text_gate * self.text_proj(self.text_axes[cond_col])
        return F.normalize(d, dim=-1)

    def velocity(self, v, t, cond_col, cond_sign, cond_mask):
        """u_θ(v, t | conds) ∈ T_v S^{d-1}.

          v          [B, 512] current point, unit
          t          [B] flow time in [0, 1]
          cond_*     the Φ contract condition tensors
          -> u       [B, 512] tangent at v
        """
        x = self._feat(v, cond_col)                          # [B, C, dim+w_emb]
        w_base = self.weight_net(x).squeeze(-1)              # [B, C] pre-softplus

        # interaction/time tokens -> per-condition weight correction
        tok = (self.attr_emb_attn(cond_col)
               + self.sign_emb((cond_sign > 0).long())
               + self.inj(torch.cat([v, self._fourier_t(t)], dim=-1)).unsqueeze(1))
        tok = tok * cond_mask.unsqueeze(-1)
        ctx = self.encoder(tok, src_key_padding_mask=~cond_mask)
        ctx = torch.nan_to_num(ctx, nan=0.0)                 # rows with 0 real conditions

        w = F.softplus(w_base + self.w_corr(ctx).squeeze(-1))          # [B, C] >= 0
        dirs = self._directions(cond_col, x)                           # [B, C, dim]
        coeff = w * cond_sign * cond_mask.to(w.dtype)                  # signed, pads -> 0
        u = self.log_scale.exp() * (coeff.unsqueeze(-1) * dirs).sum(dim=1)

        if self.free_residual:
            denom = cond_mask.sum(dim=1, keepdim=True).clamp_min(1).to(ctx.dtype)
            pooled = (ctx * cond_mask.unsqueeze(-1)).sum(dim=1) / denom
            u = u + self.free_proj(pooled)

        has_cond = cond_mask.any(dim=1, keepdim=True).to(u.dtype)
        return tangent_project(u * has_cond, v)

    def guidance_grad(self, v, cond_col, cond_sign, cond_mask):
        """Closed-form ∇_v Σ_i log p(constraint_i | v) from the linear probes.

        For probe logit z_i = W[c_i]·v + b[c_i] and sign s_i ∈ {+1, −1}:
          ∂/∂v log σ(s_i·z_i) = s_i · σ(−s_i·z_i) · W[c_i]
        Padded slots contribute 0 (mask). Tangent-projected by the caller.
        """
        z = v @ self.probe_W.t() + self.probe_b              # [B, n_attr]
        zc = z.gather(1, cond_col)                           # [B, C]
        coeff = cond_sign * torch.sigmoid(-cond_sign * zc) * cond_mask.to(z.dtype)
        return (coeff.unsqueeze(-1) * self.probe_W[cond_col]).sum(dim=1)

    # ------------------------------------------------------------------ #
    # contract
    # ------------------------------------------------------------------ #
    def forward(self, v_ref, cond_col, cond_sign, cond_mask,
                n_steps=None, guidance=None, horizon=None):
        """Integrate the flow from v_ref for n_steps Euler steps up to time T -> v_q.

          v_ref     [B, 512]  reference image features (L2-norm, frozen CLIP)
          cond_col  [B, C]    long, attribute column per condition (0 = pad)
          cond_sign [B, C]    float, +1 additive / −1 subtractive / 0 pad
          cond_mask [B, C]    bool, True for real conditions
          n_steps / guidance / horizon  optional overrides of the instance defaults
          -> v_q    [B, 512]  composed query, L2-normalized

        Each step: v ← Exp_v( (T/N)·[u_θ + λ·‖u_θ‖·ĝ] ), ĝ = unit tangent probe
        gradient. λ is a RELATIVE mix (fraction of the field's own magnitude):
        the raw probe gradient has arbitrary norm (unnormalized logistic W) and
        measured orders of magnitude above ‖u_θ‖ ≈ θ ≤ π — unscaled it destroys
        the trajectory. T < 1 stops the trajectory early — a partial edit that
        stays closer to v_ref (edit-strength dial; the CFM path parametrization
        makes time the natural magnitude axis). Differentiable end to end.
        """
        N = int(n_steps if n_steps is not None else self.n_steps)
        lam = float(guidance if guidance is not None else self.guidance)
        T = float(horizon if horizon is not None else self.horizon)
        v = F.normalize(v_ref.float(), dim=1)
        for k in range(N):
            t = torch.full((v.shape[0],), k * T / N, device=v.device, dtype=v.dtype)
            u = self.velocity(v, t, cond_col, cond_sign, cond_mask)
            if lam != 0.0 and self.has_probes():
                g = tangent_project(self.guidance_grad(v, cond_col, cond_sign, cond_mask), v)
                g_hat = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                u = u + lam * u.norm(dim=-1, keepdim=True) * g_hat
            v = exp_step(v, u * (T / N))
        return F.normalize(v, dim=1)


# --------------------------------------------------------------------------- #
# Smoke test: python -m src.solution_c.flow_phi
#   - slerp targets: on-sphere path, tangent velocity, norm θ
#   - velocity: shape, tangency, mask honesty, permutation invariance
#   - empty condition set -> v_q == normalize(v_ref)
#   - warm start: init velocity == tangent projection of T2-hybrid's edit field
#   - guidance_grad matches autograd
#   - forward: unit norm, differentiable, all params get gradient
# --------------------------------------------------------------------------- #
def _smoke():
    torch.manual_seed(0)
    B, C = 4, 3

    # 1. slerp/velocity targets
    v0 = F.normalize(torch.randn(B, DIM), dim=1)
    v1 = F.normalize(torch.randn(B, DIM), dim=1)
    t = torch.rand(B, 1)
    x_t, u = slerp_velocity(v0, v1, t)
    theta = torch.acos((v0 * v1).sum(-1).clamp(-1, 1))
    assert torch.allclose(x_t.norm(dim=1), torch.ones(B), atol=1e-5)
    assert (u * x_t).sum(-1).abs().max() < 1e-4, "slerp velocity not tangent"
    assert torch.allclose(u.norm(dim=1), theta, atol=1e-4), "velocity norm != theta"
    x_same, u_same = slerp_velocity(v0, v0, t)
    assert torch.allclose(x_same, v0) and u_same.abs().max() == 0
    print("  ok  slerp path + velocity targets (on-sphere, tangent, norm θ)")

    phi = FlowPhi()
    phi.eval()
    v_ref = F.normalize(torch.randn(B, DIM), dim=1)
    cond_col = torch.randint(0, N_ATTR, (B, C))
    cond_sign = torch.where(torch.rand(B, C) > 0.5, 1.0, -1.0)
    cond_mask = torch.ones(B, C, dtype=torch.bool)

    # 2. velocity tangency + output shape
    tt = torch.rand(B)
    u = phi.velocity(v_ref, tt, cond_col, cond_sign, cond_mask)
    assert u.shape == (B, DIM)
    assert (u * v_ref).sum(-1).abs().max() < 1e-4, "velocity not tangent"
    print("  ok  velocity shape + tangency")

    # 3. permutation invariance + mask honesty + empty set
    perm = torch.randperm(C)
    u_perm = phi.velocity(v_ref, tt, cond_col[:, perm], cond_sign[:, perm], cond_mask[:, perm])
    assert torch.allclose(u, u_perm, atol=1e-5), "not permutation-invariant"
    col_pad = torch.cat([cond_col, torch.zeros(B, 2, dtype=torch.long)], dim=1)
    sign_pad = torch.cat([cond_sign, torch.zeros(B, 2)], dim=1)
    mask_pad = torch.cat([cond_mask, torch.zeros(B, 2, dtype=torch.bool)], dim=1)
    u_pad = phi.velocity(v_ref, tt, col_pad, sign_pad, mask_pad)
    assert torch.allclose(u, u_pad, atol=1e-5), "mask not honored"
    v_empty = phi(v_ref, cond_col, cond_sign, torch.zeros(B, C, dtype=torch.bool))
    assert torch.allclose(v_empty, F.normalize(v_ref, dim=1), atol=1e-5)
    print("  ok  permutation invariance, mask honesty, empty edit")

    # 4. warm start == T2 edit field (tangent-projected), at every t
    from src.solution_a.t2_directions import T2Phi
    t2 = T2Phi(hybrid=True)
    t2.set_text_axes(F.normalize(torch.randn(N_ATTR, DIM), dim=1))
    with torch.no_grad():
        t2.text_gate.fill_(0.3)                      # exercise the hybrid path too
    t2.eval()
    phi_w = FlowPhi(hybrid=True)
    phi_w.eval()
    loaded = phi_w.warm_start_from_t2(t2.state_dict())
    assert "D" in loaded and "text_axes" in loaded and "weight_net.0.weight" in loaded
    x = t2._feat(v_ref, cond_col)
    edit_t2 = t2._compose(t2._directions(cond_col, x), cond_sign, cond_mask, t2._weights(x))
    expected = tangent_project(edit_t2, v_ref)
    for tv in (torch.zeros(B), torch.full((B,), 0.7)):
        u_w = phi_w.velocity(v_ref, tv, cond_col, cond_sign, cond_mask)
        assert torch.allclose(u_w, expected, atol=1e-5), "warm start != T2 edit field"
    print("  ok  warm start reproduces T2-hybrid edit field (corrections inert)")

    # 5. guidance gradient matches autograd
    phi.set_probes(torch.randn(N_ATTR, DIM), torch.randn(N_ATTR))
    v = v_ref.clone().requires_grad_(True)
    z = v @ phi.probe_W.t() + phi.probe_b
    zc = z.gather(1, cond_col)
    logp = (F.logsigmoid(cond_sign * zc) * cond_mask).sum()
    logp.backward()
    g = phi.guidance_grad(v_ref, cond_col, cond_sign, cond_mask)
    assert torch.allclose(g, v.grad, atol=1e-5), "guidance grad != autograd"
    print("  ok  closed-form guidance gradient (matches autograd)")

    # 6. forward: unit output, guidance changes it, backward reaches every param
    phi.train()
    v_q = phi(v_ref, cond_col, cond_sign, cond_mask, n_steps=4)
    assert v_q.shape == (B, DIM)
    assert torch.allclose(v_q.norm(dim=1), torch.ones(B), atol=1e-5)
    v_g = phi(v_ref, cond_col, cond_sign, cond_mask, n_steps=4, guidance=1.0)
    assert not torch.allclose(v_q, v_g, atol=1e-5), "guidance had no effect"
    v_h0 = phi(v_ref, cond_col, cond_sign, cond_mask, n_steps=4, horizon=1e-9)
    assert torch.allclose(v_h0, F.normalize(v_ref, dim=1), atol=1e-4), "horizon->0 != v_ref"
    v_h5 = phi(v_ref, cond_col, cond_sign, cond_mask, n_steps=4, horizon=0.5)
    ang_half = (v_h5 * F.normalize(v_ref, dim=1)).sum(1)
    ang_full = (v_q * F.normalize(v_ref, dim=1)).sum(1)
    assert (ang_half >= ang_full - 1e-5).all(), "shorter horizon should stay closer to v_ref"
    loss = (1 - (v_q * F.normalize(v_ref, dim=1)).sum(1)).mean()
    loss.backward()
    for name, p in phi.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad on {name}"
    n_params = sum(p.numel() for p in phi.parameters())
    print(f"  ok  forward/backward through {4} integration steps ({n_params:,} params)")

    print("smoke test passed.")


if __name__ == "__main__":
    _smoke()
