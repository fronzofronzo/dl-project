"""T1 — Conditional cross-attention adapter Φ (Solution B, owner: person 1).

A small attention network looks at ALL signed conditions jointly and relative to
v_ref, then emits a FiLM-gated residual edit. Closes the four CLAY limits:

  P1 sign        -> sign_emb added to each attribute token (explicit polarity)
  P2 weighting   -> FiLM gate g(v_ref, conds) doses the edit
  P3 interaction -> cross-attention over the condition SET (not a blind sum)
  P4 dynamic     -> query = v_ref, attention conditioned on the actual reference

Implements the shared Φ contract from phi.py so it drops into train.py / run.py
unchanged:

    forward(v_ref, cond_col, cond_sign, cond_mask) -> v_q

CLIP stays frozen; only Φ is trained. The residual on v_ref preserves identity.

NOTE: pad vs attribute-0 — cond_col==0 is BOTH padding and attribute 0
(5_o_Clock_Shadow). The truth is cond_mask; never use cond_col==0 as "is pad".

Structure only — components are stubbed for later implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ATTR = 40
DIM = 512


class T1Phi(nn.Module):
    """Cross-attention fusion module Φ_T1.

    Pipeline (forward):
      (a) signed condition tokens   attr_emb[col] + sign_emb[sign]
      (b) query token from v_ref    q_proj(v_ref)
      (c) cross-attention           query=v_ref token, key/value=condition tokens
      (d) project Δ to CLIP space   delta_proj(attn_out)
      (e) FiLM-gated residual       v_q = v_ref + g(v_ref, conds) * Δ  -> normalize
    """

    def __init__(self, n_attr=N_ATTR, dim=DIM, d_model=256, n_heads=4, n_layers=1,
                 gate="vector", residual="ambient", gate_hidden=256):
        """
        n_attr      number of CelebA attributes (token table size)
        dim         CLIP feature dim (v_ref / v_q), = 512
        d_model     internal attention width
        n_heads     attention heads
        n_layers    cross-attention blocks (>1 = stacked, ablation)
        gate        "vector" (512-dim FiLM) | "scalar" (single coeff)
        residual    "ambient" | "tangent"  (tangent = Exp map at v_ref, ablation)
        gate_hidden hidden width of the FiLM gate MLP
        """
        super().__init__()
        self.dim = dim
        self.d_model = d_model
        self.gate_kind = gate
        self.residual = residual

        # (a) signed condition tokens
        self.attr_emb = nn.Embedding(n_attr, d_model)
        self.sign_emb = nn.Embedding(2, d_model)        # idx 0 -> negative, 1 -> positive

        # (b) query token from v_ref
        self.q_proj = nn.Linear(dim, d_model)

        # (c) cross-attention block(s)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # (d) project attention output -> CLIP-space edit Δ
        self.delta_proj = nn.Linear(d_model, dim)

        # (e) FiLM gate net: [v_ref ; cond summary] -> sigmoid gate (vector or scalar)
        gate_out = dim if gate == "vector" else 1
        self.gate_net = nn.Sequential(
            nn.Linear(dim + d_model, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_out),
            nn.Sigmoid(),                               # gate in [0, 1]
        )

    # ------------------------------------------------------------------ #
    # components
    # ------------------------------------------------------------------ #
    def _build_tokens(self, cond_col, cond_sign, cond_mask):
        """(a) signed condition tokens.

          cond_col  [B, C] long   -> attr_emb
          cond_sign [B, C] float  -> {-1,+1} mapped to sign_emb index {0,1}
          cond_mask [B, C] bool   -> zero out padded tokens
        Returns tokens [B, C, d_model].
        """
        attr = self.attr_emb(cond_col)                    # [B, C, d_model]
        sign_idx = (cond_sign > 0).long()                 # +1 -> 1, -1 -> 0
        sign = self.sign_emb(sign_idx)                    # [B, C, d_model]
        tokens = attr + sign                              # signed tokens
        tokens = tokens * cond_mask.unsqueeze(-1)         # zero out padded tokens
        return tokens   
        raise NotImplementedError

    def _attend(self, q_token, tokens, cond_mask):
        """(c) cross-attention: query=v_ref token, key/value=condition tokens.

          q_token   [B, d_model]
          tokens    [B, C, d_model]
          cond_mask [B, C] bool   -> key_padding_mask = ~cond_mask
        Returns attended summary [B, d_model].
        """
        q = q_token.unsqueeze(1)                          # [B, 1, d_model]
        key_padding_mask = ~cond_mask                     # [B, C]
        attn_out, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        return attn_out.squeeze(1)                         # [B, d_model]

    def _gate(self, v_ref, cond_summary):
        """(e) FiLM gate g(v_ref, conds) in [0,1], vector [B, dim] or scalar [B, 1].

        Doses the edit per input -> dynamic weighting (P2). cond_summary is the
        attended condition vector from (c), so the gate sees both the reference and
        what the conditions ask for.
        """
        x = torch.cat([v_ref, cond_summary], dim=1)       # [B, dim + d_model]
        return self.gate_net(x)                            # [B, dim] or [B, 1]

    def _apply_residual(self, v_ref, delta, gate):
        """Compose v_q from the reference and the gated edit. The `+ v_ref` residual
        preserves identity; `gate` doses Δ (broadcast for the scalar gate).

          ambient: v_q = v_ref + gate * delta
          tangent: geodesic step from v_ref along the tangent-projected edit
                   (Exp map anchored at v_ref); findings A: tangent ~= ambient.

        Returns un-normalized v_q; forward() applies the final L2-normalize.
        """
        step = gate * delta                                # [B, dim], gate broadcasts
        if self.residual == "ambient":
            return v_ref + step

        # tangent: anchor μ = v_ref per row, drop the radial part, Exp back
        vr = F.normalize(v_ref, dim=1)
        step = step - (step * vr).sum(dim=1, keepdim=True) * vr     # tangent component
        r = step.norm(dim=1, keepdim=True).clamp_min(1e-7)
        return vr * torch.cos(r) + (step / r) * torch.sin(r)

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
        tokens = self._build_tokens(cond_col, cond_sign, cond_mask)   # (a) [B, C, d_model]
        q_token = self.q_proj(v_ref)                                  # (b) [B, d_model]
        summary = self._attend(q_token, tokens, cond_mask)            # (c) [B, d_model]

        # rows with NO condition: every key is masked -> softmax over all -inf -> NaN.
        # null the summary there so the gate/edit see a clean zero (empty-edit case).
        summary = torch.nan_to_num(summary, nan=0.0)

        delta = self.delta_proj(summary)                              # (d) [B, dim]
        gate = self._gate(v_ref, summary)                             # (e) [B, dim]/[B, 1]

        # force a no-op edit when there are no real conditions -> v_q == normalize(v_ref)
        has_cond = cond_mask.any(dim=1, keepdim=True).to(delta.dtype)
        delta = delta * has_cond

        v_q = self._apply_residual(v_ref, delta, gate)               # residual + identity
        return F.normalize(v_q, dim=1)


# --------------------------------------------------------------------------- #
# Smoke test: python -m src.solution_b.t1_attention
# Shape + invariants (fill in once forward is implemented):
#   - output [B, 512], L2-normalized (v_q.norm(dim=1) ~ 1)
#   - permutation-invariance: shuffling the condition order leaves v_q unchanged
#   - mask honesty: extra padded columns leave v_q unchanged
#   - empty edit: cond_mask all False -> v_q ~ normalize(v_ref)
#   - backward: finite, non-zero grads on every parameter
# --------------------------------------------------------------------------- #
def _smoke():
    raise NotImplementedError


if __name__ == "__main__":
    _smoke()
