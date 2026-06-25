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
                 gate="vector", residual="ambient"):
        """
        n_attr    number of CelebA attributes (token table size)
        dim       CLIP feature dim (v_ref / v_q), = 512
        d_model   internal attention width
        n_heads   attention heads
        n_layers  cross-attention blocks (>1 = stacked, ablation)
        gate      "vector" (512-dim FiLM) | "scalar" (single coeff)
        residual  "ambient" | "tangent"  (tangent = Log/Exp around mean, ablation)
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
        # TODO: small MLP; output dim = `dim` (vector) or 1 (scalar)
        self.gate_net = None

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
        """(e) FiLM gate g(v_ref, conds) in [0,1], vector [B, dim] or scalar [B, 1]."""
        raise NotImplementedError

    def _apply_residual(self, v_ref, delta, gate):
        """v_q = v_ref + gate * delta (ambient) or the tangent-space variant."""
        raise NotImplementedError

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
        # (a) tokens   = self._build_tokens(cond_col, cond_sign, cond_mask)
        # (b) q_token  = self.q_proj(v_ref)
        # (c) summary  = self._attend(q_token, tokens, cond_mask)
        # (d) delta    = self.delta_proj(summary)
        # (e) gate     = self._gate(v_ref, summary)
        #     v_q      = self._apply_residual(v_ref, delta, gate)
        # return F.normalize(v_q, dim=1)
        raise NotImplementedError


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
