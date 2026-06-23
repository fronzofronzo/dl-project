"""Φ fusion-module interface (shared by T1/T2) + a minimal MLP baseline.

The CONTRACT every Φ implements (so T1, T2 and the MLP baseline are swappable in
the training loop):

    forward(v_ref, cond_col, cond_sign, cond_mask) -> v_q

      v_ref     [B, 512]  reference image features (L2-normalized, frozen CLIP)
      cond_col  [B, C]    long, attribute column index per condition (0=pad)
      cond_sign [B, C]    float, +1 (additive) / -1 (subtractive) / 0 (pad)
      cond_mask [B, C]    bool, True for real conditions
      -> v_q    [B, 512]  composed query, L2-normalized, query-side only

CLIP stays frozen; only Φ is trained. The residual on v_ref preserves identity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ATTR = 40
DIM = 512


class MLPPhi(nn.Module):
    """Baseline Φ: signed attribute embeddings -> masked mean-pool -> concat with
    v_ref -> MLP -> residual -> normalize.

    Deliberately simple: no attention (T1), no direction dictionary (T2). Serves
    as the smoke test of the sampler->Φ->loss path AND as the "MLP vs
    attention/directions" ablation required by the spec.
    """

    def __init__(self, n_attr=N_ATTR, dim=DIM, hidden=512):
        super().__init__()
        self.attr_emb = nn.Embedding(n_attr, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, v_ref, cond_col, cond_sign, cond_mask):
        # signed condition embeddings, padded entries zeroed by the mask
        emb = self.attr_emb(cond_col)                      # [B, C, dim]
        emb = emb * cond_sign.unsqueeze(-1)                # apply polarity
        emb = emb * cond_mask.unsqueeze(-1)                # zero padding
        denom = cond_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        pooled = emb.sum(dim=1) / denom                    # [B, dim] masked mean

        delta = self.mlp(torch.cat([v_ref, pooled], dim=1))  # [B, dim]
        v_q = v_ref + delta                                # residual preserves identity
        return F.normalize(v_q, dim=1)
