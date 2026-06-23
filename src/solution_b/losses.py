"""Contrastive training objective for Φ: InfoNCE + identity anchor.

The frozen CLIP DB is never touched here — all tensors are cached features.
"""
import torch
import torch.nn.functional as F


def info_nce(v_q, pos_feat, hneg_feat, hneg_mask, tau=0.07):
    """InfoNCE over the in-batch positives + masked hard negatives.

      v_q       [B, 512]      composed queries (Φ output)
      pos_feat  [B, 512]      the positive image for each anchor (row i <-> anchor i)
      hneg_feat [B, K, 512]   hard negatives per anchor (polarity violators)
      hneg_mask [B, K]        bool, True for real hard negatives (False = pad)

    Candidate pool = the B positives (anchor i's target is column i) + every valid
    hard negative, shared across the batch. So anchor i's negatives are the other
    in-batch positives (easy) plus all valid hard negatives. Standard in-batch
    contrastive setup; cosine similarities (everything L2-normalized) over tau.
    """
    B, K, d = hneg_feat.shape
    vq = F.normalize(v_q, dim=1)
    pos = F.normalize(pos_feat, dim=1)
    hneg = F.normalize(hneg_feat.reshape(B * K, d), dim=1)

    candidates = torch.cat([pos, hneg], dim=0)             # [B + B*K, d]
    logits = vq @ candidates.t() / tau                     # [B, B + B*K]

    # mask out padded hard negatives (the first B columns — positives — are valid)
    valid = torch.cat([torch.ones(B, device=vq.device, dtype=torch.bool),
                       hneg_mask.reshape(B * K)], dim=0)    # [B + B*K]
    logits = logits.masked_fill(~valid.unsqueeze(0), float('-inf'))

    target = torch.arange(B, device=vq.device)             # own positive at column i
    return F.cross_entropy(logits, target)


def identity_anchor(v_q, v_ref):
    """1 - mean cos(v_q, v_ref): keeps the composed query near the reference."""
    return 1.0 - (F.normalize(v_q, dim=1) * F.normalize(v_ref, dim=1)).sum(dim=1).mean()


def total_loss(v_q, v_ref, pos_feat, hneg_feat, hneg_mask, tau=0.07, lam_id=0.1):
    """InfoNCE + lam_id * identity anchor. Returns (loss, parts dict).

    T2's orthogonality regularizer lives in the T2 model, added by the train loop.
    """
    nce = info_nce(v_q, pos_feat, hneg_feat, hneg_mask, tau=tau)
    ida = identity_anchor(v_q, v_ref)
    loss = nce + lam_id * ida
    return loss, {'info_nce': nce.detach(), 'identity': ida.detach()}
