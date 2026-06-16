"""Naive ambient-arithmetic baseline (lower bound).

v_target = v_ref + Σ t_pos − Σ t_neg, composed directly in ℝ⁵¹².
Deliberately ignores spherical geometry and the modality gap (text added to
image vectors in ambient space) — this is the floor the geometric / trained
methods must beat. Antonym prompts and sign-aware axes belong to Solution A.
"""
import torch

from feature_extraction import encode_data


def attr_to_prompt(name):
    """'Heavy_Makeup' -> 'a photo of a person with heavy makeup'."""
    readable = name.replace('_', ' ').lower()
    return f"a photo of a person with {readable}"


def precompute_text(all_names):
    """Encode each unique attribute prompt once -> {name: L2-normed [512]}."""
    names = sorted(set(all_names))
    if not names:
        return {}
    _, txt_z = encode_data(texts=[attr_to_prompt(n) for n in names])
    return {n: txt_z[i].cpu() for i, n in enumerate(names)}


def naive_query(v_ref, pos_names, neg_names, text_cache):
    """v_ref + Σ t_pos − Σ t_neg, then L2-normalized. All tensors on CPU."""
    v = v_ref.float().clone()
    for n in pos_names:
        v = v + text_cache[n].float()
    for n in neg_names:
        v = v - text_cache[n].float()
    return v / v.norm().clamp_min(1e-12)
