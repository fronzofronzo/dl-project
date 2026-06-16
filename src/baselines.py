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


def attr_to_antonym_prompts(name):
    """Coppia con/senza generica per costruire un asse orientato.
    'Eyeglasses' -> ('a photo of a person with eyeglasses',
                     'a photo of a person without eyeglasses')."""
    readable = name.replace('_', ' ').lower()
    return (f"a photo of a person with {readable}",
            f"a photo of a person without {readable}")


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


# ---------------------------------------------------------------------------
# Vettori-direzione contrastivi (Fase 4, ingrediente 1) — ambient.
# d_attr = z("...with attr") - z("...without attr"): la differenza testo-testo
# cancella template + u_0 + modality gap (Trager, Lemma 2.1 / Sez. 6), lasciando
# un asse ORIENTATO assente->presente. La polarita' vive nel vettore (segno).
# ---------------------------------------------------------------------------
def build_direction_axes(all_names):
    """Per ogni attributo unico costruisce d = z_with - z_without (ambient).
    Encode di tutti i prompt (con+senza) in un solo batch. -> {name: [512]}."""
    names = sorted(set(all_names))
    if not names:
        return {}
    prompts = []
    for n in names:
        p_with, p_without = attr_to_antonym_prompts(n)
        prompts.append(p_with)
        prompts.append(p_without)
    _, z = encode_data(texts=prompts)            # [2*len(names), 512], L2-norm
    z = z.cpu()
    return {n: (z[2 * i] - z[2 * i + 1]).float() for i, n in enumerate(names)}


def contrastive_query(v_ref, pos_names, neg_names, axes, alpha=1.0):
    """v_ref + alpha*(Σ d_pos − Σ d_neg), poi L2-normalized. Edit in ambiente."""
    v = v_ref.float().clone()
    for n in pos_names:
        v = v + alpha * axes[n]
    for n in neg_names:
        v = v - alpha * axes[n]
    return v / v.norm().clamp_min(1e-12)
