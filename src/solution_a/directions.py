"""Solution A — oriented contrastive direction axes (ambient).

d_attr = z("...with attr") - z("...without attr"): the text-text difference
cancels the template + u_0 + modality gap (Trager, Lemma 2.1 / Sec. 6), leaving
an ORIENTED absent->present axis. Polarity lives in the vector (sign). This is
the best-performing no-training ingredient (see results/findings_2026-06-19.md).
"""
from src.common.features import encode_data


def attr_to_antonym_prompts(name):
    """Coppia con/senza generica per costruire un asse orientato.
    'Eyeglasses' -> ('a photo of a person with eyeglasses',
                     'a photo of a person without eyeglasses')."""
    readable = name.replace('_', ' ').lower()
    return (f"a photo of a person with {readable}",
            f"a photo of a person without {readable}")


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
