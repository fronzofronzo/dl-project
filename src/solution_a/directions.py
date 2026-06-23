"""Solution A — oriented contrastive direction axes (ambient).

d_attr = z("...with attr") - z("...without attr"): the text-text difference
cancels the template + u_0 + modality gap (Trager, Lemma 2.1 / Sec. 6), leaving
an ORIENTED absent->present axis. Polarity lives in the vector (sign). This is
the best-performing no-training ingredient (see results/findings_2026-06-19.md).
"""
from src.common.features import encode_data
from src.common.geometry import gram_schmidt, gram_schmidt_conflict


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


def contrastive_query(v_ref, pos_names, neg_names, axes, alpha=1.0, orth="off"):
    """v_ref + alpha*(Σ d_pos − Σ d_neg), poi L2-normalized. Edit in ambiente.

    orth (fix P3, ortogonalizzazione delle direzioni con segno prima di sommare):
      "off"      → somma cruda (default).
      "full"     → Gram–Schmidt su tutte le direzioni attive.
      "conflict" → Gram–Schmidt solo tra direzioni a segno OPPOSTO; coppie con lo
                   stesso segno (positivi correlati) tengono la componente condivisa.
                   Misurato come il miglior compromesso (vedi solution_a_sweep).

    I positivi vengono per primi (più importanti per l'identità target), poi i
    negativi: l'ordine conta in Gram–Schmidt."""
    signed = [axes[n] for n in pos_names] + [-axes[n] for n in neg_names]
    signs = [1] * len(pos_names) + [-1] * len(neg_names)
    if orth == "full":
        signed = gram_schmidt(signed)
    elif orth == "conflict":
        signed = gram_schmidt_conflict(signed, signs)
    elif orth not in ("off", False, None):
        raise ValueError(f"orth must be off|full|conflict, got {orth!r}")
    v = v_ref.float().clone()
    for d in signed:
        v = v + alpha * d
    return v / v.norm().clamp_min(1e-12)
