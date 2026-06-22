"""CLAY stacked-SVD baseline — the real SOTA to beat.

CLAY (Lim et al., CVPR 2026) does training-free *conditional visual similarity
modulation*. Pipeline reimplemented here from the paper description (no
third-party code; policy-compliant):

  1. For each condition, generate several text prompts.
  2. Encode -> stacked text-feature matrix  T^c ∈ ℝ^{n×d}.
  3. Map to the tangent plane at the mean μ via the Log map (spherical geometry).
  4. ONE SVD:  Log_μ(T^c) = U Σ Vᵀ.
  5. Keep the top-k right singular vectors V_k -> projection P = V_k V_kᵀ
     (k = min(50, n)).
  6. Score DB images by conditional similarity: cos(P·v_ref, P·v_db).

Why it is the baseline we beat — its documented limitations:
  * P1 no sign: span(V_k) is direction-agnostic. We STACK +attr and -attr prompts
    the same way (no polarity) on purpose — that is exactly CLAY's failure mode.
  * P2 no weighting, P3 no interaction, P4 static (text-only subspace).

Simplifications vs the paper (documented for the report):
  * Prompts are fixed templates, not LLM-generated. We use a small paraphrase set
    per condition so the SVD has rank > 1.
  * Modality-gap handling: instead of CLAY's learned mean-alignment rotation H,
    we project both v_ref and the DB features into the SAME text-tangent subspace
    and take cosine. Mean-centering at μ approximates the alignment.

Query-side only. The frozen DB features are never re-encoded through CLIP; the
projection is plain linear algebra at query time.
"""
import torch

from src.common.features import encode_data
from src.common.geometry import log_map, sphere_mean

# Paraphrase templates: {} is filled with the readable attribute name.
# Multiple prompts per condition -> the stacked matrix has rank > 1 so the SVD
# captures a meaningful subspace (CLAY relies on LLM-generated prompt variety).
PROMPT_TEMPLATES = (
    "a photo of a person with {}",
    "a portrait of someone with {}",
    "a close-up photo showing {}",
    "an image of a face with {}",
    "a picture of a person who has {}",
    "a headshot of a person with {}",
)


def _readable(name):
    return name.replace('_', ' ').lower()


def build_condition_prompts(names):
    """Flatten {name: [prompts]} into a single prompt list + index slices.

    Returns (prompts, spans) where spans[i] = (start, end) rows of condition i.
    Sign is intentionally dropped — pos and neg conditions are treated alike.
    """
    prompts, spans = [], []
    for n in names:
        start = len(prompts)
        prompts.extend(t.format(_readable(n)) for t in PROMPT_TEMPLATES)
        spans.append((start, len(prompts)))
    return prompts, spans


def clay_subspace(pos_names, neg_names, k=50):
    """Build the CLAY projection matrix P = V_k V_kᵀ and the tangent mean μ.

    Stacks ALL condition prompts (positive and negative, no sign), encodes them,
    Log-maps at their mean, runs one SVD, keeps the top-k right singular vectors.
    Returns (P [d,d], mu [d]). k is capped at the number of prompts.
    """
    names = list(pos_names) + list(neg_names)
    prompts, _ = build_condition_prompts(names)
    _, z = encode_data(texts=prompts)               # [n, d], L2-normalized
    z = z.cpu().float()

    mu = sphere_mean(z)                              # tangency point
    tangent = torch.stack([log_map(mu, z[i]) for i in range(z.shape[0])])  # [n, d]

    # one SVD; right singular vectors V are the columns of Vh.T
    _, _, Vh = torch.linalg.svd(tangent, full_matrices=False)
    kk = min(k, Vh.shape[0])
    Vk = Vh[:kk].T                                  # [d, kk]
    P = Vk @ Vk.T                                   # [d, d]
    return P, mu


def project_db(db, P, mu):
    """Project + L2-normalize the whole DB into the conditional subspace.

    Depends only on (P, μ) — i.e. on the query, NOT on the reference. Compute it
    ONCE per query and reuse across all that query's sources (the full-DB Log map
    is the expensive step). Returns the normalized projected DB [N, d].
    """
    db = db.float()
    mu = mu.float()
    # vectorized log-map for the DB: tangent_i = (x_i - μ cosθ_i) · θ/sinθ
    cos_t = (db @ mu).clamp(-1 + 1e-7, 1 - 1e-7)    # [N]
    theta = torch.acos(cos_t)                       # [N]
    x_perp = db - mu.unsqueeze(0) * cos_t.unsqueeze(1)
    scale = (theta / theta.sin().clamp_min(1e-12)).unsqueeze(1)
    tangent = x_perp * scale                        # [N, d]
    proj = tangent @ P.T                            # [N, d]
    return proj / proj.norm(dim=1, keepdim=True).clamp_min(1e-12)


def clay_scores(v_ref, db_proj, P, mu):
    """Conditional similarity of every DB row vs v_ref.

    `db_proj` is the precomputed normalized projected DB from project_db().
    Projects the reference into the same subspace and takes cosine. Returns [N].
    """
    r = log_map(mu.float(), v_ref.float())          # [d]
    rp = P @ r
    rp = rp / rp.norm().clamp_min(1e-12)
    return db_proj @ rp                             # [N]


def clay_rank(v_ref, db_proj, P, mu, *, exclude=None, k=10):
    """Top-k DB indices by CLAY conditional similarity, source excluded.

    `db_proj` from project_db() (computed once per query).
    """
    scores = clay_scores(v_ref, db_proj, P, mu)
    exclude = set(exclude) if exclude else set()
    topn = min(len(scores), k + len(exclude))
    idx = torch.topk(scores, topn).indices.tolist()
    return [i for i in idx if i not in exclude][:k]
