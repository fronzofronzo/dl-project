"""Negative re-ranking for Solution A.

The contrastive edit pushes the query *away* from negated attributes, but it
acts in query space and is imprecise: the cosine search still returns candidates
that *still contain* the negated attribute (CLAY's P1 / sign failure). This is a
second, explicit candidate-side check applied *after* retrieval.

Two stages:
  1. retrieve a candidate pool by cosine vs the edited query (cheap, query-side);
  2. demote candidates whose presence score on a negated attribute is high.

Presence probe = image-space attribute axis
    d_img = mean(img WITH attr) - mean(img WITHOUT attr)
built from labeled features. Image-image cosine sidesteps the text->image
modality gap (cos(image, text-direction) ~ 0 -> useless as a presence probe,
verified empirically in src/diag.py).

NOTE on leakage: probes here are built from the SAME (test) DB we retrieve from
— fine for prototyping. For the final report build them from the TRAIN split so
no test labels are used. DB is never re-encoded; this stays query-side.
"""
import torch
import torch.nn.functional as F


def build_image_probes(db, attr, attr_index, names):
    """{name: unit image-space direction d_img} from labeled features.

    db: [N,512]. attr: [N,40] in {0,1}. attr_index: {name: column}.
    """
    db = F.normalize(db.float(), dim=-1)
    probes = {}
    for n in names:
        col = attr[:, attr_index[n]]
        pos = db[col == 1].mean(0)
        neg = db[col == 0].mean(0)
        probes[n] = F.normalize(pos - neg, dim=0)
    return probes


def negative_rerank(v_target, db, neg_names, probes, *, exclude=None,
                    k=10, pool=200, lam=1.0):
    """Cosine pool -> demote negated-attribute violators -> top-k.

        score_final = cos(cand, v_target) - lam * Σ_neg cos(cand, d_neg)

    lam=0 reproduces plain cosine retrieval (no re-rank). With no negated
    attributes the pool stage already equals plain retrieval. Returns DB
    indices, most relevant first, source excluded.
    """
    db = db.float()
    v = F.normalize(v_target.float(), dim=0)
    base = db @ v                                   # [N]
    exclude = set(exclude) if exclude else set()

    # no negatives (or no penalty) -> plain top-k
    if not neg_names or lam == 0.0:
        topn = min(len(base), k + len(exclude))
        order = torch.topk(base, topn).indices.tolist()
        return [i for i in order if i not in exclude][:k]

    # stage 1: candidate pool by base similarity
    pooln = min(len(base), pool + len(exclude))
    cand = torch.topk(base, pooln).indices.tolist()
    cand = torch.tensor([i for i in cand if i not in exclude])

    # stage 2: presence penalty per negated attribute (image-space probe)
    penalty = torch.zeros(len(cand))
    for n in neg_names:
        penalty = penalty + db[cand] @ probes[n]
    final = base[cand] - lam * penalty

    order = final.argsort(descending=True)
    return cand[order][:k].tolist()
