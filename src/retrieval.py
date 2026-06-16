import torch

DEFAULT_DB = "data/clip_features_test.pt"


def load_db(path=DEFAULT_DB):
    """Load the frozen visual DB [N,512]. Extracted offline, never re-encoded."""
    return torch.load(path)


def _l2norm(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def rank(v_query, db, exclude=None, k=None):
    """Cosine ranking of DB rows against a single query vector.

    db and v_query are expected L2-normalized; normalized defensively so the
    matmul is a true cosine similarity. `exclude` (e.g. the source index) is
    dropped from the result. With `k` set, only the top-k indices are returned
    (via torch.topk — avoids a full argsort and keeps memory bounded).
    Returns DB indices, most similar first.
    """
    v = _l2norm(v_query.float())
    sims = db.float() @ v                       # [N]
    exclude = (exclude if isinstance(exclude, set) else set(exclude)) if exclude else set()

    if k is not None:
        topn = min(len(sims), k + len(exclude))
        idx = torch.topk(sims, topn).indices.tolist()
        idx = [i for i in idx if i not in exclude]
        return idx[:k]

    order = torch.argsort(sims, descending=True).tolist()
    return [i for i in order if i not in exclude]
