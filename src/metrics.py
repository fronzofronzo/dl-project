
def _topk(ranked_ids, k, exclude=None):
    """Top-k ids, optionally dropping `exclude` ids first."""
    if exclude:
        exclude = set(exclude) if not isinstance(exclude, set) else exclude
        ranked_ids = [i for i in ranked_ids if i not in exclude]
    return ranked_ids[:k]


def recall_at_k(ranked_ids, gt_set, k, exclude=None):
    """1.0 if any ground-truth id is in the top-k, else 0.0 (hit-rate)."""
    if not gt_set:
        return 0.0
    top = _topk(ranked_ids, k, exclude)
    return 1.0 if (set(top) & gt_set) else 0.0


def precision_at_k(ranked_ids, gt_set, k, exclude=None):
    """|top-k cap gt| / k."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top = _topk(ranked_ids, k, exclude)
    return len(set(top) & gt_set) / k


def evaluate_query(rankings, query_gt, ks=(1, 5, 10)):
    """Average Recall@K / Precision@K over the valid sources of one query.

    rankings: dict[int, list[int]] mapping source (reference) index -> ranked DB indices
              (self already excluded). Sources missing from `rankings` are skipped.
    query_gt: a QueryGT (uses its .gt dict: ref_idx -> set of target indices).

    Returns {"recall@k": float, ..., "precision@k": float, ..., "n_sources": int}.
    """
    sources = [s for s in query_gt.gt if s in rankings]
    out = {}
    for k in ks:
        if sources:
            out[f"recall@{k}"] = sum(
                recall_at_k(rankings[s], query_gt.gt[s], k) for s in sources
            ) / len(sources)
            out[f"precision@{k}"] = sum(
                precision_at_k(rankings[s], query_gt.gt[s], k) for s in sources
            ) / len(sources)
        else:
            out[f"recall@{k}"] = 0.0
            out[f"precision@{k}"] = 0.0
    out["n_sources"] = len(sources)
    return out


def evaluate_all(rankings_per_query, gts, ks=(1, 5, 10)):
    """Evaluate every query, plus a macro-average row across queries.

    rankings_per_query: dict[str, dict[int, list[int]]]  query string -> per-source rankings.
    gts: list[QueryGT].

    Returns dict[str, dict]: one row per query (keyed by query string) and a "MACRO" row
    averaging each metric over the evaluated queries (n_sources-unweighted).
    """
    rows = {}
    for qgt in gts:
        if qgt.query in rankings_per_query:
            rows[qgt.query] = evaluate_query(rankings_per_query[qgt.query], qgt, ks)

    if rows:
        macro = {}
        metric_keys = [f"{m}@{k}" for k in ks for m in ("recall", "precision")]
        for key in metric_keys:
            macro[key] = sum(r[key] for r in rows.values()) / len(rows)
        macro["n_sources"] = sum(r["n_sources"] for r in rows.values())
        rows["MACRO"] = macro
    return rows
