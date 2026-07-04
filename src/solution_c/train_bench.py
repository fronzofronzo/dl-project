"""Solution C — train-split pseudo-benchmark for tuning inference dials.

The 13 evaluation query STRINGS are assignment-given (a-priori knowledge), but
their ground truth lives on the test split — any dial tuned against it leaks
test information (the existing N/λ/T sweep already has this caveat; do not add
more). This module rebuilds the SAME queries on the CelebA TRAIN split, using
only the frozen cached features/labels already on disk, with the benchmark's
own validity rules verified against the eval JSON:

  source  = violates ALL queried constraints (FLIP-REF: measured 1.000 on every
            test-JSON query), with >= min_targets valid targets
  target  = strictly satisfies ALL constraints AND Hamming <= 2 on the
            remaining attributes (identity band)

New dials (PoE η, reliability weighting, the |C| gate) are tuned HERE and then
applied ONCE to the test benchmark. Returns standard QueryGT objects, so the
shared eval harness (src/solution_b/run.eval_phi, src/solution_c/rerank) runs
on it verbatim with db = frozen TRAIN features.

NOTE: the eval JSON has 14 entries but 13 unique queries (`-Young` twice);
queries are deduped by string here so macros stay comparable.

Run from repo root:  python -m src.solution_c.train_bench          (build + stats)
                     python -m src.solution_c.train_bench --smoke  (synthetic checks)
"""
import argparse

import torch

from src.common.paths import ATTRS_TRAIN, EVAL_JSON
from src.common.groundtruth import QueryGT, build_ground_truth


def unique_queries(gts):
    """Dedupe QueryGTs by query string, keeping first occurrence order."""
    seen, out = set(), []
    for q in gts:
        if q.query not in seen:
            seen.add(q.query)
            out.append(q)
    return out


def build_gt_for_query(L, cols, req, n_sources=200, min_targets=5, max_ham=2,
                       generator=None):
    """Ground truth of one query on the label matrix L [N, 40] (bool).

    cols: LongTensor [C] queried attribute columns; req: BoolTensor [C] required
    state (True = present). Returns {source_idx: set(target_idx)} for up to
    n_sources seeded-random sources among the violate-all candidates.
    """
    N, n_attr = L.shape
    qmask = torch.zeros(n_attr, dtype=torch.bool, device=L.device)
    qmask[cols] = True
    rest = ~qmask

    sat = (L[:, cols] == req).all(1)                      # strict satisfy [N]
    src_cand = (L[:, cols] != req).all(1).nonzero(as_tuple=False).squeeze(1)
    if src_cand.numel() == 0:
        return {}

    # Hamming only against the images that already satisfy the constraints
    tgt_idx = sat.nonzero(as_tuple=False).squeeze(1)      # [T]
    Lt = L[tgt_idx][:, rest]                              # [T, 40-C]

    order = torch.randperm(src_cand.numel(), generator=generator)
    gt = {}
    for j in order.tolist():
        s = int(src_cand[j])
        ham = (Lt != L[s, rest]).sum(1)                   # [T]
        targets = tgt_idx[ham <= max_ham]
        if targets.numel() >= min_targets:
            gt[s] = set(targets.tolist())
            if len(gt) >= n_sources:
                break
    return gt


def build_train_gts(n_sources=200, min_targets=5, max_ham=2, seed=0,
                    device="cpu", attrs_path=ATTRS_TRAIN, eval_json=EVAL_JSON,
                    attr_index=None):
    """Mirror the eval queries on the train split -> list[QueryGT].

    attr_index: {attr_name: column}; defaults to the CelebA name->column map
    (split-independent), read via src.solution_b.run.attr_index_test.
    """
    if attr_index is None:
        from src.solution_b.run import attr_index_test
        attr_index = attr_index_test()

    L = torch.load(attrs_path).bool().to(device)
    gen = torch.Generator().manual_seed(seed)

    out = []
    for q in unique_queries(build_ground_truth(eval_json, min_targets=min_targets)):
        cols = torch.tensor([attr_index[n] for n in q.pos + q.neg],
                            dtype=torch.long, device=device)
        req = torch.tensor([True] * len(q.pos) + [False] * len(q.neg),
                           device=device)
        gt = build_gt_for_query(L, cols, req, n_sources=n_sources,
                                min_targets=min_targets, max_ham=max_ham,
                                generator=gen)
        out.append(QueryGT(query=q.query, pos=list(q.pos), neg=list(q.neg), gt=gt))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sources", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gts = build_train_gts(n_sources=args.n_sources, seed=args.seed, device=device)
    print(f"{len(gts)} unique queries (train split, n_sources<={args.n_sources})")
    for q in gts:
        n_tgt = [len(t) for t in q.gt.values()]
        mean_t = sum(n_tgt) / max(len(n_tgt), 1)
        print(f"  {q.query:50s} sources={len(q.gt):4d}  mean_targets={mean_t:8.1f}")
    thin = [q.query for q in gts if len(q.gt) < 30]
    if thin:
        print(f"WARNING: <30 sources for: {thin}")


# --------------------------------------------------------------------------- #
# Smoke: synthetic labels, no files needed
# --------------------------------------------------------------------------- #
def _smoke():
    torch.manual_seed(0)
    # clustered labels (i.i.d. p=0.5 would leave every Hamming<=2 band empty):
    # 20 prototypes with ~1 bit flipped per sample define the "identity" rest;
    # the queried columns vary independently of the cluster (as in CelebA).
    N, n_attr, n_proto = 3000, 40, 20
    protos = torch.rand(n_proto, n_attr) > 0.5
    L = protos[torch.randint(n_proto, (N,))].clone()
    flip = torch.randint(n_attr, (N,))
    keep = torch.rand(N) > 0.5
    rows = torch.arange(N)[~keep]
    L[rows, flip[~keep]] = ~L[rows, flip[~keep]]

    cols = torch.tensor([3, 17])
    req = torch.tensor([True, False])
    L[:, cols] = torch.rand(N, 2) > 0.5
    gen = torch.Generator().manual_seed(0)
    gt = build_gt_for_query(L, cols, req, n_sources=50, min_targets=5,
                            max_ham=2, generator=gen)
    assert len(gt) >= 20, f"expected >=20 sources, got {len(gt)}"

    qmask = torch.zeros(n_attr, dtype=torch.bool)
    qmask[cols] = True
    for s, targets in gt.items():
        assert (L[s, cols] != req).all(), "source does not violate all constraints"
        assert len(targets) >= 5 and s not in targets
        for t in targets:
            assert (L[t, cols] == req).all(), "target violates a constraint"
            assert (L[t, ~qmask] != L[s, ~qmask]).sum() <= 2, "target outside Hamming band"
    print("  ok  sources violate-all, targets satisfy-all + Hamming<=2, counts")

    # completeness: every valid target of a kept source is in its set
    s = next(iter(gt))
    sat = (L[:, cols] == req).all(1)
    ham = (L[:, ~qmask] != L[s, ~qmask]).sum(1) <= 2
    full = set((sat & ham).nonzero(as_tuple=False).squeeze(1).tolist()) - {s}
    assert full == gt[s], "target set incomplete"
    print("  ok  target sets are exhaustive")

    # determinism + impossible query -> empty
    gt2 = build_gt_for_query(L, cols, req, n_sources=50, min_targets=5, max_ham=2,
                             generator=torch.Generator().manual_seed(0))
    assert gt.keys() == gt2.keys() and all(gt[k] == gt2[k] for k in gt)
    L_all_true = torch.ones(N, n_attr, dtype=torch.bool)
    assert build_gt_for_query(L_all_true, cols, req, generator=gen) == {}
    print("  ok  seeded determinism, impossible query -> empty")

    # dedupe helper
    a, b = QueryGT("q1"), QueryGT("q2")
    assert [x.query for x in unique_queries([a, b, QueryGT("q1")])] == ["q1", "q2"]
    print("smoke test passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--smoke", action="store_true")
    known, _ = ap.parse_known_args()
    if known.smoke:
        _smoke()
    else:
        main()
