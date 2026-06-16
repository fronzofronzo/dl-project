"""Fase 3 runner: naive ambient-arithmetic baseline over all eval queries.

Builds per-source rankings against the frozen visual DB, evaluates Recall@K /
Precision@K, runs sanity checks, and freezes the numbers (lower bound) to
results/baseline_naive.{json,md}.

Run from repo root:  .venv/bin/python src/run_baseline.py
"""
import json
from pathlib import Path

import torch

from baselines import naive_query, precompute_text
from groundtruth import build_ground_truth
from metrics import evaluate_all
from retrieval import load_db, rank

ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = ROOT / "data" / "celeba_evaluation.json"
RESULTS = ROOT / "results"
KS = (1, 5, 10)


def sanity_checks(gts, rankings_per_query, db):
    print("== sanity checks ==")
    # 1. identity probe: nearest neighbour of an untouched ref is a near-duplicate
    s0 = next(iter(gts[0].gt))
    v_ref = db[s0]
    nn = rank(v_ref, db, exclude={s0})[0]
    top_sim = (db[nn].float() @ (v_ref.float() / v_ref.norm())).item()
    assert top_sim > 0.7, f"identity probe weak: top cosine {top_sim:.3f}"
    print(f"  ok  identity probe: source {s0} -> nn {nn} cos={top_sim:.3f}")

    # 2. self never leaks into a ranking; n_sources matches JSON post-filter keys
    for qgt in gts:
        rpq = rankings_per_query[qgt.query]
        for s, ranked in rpq.items():
            assert s not in ranked, f"self {s} leaked into ranking for {qgt.query!r}"
    print("  ok  self excluded from every ranking")


def main():
    gts = build_ground_truth(EVAL_JSON)            # names mode (pos/neg are strings)
    names = {n for q in gts for n in (*q.pos, *q.neg)}
    text_cache = precompute_text(names)
    db = load_db(ROOT / "data" / "clip_features_test.pt")
    print(f"DB {tuple(db.shape)} | {len(gts)} queries | {len(names)} unique attrs")

    rankings_per_query = {}
    for qgt in gts:
        rpq = rankings_per_query.setdefault(qgt.query, {})
        for s in qgt.gt:
            v_t = naive_query(db[s], qgt.pos, qgt.neg, text_cache)
            rpq[s] = rank(v_t, db, exclude={s}, k=max(KS))
        # n_sources sanity: ranking built for every JSON post-filter source
        assert len(rpq) == len(qgt.gt), f"{qgt.query!r}: source count mismatch"

    rows = evaluate_all(rankings_per_query, gts, ks=KS)

    sanity_checks(gts, rankings_per_query, db)
    for k in KS:                                   # range sanity
        for q, r in rows.items():
            assert 0.0 <= r[f"recall@{k}"] <= 1.0 and 0.0 <= r[f"precision@{k}"] <= 1.0, q
    print("  ok  all metrics in [0,1]")

    write_results(rows)


def write_results(rows):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline_naive.json").write_text(json.dumps(rows, indent=2))

    cols = [f"{m}@{k}" for k in KS for m in ("recall", "precision")]
    header = "| query | " + " | ".join(cols) + " | n_sources |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = ["# Baseline — naive ambient arithmetic (lower bound)", "",
             header, sep]
    ordered = [q for q in rows if q != "MACRO"] + (["MACRO"] if "MACRO" in rows else [])
    for q in ordered:
        r = rows[q]
        vals = " | ".join(f"{r[c]:.3f}" for c in cols)
        lines.append(f"| {q} | {vals} | {r['n_sources']} |")
    (RESULTS / "baseline_naive.md").write_text("\n".join(lines) + "\n")
    print(f"frozen -> {RESULTS / 'baseline_naive.json'} , {RESULTS / 'baseline_naive.md'}")


if __name__ == "__main__":
    main()
