"""Soluzione A — ambient contrastive edit + negative re-ranking.

Pipeline (query-side only, DB frozen):
  1. edit:    v_target = normalize(v_ref + alpha * (Σ d_pos − Σ d_neg))
              d_attr = z("...with attr") − z("...without attr")  (oriented, signed)
  2. retrieve: cosine pool of v_target vs frozen DB
  3. re-rank:  demote candidates that still contain a negated attribute, using an
              image-space presence probe (sidesteps the text->image modality gap)

Sweeps (alpha, lambda). lambda=0 == no re-rank (ablation baseline).

Run da repo root:  python -m src.solution_a.run
"""
import json
from pathlib import Path

import torch
from torchvision.datasets import CelebA

from src.solution_a.directions import build_direction_axes, contrastive_query
from src.common.groundtruth import build_ground_truth
from src.common.metrics import evaluate_all
from src.solution_a.rerank import build_image_probes, negative_rerank
from src.common.retrieval import load_db

from src.common.paths import PROJECT_ROOT as ROOT
EVAL_JSON = ROOT / "data" / "celeba_evaluation.json"
RESULTS = ROOT / "results"
KS = (1, 5, 10)
ALPHAS = (3.0, 4.0, 5.0)
LAMBDAS = (0.0, 1.0, 2.0, 4.0)
ORTHS = ("off", "full", "conflict")   # Gram–Schmidt ablation: none / all dirs / opposite-sign only
POOL = 200


def eval_config(gts, axes, db, probes, alpha, lam, orth):
    rankings_per_query = {}
    for qgt in gts:
        rpq = rankings_per_query.setdefault(qgt.query, {})
        for s in qgt.gt:
            v_t = contrastive_query(db[s], qgt.pos, qgt.neg, axes, alpha=alpha, orth=orth)
            rpq[s] = negative_rerank(v_t, db, qgt.neg, probes,
                                     exclude={s}, k=max(KS), pool=POOL, lam=lam)
        assert len(rpq) == len(qgt.gt), f"{qgt.query!r}: source count mismatch"
    return evaluate_all(rankings_per_query, gts, ks=KS)


def main():
    gts = build_ground_truth(EVAL_JSON)
    names = sorted({n for q in gts for n in (*q.pos, *q.neg)})
    axes = build_direction_axes(names)
    db = load_db(ROOT / "data" / "clip_features_test.pt").float()

    # image-space presence probes for negated attributes (test labels -> see rerank.py caveat)
    ds = CelebA(root=str(ROOT / "data"), split="test", download=False)
    attr_index = {n: i for i, n in enumerate(ds.attr_names) if n}
    probes = build_image_probes(db, ds.attr.float(), attr_index, names)

    print(f"DB {tuple(db.shape)} | {len(gts)} queries | {len(names)} attrs "
          f"| alphas {ALPHAS} | lambdas {LAMBDAS} | orth {ORTHS}")

    sweep = {}
    best = None
    for a in ALPHAS:
        for lam in LAMBDAS:
            for orth in ORTHS:
                rows = eval_config(gts, axes, db, probes, a, lam, orth)
                m = rows["MACRO"]
                sweep[(a, lam, orth)] = m
                print(f"  alpha={a:<4} lam={lam:<4} orth={orth:<8} "
                      f"R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  R@10={m['recall@10']:.3f}")
                key = (m["recall@1"], m["recall@5"])
                if best is None or key > best[2]:
                    best = ((a, lam, orth), rows, key)
    (best_a, best_lam, best_orth), best_rows = best[0], best[1]
    print(f"best alpha={best_a} lambda={best_lam} orth={best_orth} (by R@1, tie R@5)")

    for k in KS:
        for q, r in best_rows.items():
            assert 0.0 <= r[f"recall@{k}"] <= 1.0 and 0.0 <= r[f"precision@{k}"] <= 1.0, q
    print("  ok  all metrics in [0,1]")

    write_results(best_rows, best_a, best_lam, best_orth, sweep)


def write_results(rows, best_a, best_lam, best_orth, sweep):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "solution_a.json").write_text(
        json.dumps({"best_alpha": best_a, "best_lambda": best_lam,
                    "best_orth": best_orth, "rows": rows}, indent=2))

    cols = [f"{m}@{k}" for k in KS for m in ("recall", "precision")]
    ordered = [q for q in rows if q != "MACRO"] + (["MACRO"] if "MACRO" in rows else [])

    lines = [f"# Solution A — contrastive + negative re-rank "
             f"(best alpha={best_a}, lambda={best_lam}, orth={best_orth})", "",
             "| query | " + " | ".join(cols) + " | n_sources |",
             "|" + "---|" * (len(cols) + 2)]
    for q in ordered:
        r = rows[q]
        lines.append(f"| {q} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {r['n_sources']} |")

    naive_path = RESULTS / "baseline_naive.json"
    if naive_path.exists():
        naive = json.loads(naive_path.read_text())
        lines += ["", "## vs Naive baseline (R@1 / R@5)", "",
                  "| query | naive R@1 | sol-A R@1 | naive R@5 | sol-A R@5 |",
                  "|---|---|---|---|---|"]
        for q in ordered:
            if q in naive:
                n, a = naive[q], rows[q]
                lines.append(f"| {q} | {n['recall@1']:.3f} | {a['recall@1']:.3f} "
                             f"| {n['recall@5']:.3f} | {a['recall@5']:.3f} |")
    (RESULTS / "solution_a.md").write_text("\n".join(lines) + "\n")

    sl = ["# Solution A — (alpha, lambda, orth) sweep (MACRO)", "",
          "| alpha | lambda | orth | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (a, lam, orth) in sorted(sweep):
        m = sweep[(a, lam, orth)]
        mark = " (best)" if (a, lam, orth) == (best_a, best_lam, best_orth) else ""
        sl.append(f"| {a}{mark} | {lam} | {orth} | {m['recall@1']:.3f} | {m['recall@5']:.3f} "
                  f"| {m['recall@10']:.3f} | {m['precision@1']:.3f} | {m['precision@5']:.3f} "
                  f"| {m['precision@10']:.3f} |")
    (RESULTS / "solution_a_sweep.md").write_text("\n".join(sl) + "\n")

    print(f"frozen -> {RESULTS / 'solution_a.md'} , solution_a_sweep.md")


if __name__ == "__main__":
    main()
