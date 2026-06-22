"""CLAY stacked-SVD baseline runner over all eval queries.

The real SOTA to beat. For each query, builds the conditional subspace from the
condition prompts (one SVD), then ranks every valid source against the frozen DB
by conditional similarity. Sweeps k (top singular vectors). Freezes the numbers
to results/baseline_clay.{json,md}.

Run from repo root:  python -m src.baselines.run_clay
"""
import json

from src.common.paths import PROJECT_ROOT as ROOT, EVAL_JSON
from src.baselines.clay import clay_subspace, clay_rank, project_db
from src.common.groundtruth import build_ground_truth
from src.common.metrics import evaluate_all
from src.common.retrieval import load_db

RESULTS = ROOT / "results"
KS = (1, 5, 10)
KSVD = (10, 30, 50)          # top-k right singular vectors to keep


def eval_ksvd(gts, db, k_svd):
    rankings_per_query = {}
    for qgt in gts:
        # subspace + DB projection built once per query (shared by all its sources)
        P, mu = clay_subspace(qgt.pos, qgt.neg, k=k_svd)
        db_proj = project_db(db, P, mu)
        rpq = rankings_per_query.setdefault(qgt.query, {})
        for s in qgt.gt:
            rpq[s] = clay_rank(db[s], db_proj, P, mu, exclude={s}, k=max(KS))
        assert len(rpq) == len(qgt.gt), f"{qgt.query!r}: source count mismatch"
    return evaluate_all(rankings_per_query, gts, ks=KS)


def main():
    gts = build_ground_truth(EVAL_JSON)
    db = load_db().float()
    names = sorted({n for q in gts for n in (*q.pos, *q.neg)})
    print(f"DB {tuple(db.shape)} | {len(gts)} queries | {len(names)} attrs | k_svd {KSVD}")

    sweep, best = {}, None
    for k_svd in KSVD:
        rows = eval_ksvd(gts, db, k_svd)
        m = rows["MACRO"]
        sweep[k_svd] = m
        print(f"  k_svd={k_svd:<3}  R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  R@10={m['recall@10']:.3f}")
        key = (m["recall@1"], m["recall@5"])
        if best is None or key > best[2]:
            best = (k_svd, rows, key)
    best_k, best_rows = best[0], best[1]
    print(f"best k_svd = {best_k} (by R@1, tie R@5)")

    for k in KS:
        for q, r in best_rows.items():
            assert 0.0 <= r[f"recall@{k}"] <= 1.0 and 0.0 <= r[f"precision@{k}"] <= 1.0, q
    print("  ok  all metrics in [0,1]")

    write_results(best_rows, best_k, sweep)


def write_results(rows, best_k, sweep):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline_clay.json").write_text(
        json.dumps({"best_k_svd": best_k, "rows": rows}, indent=2))

    cols = [f"{m}@{k}" for k in KS for m in ("recall", "precision")]
    ordered = [q for q in rows if q != "MACRO"] + (["MACRO"] if "MACRO" in rows else [])
    lines = [f"# Baseline — CLAY stacked-SVD (best k_svd = {best_k})", "",
             "| query | " + " | ".join(cols) + " | n_sources |",
             "|" + "---|" * (len(cols) + 2)]
    for q in ordered:
        r = rows[q]
        lines.append(f"| {q} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {r['n_sources']} |")

    # vs naive baseline if frozen
    naive_path = RESULTS / "baseline_naive.json"
    if naive_path.exists():
        naive = json.loads(naive_path.read_text())
        lines += ["", "## Naive vs CLAY (per query, R@1 / R@5)", "",
                  "| query | naive R@1 | clay R@1 | naive R@5 | clay R@5 |",
                  "|---|---|---|---|---|"]
        for q in ordered:
            if q in naive:
                n, c = naive[q], rows[q]
                lines.append(f"| {q} | {n['recall@1']:.3f} | {c['recall@1']:.3f} "
                             f"| {n['recall@5']:.3f} | {c['recall@5']:.3f} |")
    (RESULTS / "baseline_clay.md").write_text("\n".join(lines) + "\n")

    sl = ["# CLAY — k_svd sweep (MACRO)", "",
          "| k_svd | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 |", "|---|---|---|---|---|---|---|"]
    for k_svd in sorted(sweep):
        m = sweep[k_svd]
        mark = " (best)" if k_svd == best_k else ""
        sl.append(f"| {k_svd}{mark} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
                  f"| {m['precision@1']:.3f} | {m['precision@5']:.3f} | {m['precision@10']:.3f} |")
    (RESULTS / "baseline_clay_ksvd_sweep.md").write_text("\n".join(sl) + "\n")

    print(f"frozen -> {RESULTS / 'baseline_clay.md'} , baseline_clay_ksvd_sweep.md")


if __name__ == "__main__":
    main()
