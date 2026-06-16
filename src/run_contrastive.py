"""Fase 4 (ingrediente 1): vettori-direzione contrastivi in spazio ambiente.

Per ogni attributo costruisce un asse orientato d = z_with - z_without, poi
v_target = v_ref + alpha*(Σ d_pos − Σ d_neg). Sweep di alpha, freeze del best e
confronto diretto con la baseline naive.

Run da repo root:  .venv/bin/python src/run_contrastive.py
"""
import json
from pathlib import Path

from baselines import build_direction_axes, contrastive_query
from groundtruth import build_ground_truth
from metrics import evaluate_all
from retrieval import load_db, rank

ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = ROOT / "data" / "celeba_evaluation.json"
RESULTS = ROOT / "results"
KS = (1, 5, 10)
ALPHAS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def eval_alpha(gts, axes, db, alpha):
    rankings_per_query = {}
    for qgt in gts:
        rpq = rankings_per_query.setdefault(qgt.query, {})
        for s in qgt.gt:
            v_t = contrastive_query(db[s], qgt.pos, qgt.neg, axes, alpha=alpha)
            rpq[s] = rank(v_t, db, exclude={s}, k=max(KS))
        assert len(rpq) == len(qgt.gt), f"{qgt.query!r}: source count mismatch"
    return evaluate_all(rankings_per_query, gts, ks=KS), rankings_per_query


def main():
    gts = build_ground_truth(EVAL_JSON)
    names = {n for q in gts for n in (*q.pos, *q.neg)}
    axes = build_direction_axes(names)
    db = load_db(ROOT / "data" / "clip_features_test.pt")
    print(f"DB {tuple(db.shape)} | {len(gts)} queries | {len(names)} attrs | alphas {ALPHAS}")

    sweep = {}                          # alpha -> MACRO row
    best = None                         # (alpha, rows)
    for a in ALPHAS:
        rows, _ = eval_alpha(gts, axes, db, a)
        macro = rows["MACRO"]
        sweep[a] = macro
        print(f"  alpha={a:<4}  R@1={macro['recall@1']:.3f}  R@5={macro['recall@5']:.3f}  R@10={macro['recall@10']:.3f}")
        key = (macro["recall@1"], macro["recall@5"])
        if best is None or key > best[2]:
            best = (a, rows, key)
    best_alpha, best_rows = best[0], best[1]
    print(f"best alpha = {best_alpha} (by R@1, tie R@5)")

    # range sanity
    for k in KS:
        for q, r in best_rows.items():
            assert 0.0 <= r[f"recall@{k}"] <= 1.0 and 0.0 <= r[f"precision@{k}"] <= 1.0, q
    print("  ok  all metrics in [0,1]")

    write_results(best_rows, best_alpha, sweep)


def write_results(rows, best_alpha, sweep):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "contrastive_ambient.json").write_text(
        json.dumps({"best_alpha": best_alpha, "rows": rows}, indent=2))

    cols = [f"{m}@{k}" for k in KS for m in ("recall", "precision")]
    ordered = [q for q in rows if q != "MACRO"] + (["MACRO"] if "MACRO" in rows else [])

    # tabella principale (best alpha)
    lines = [f"# Contrastive direction axes — ambient (best alpha = {best_alpha})", "",
             "| query | " + " | ".join(cols) + " | n_sources |",
             "|" + "---|" * (len(cols) + 2)]
    for q in ordered:
        r = rows[q]
        lines.append(f"| {q} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {r['n_sources']} |")

    # confronto MACRO naive vs contrastive (se la baseline esiste)
    naive_path = RESULTS / "baseline_naive.json"
    if naive_path.exists():
        naive = json.loads(naive_path.read_text())
        lines += ["", "## Naive vs Contrastive (per query, R@1 / R@5)", "",
                  "| query | naive R@1 | contr R@1 | naive R@5 | contr R@5 |",
                  "|---|---|---|---|---|"]
        for q in ordered:
            if q in naive:
                n, c = naive[q], rows[q]
                lines.append(f"| {q} | {n['recall@1']:.3f} | {c['recall@1']:.3f} "
                             f"| {n['recall@5']:.3f} | {c['recall@5']:.3f} |")
    (RESULTS / "contrastive_ambient.md").write_text("\n".join(lines) + "\n")

    # alpha sweep
    sl = ["# Contrastive — alpha sweep (MACRO)", "",
          "| alpha | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 |", "|---|---|---|---|---|---|---|"]
    for a in sorted(sweep):
        m = sweep[a]
        mark = " (best)" if a == best_alpha else ""
        sl.append(f"| {a}{mark} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
                  f"| {m['precision@1']:.3f} | {m['precision@5']:.3f} | {m['precision@10']:.3f} |")
    (RESULTS / "contrastive_alpha_sweep.md").write_text("\n".join(sl) + "\n")

    print(f"frozen -> {RESULTS / 'contrastive_ambient.md'} , contrastive_alpha_sweep.md")


if __name__ == "__main__":
    main()
"""Fase 4 (ingrediente 1): vettori-direzione contrastivi in spazio ambiente.

Per ogni attributo costruisce un asse orientato d = z_with - z_without, poi
v_target = v_ref + alpha*(Σ d_pos − Σ d_neg). Sweep di alpha, freeze del best e
confronto diretto con la baseline naive.

Run da repo root:  .venv/bin/python src/run_contrastive.py
"""
import json
from pathlib import Path

from baselines import build_direction_axes, contrastive_query
from groundtruth import build_ground_truth
from metrics import evaluate_all
from retrieval import load_db, rank

ROOT = Path(__file__).resolve().parent.parent
EVAL_JSON = ROOT / "data" / "celeba_evaluation.json"
RESULTS = ROOT / "results"
KS = (1, 5, 10)
ALPHAS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def eval_alpha(gts, axes, db, alpha):
    rankings_per_query = {}
    for qgt in gts:
        rpq = rankings_per_query.setdefault(qgt.query, {})
        for s in qgt.gt:
            v_t = contrastive_query(db[s], qgt.pos, qgt.neg, axes, alpha=alpha)
            rpq[s] = rank(v_t, db, exclude={s}, k=max(KS))
        assert len(rpq) == len(qgt.gt), f"{qgt.query!r}: source count mismatch"
    return evaluate_all(rankings_per_query, gts, ks=KS), rankings_per_query


def main():
    gts = build_ground_truth(EVAL_JSON)
    names = {n for q in gts for n in (*q.pos, *q.neg)}
    axes = build_direction_axes(names)
    db = load_db(ROOT / "data" / "clip_features_test.pt")
    print(f"DB {tuple(db.shape)} | {len(gts)} queries | {len(names)} attrs | alphas {ALPHAS}")

    sweep = {}                          # alpha -> MACRO row
    best = None                         # (alpha, rows)
    for a in ALPHAS:
        rows, _ = eval_alpha(gts, axes, db, a)
        macro = rows["MACRO"]
        sweep[a] = macro
        print(f"  alpha={a:<4}  R@1={macro['recall@1']:.3f}  R@5={macro['recall@5']:.3f}  R@10={macro['recall@10']:.3f}")
        key = (macro["recall@1"], macro["recall@5"])
        if best is None or key > best[2]:
            best = (a, rows, key)
    best_alpha, best_rows = best[0], best[1]
    print(f"best alpha = {best_alpha} (by R@1, tie R@5)")

    # range sanity
    for k in KS:
        for q, r in best_rows.items():
            assert 0.0 <= r[f"recall@{k}"] <= 1.0 and 0.0 <= r[f"precision@{k}"] <= 1.0, q
    print("  ok  all metrics in [0,1]")

    write_results(best_rows, best_alpha, sweep)


def write_results(rows, best_alpha, sweep):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "contrastive_ambient.json").write_text(
        json.dumps({"best_alpha": best_alpha, "rows": rows}, indent=2))

    cols = [f"{m}@{k}" for k in KS for m in ("recall", "precision")]
    ordered = [q for q in rows if q != "MACRO"] + (["MACRO"] if "MACRO" in rows else [])

    # tabella principale (best alpha)
    lines = [f"# Contrastive direction axes — ambient (best alpha = {best_alpha})", "",
             "| query | " + " | ".join(cols) + " | n_sources |",
             "|" + "---|" * (len(cols) + 2)]
    for q in ordered:
        r = rows[q]
        lines.append(f"| {q} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {r['n_sources']} |")

    # confronto MACRO naive vs contrastive (se la baseline esiste)
    naive_path = RESULTS / "baseline_naive.json"
    if naive_path.exists():
        naive = json.loads(naive_path.read_text())
        lines += ["", "## Naive vs Contrastive (per query, R@1 / R@5)", "",
                  "| query | naive R@1 | contr R@1 | naive R@5 | contr R@5 |",
                  "|---|---|---|---|---|"]
        for q in ordered:
            if q in naive:
                n, c = naive[q], rows[q]
                lines.append(f"| {q} | {n['recall@1']:.3f} | {c['recall@1']:.3f} "
                             f"| {n['recall@5']:.3f} | {c['recall@5']:.3f} |")
    (RESULTS / "contrastive_ambient.md").write_text("\n".join(lines) + "\n")

    # alpha sweep
    sl = ["# Contrastive — alpha sweep (MACRO)", "",
          "| alpha | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 |", "|---|---|---|---|---|---|---|"]
    for a in sorted(sweep):
        m = sweep[a]
        mark = " (best)" if a == best_alpha else ""
        sl.append(f"| {a}{mark} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
                  f"| {m['precision@1']:.3f} | {m['precision@5']:.3f} | {m['precision@10']:.3f} |")
    (RESULTS / "contrastive_alpha_sweep.md").write_text("\n".join(sl) + "\n")

    print(f"frozen -> {RESULTS / 'contrastive_ambient.md'} , contrastive_alpha_sweep.md")


if __name__ == "__main__":
    main()
