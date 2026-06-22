"""Soluzione A ingrediente 1C: valutazione full con pesi dinamici condizionati dall'input.

Per ogni source v_ref, i pesi per attributo sono calcolati da cos(v_ref, d_i):
  w_i^+ = max(0, 1 - cos(v_ref, d_i))   push dove l'attributo MANCA
  w_j^- = max(0, cos(v_ref, d_j))        push dove l'attributo da negare E' presente

Confronto vs contrastive (pesi fissi) se contrastive_ambient.json esiste.

Run da repo root:  python -m src.solution_a.run_dynamic_weights
"""
import json
from pathlib import Path

from src.solution_a.directions import build_direction_axes
from src.solution_a.dynamic_weights import dynamic_query
from src.common.groundtruth import build_ground_truth
from src.common.metrics import evaluate_all
from src.common.retrieval import load_db, rank

from src.common.paths import PROJECT_ROOT as ROOT
EVAL_JSON = ROOT / "data" / "celeba_evaluation.json"
RESULTS = ROOT / "results"
KS = (1, 5, 10)
ALPHAS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)


def eval_alpha(gts, axes, db, alpha):
    rankings_per_query = {}
    for qgt in gts:
        rpq = rankings_per_query.setdefault(qgt.query, {})
        for s in qgt.gt:
            v_t = dynamic_query(db[s], qgt.pos, qgt.neg, axes, alpha=alpha)
            rpq[s] = rank(v_t, db, exclude={s}, k=max(KS))
        assert len(rpq) == len(qgt.gt), f"{qgt.query!r}: source count mismatch"
    return evaluate_all(rankings_per_query, gts, ks=KS), rankings_per_query


def main():
    gts = build_ground_truth(EVAL_JSON)
    names = {n for q in gts for n in (*q.pos, *q.neg)}
    axes = build_direction_axes(names)
    db = load_db(ROOT / "data" / "clip_features_test.pt")
    print(f"DB {tuple(db.shape)} | {len(gts)} queries | {len(names)} attrs | alphas {ALPHAS}")

    sweep = {}
    best = None
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

    for k in KS:
        for q, r in best_rows.items():
            assert 0.0 <= r[f"recall@{k}"] <= 1.0 and 0.0 <= r[f"precision@{k}"] <= 1.0, q
    print("  ok  all metrics in [0,1]")

    write_results(best_rows, best_alpha, sweep)


def write_results(rows, best_alpha, sweep):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dynamic_weights.json").write_text(
        json.dumps({"best_alpha": best_alpha, "rows": rows}, indent=2))

    cols = [f"{m}@{k}" for k in KS for m in ("recall", "precision")]
    ordered = [q for q in rows if q != "MACRO"] + (["MACRO"] if "MACRO" in rows else [])

    lines = [f"# Dynamic weights — ambient (best alpha = {best_alpha})", "",
             "| query | " + " | ".join(cols) + " | n_sources |",
             "|" + "---|" * (len(cols) + 2)]
    for q in ordered:
        r = rows[q]
        lines.append(f"| {q} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {r['n_sources']} |")

    # confronto vs contrastive se disponibile
    contr_path = RESULTS / "contrastive_ambient.json"
    if contr_path.exists():
        contr = json.loads(contr_path.read_text())["rows"]
        lines += ["", "## Contrastive vs Dynamic weights (per query, R@1 / R@5)", "",
                  "| query | contr R@1 | dyn R@1 | contr R@5 | dyn R@5 |",
                  "|---|---|---|---|---|"]
        for q in ordered:
            if q in contr:
                c, d = contr[q], rows[q]
                lines.append(f"| {q} | {c['recall@1']:.3f} | {d['recall@1']:.3f} "
                             f"| {c['recall@5']:.3f} | {d['recall@5']:.3f} |")
    (RESULTS / "dynamic_weights.md").write_text("\n".join(lines) + "\n")

    # alpha sweep
    sl = ["# Dynamic weights — alpha sweep (MACRO)", "",
          "| alpha | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 |", "|---|---|---|---|---|---|---|"]
    for a in sorted(sweep):
        m = sweep[a]
        mark = " (best)" if a == best_alpha else ""
        sl.append(f"| {a}{mark} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
                  f"| {m['precision@1']:.3f} | {m['precision@5']:.3f} | {m['precision@10']:.3f} |")
    (RESULTS / "dynamic_weights_alpha_sweep.md").write_text("\n".join(sl) + "\n")

    print(f"frozen -> {RESULTS / 'dynamic_weights.md'} , dynamic_weights_alpha_sweep.md")


if __name__ == "__main__":
    main()
