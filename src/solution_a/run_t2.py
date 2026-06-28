"""Evaluate the trained T2 adapter Φ on the CelebA test benchmark, vs the frozen DB.

T2's OWN eval harness — independent of solution_b's run.py (which person 1 uses for
T1). Reuses only the shared common/* utilities (metrics / groundtruth / retrieval),
never the colleague's launch scripts. Query-side only: Φ_T2 composes v_q from the
test reference + signed conditions, then cosine against the frozen test DB. Same
protocol as the baselines (restrict to the JSON source keys, Recall@K / Precision@K
hit-rate).

`eval_phi` is imported by train_t2.py for periodic validation; `main` re-evaluates a
saved checkpoint and freezes the results table.

Run from repo root:  python -m src.solution_a.run_t2
"""
import json

import torch
from torchvision.datasets import CelebA

from src.common.paths import PROJECT_ROOT as ROOT, EVAL_JSON, DB_TEST, RESULTS
from src.common.groundtruth import build_ground_truth
from src.common.metrics import evaluate_all
from src.common.retrieval import load_db, rank
from src.solution_a.t2_directions import T2Phi

KS = (1, 5, 10)


def attr_index_test():
    """{attr_name: column} from the CelebA test split (underscored names match the
    eval-JSON query tokens directly)."""
    ds = CelebA(root=str(ROOT / "data"), split="test", download=False)
    return {n: i for i, n in enumerate(ds.attr_names) if n}


def query_conditions(qgt, attr_index, device):
    """One query -> (cond_col [C], cond_sign [C]). +1 for positives, -1 for negatives."""
    cols = [attr_index[n] for n in qgt.pos] + [attr_index[n] for n in qgt.neg]
    signs = [1.0] * len(qgt.pos) + [-1.0] * len(qgt.neg)
    return (torch.tensor(cols, dtype=torch.long, device=device),
            torch.tensor(signs, device=device))


@torch.no_grad()
def eval_phi(phi, db, gts, attr_index, device, ks=KS):
    """Rank every valid source of every query through Φ_T2; return evaluate_all rows.

    db is the frozen test DB already on `device`. All sources of a query share the
    same conditions, so Φ runs once per query as a batch.
    """
    phi.eval()
    rankings = {}
    for qgt in gts:
        if not qgt.gt:
            continue
        cc, cs = query_conditions(qgt, attr_index, device)
        sources = list(qgt.gt)
        v_ref = db[sources]                                # [B, 512]
        B, C = len(sources), cc.numel()
        v_q = phi(v_ref,
                  cc.unsqueeze(0).expand(B, C),
                  cs.unsqueeze(0).expand(B, C),
                  torch.ones(B, C, dtype=torch.bool, device=device))
        rankings[qgt.query] = {s: rank(v_q[i], db, exclude={s}, k=max(ks))
                               for i, s in enumerate(sources)}
    return evaluate_all(rankings, gts, ks=ks)


def write_results(rows, name="t2"):
    """Freeze a per-query table (+ MACRO, + vs naive/clay/solution_a if present).

    Output is namespaced solution_a_<name> so it never collides with the colleague's
    solution_b_* tables.
    """
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"solution_a_{name}.json").write_text(json.dumps(rows, indent=2))

    cols = [f"{m}@{k}" for k in KS for m in ("recall", "precision")]
    ordered = [q for q in rows if q != "MACRO"] + (["MACRO"] if "MACRO" in rows else [])
    lines = [f"# Solution A — T2 learned directions (Φ = {name})", "",
             "| query | " + " | ".join(cols) + " | n_sources |",
             "|" + "---|" * (len(cols) + 2)]
    for q in ordered:
        r = rows[q]
        lines.append(f"| {q} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {r['n_sources']} |")

    # head-to-head vs the lower bound, the SOTA baseline, and no-training Solution A
    for base, tag, key in (("baseline_naive", "naive", None),
                           ("baseline_clay", "clay", "rows"),
                           ("solution_a", "sol-A", "rows")):
        p = RESULTS / f"{base}.json"
        if not p.exists():
            continue
        bj = json.loads(p.read_text())
        brows = bj.get(key, bj) if key else bj
        lines += ["", f"## vs {tag} (R@1 / R@5)", "",
                  f"| query | {tag} R@1 | {name} R@1 | {tag} R@5 | {name} R@5 |",
                  "|---|---|---|---|---|"]
        for q in ordered:
            if q in brows and q in rows:
                b, a = brows[q], rows[q]
                lines.append(f"| {q} | {b['recall@1']:.3f} | {a['recall@1']:.3f} "
                             f"| {b['recall@5']:.3f} | {a['recall@5']:.3f} |")
    (RESULTS / f"solution_a_{name}.md").write_text("\n".join(lines) + "\n")
    print(f"frozen -> {RESULTS / f'solution_a_{name}.md'}")


def main(ckpt=None, name="t2"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db = load_db(DB_TEST).float().to(device)
    gts = build_ground_truth(EVAL_JSON)
    attr_index = attr_index_test()

    ckpt = ckpt or RESULTS / f"phi_{name}.pt"
    phi = T2Phi().to(device)
    phi.load_state_dict(torch.load(ckpt, map_location=device))
    rows = eval_phi(phi, db, gts, attr_index, device)
    m = rows["MACRO"]
    print(f"{name}: R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f}")
    write_results(rows, name)


if __name__ == "__main__":
    main()
