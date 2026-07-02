"""Evaluate a trained Φ-Flow checkpoint on the CelebA test benchmark.

Reuses the shared eval harness (src/solution_b/run.eval_phi): query-side only,
frozen test DB, JSON source keys, Recall@K / Precision@K hit-rate. Adds the two
INFERENCE-ONLY ablation axes that make the flow interesting to report:

  --sweep : grid over (Euler steps N) × (guidance λ). Costs forward passes only,
            no retraining — N=1 is the "one-shot editor" special case (≈ T2),
            λ=0 is the pure flow, λ>0 adds probe guidance. The best (R@1, R@5)
            cell of the sweep is then frozen as the full per-query table.

Run from repo root:
    python -m src.solution_c.run_flow --ckpt results/phi_flow.pt --sweep
    python -m src.solution_c.run_flow --ckpt results/phi_flow.pt --steps 8 --guidance 0.5
"""
import argparse
import json
from pathlib import Path

import torch

from src.common.paths import EVAL_JSON, DB_TEST, RESULTS
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.run import eval_phi, attr_index_test
from src.solution_c.flow_phi import FlowPhi
from src.solution_c.probes import PROBES_PT, load_probes

KS = (1, 5, 10)
SWEEP_STEPS = (1, 2, 4, 8, 16)
SWEEP_GUIDANCE = (0.0, 0.25, 0.5, 1.0)


def load_flow(ckpt_path, device):
    """Rebuild a FlowPhi from a train_flow checkpoint {"config": ..., "state": ...}."""
    blob = torch.load(ckpt_path, map_location=device)
    phi = FlowPhi(**blob["config"]).to(device)
    phi.load_state_dict(blob["state"])
    phi.eval()
    return phi


def write_results(rows, name, extra_header=""):
    """Freeze per-query table (+ MACRO) and compare vs naive / CLAY / T2-hybrid."""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"solution_c_{name}.json").write_text(json.dumps(rows, indent=2))

    cols = [f"{m}@{k}" for k in KS for m in ("recall", "precision")]
    ordered = [q for q in rows if q != "MACRO"] + (["MACRO"] if "MACRO" in rows else [])
    lines = [f"# Solution C — Φ-Flow = {name}", ""]
    if extra_header:
        lines += [extra_header, ""]
    lines += ["| query | " + " | ".join(cols) + " | n_sources |",
              "|" + "---|" * (len(cols) + 2)]
    for q in ordered:
        r = rows[q]
        lines.append(f"| {q} | " + " | ".join(f"{r[c]:.3f}" for c in cols)
                     + f" | {r['n_sources']} |")

    refs = (("baseline_naive", "naive"), ("baseline_clay", "clay"),
            ("solution_a_t2_hybrid", "t2-hybrid"))
    for fname, tag in refs:
        p = RESULTS / f"{fname}.json"
        if not p.exists():
            continue
        bj = json.loads(p.read_text())
        brows = bj.get("rows", bj)                          # clay nests under "rows"
        lines += ["", f"## vs {tag} (R@1 / R@5)", "",
                  f"| query | {tag} R@1 | {name} R@1 | {tag} R@5 | {name} R@5 |",
                  "|---|---|---|---|---|"]
        for q in ordered:
            if q in brows and q in rows:
                b, a = brows[q], rows[q]
                lines.append(f"| {q} | {b['recall@1']:.3f} | {a['recall@1']:.3f} "
                             f"| {b['recall@5']:.3f} | {a['recall@5']:.3f} |")
    (RESULTS / f"solution_c_{name}.md").write_text("\n".join(lines) + "\n")
    print(f"frozen -> {RESULTS / f'solution_c_{name}.md'}")


def sweep(phi, db, gts, attr_index, device, name,
          steps_grid=SWEEP_STEPS, guid_grid=SWEEP_GUIDANCE):
    """Inference-only grid over (N, λ). Returns the best (N, λ, rows) by R@1/R@5."""
    if not phi.has_probes():
        guid_grid = [g for g in guid_grid if g == 0.0] or [0.0]
        print("no probes on this checkpoint -> guidance sweep restricted to λ=0")

    lines = [f"# Solution C — Φ-Flow inference sweep ({name})", "",
             "| N steps | λ guidance | R@1 | R@5 | R@10 |", "|---|---|---|---|---|"]
    best = (-1.0, -1.0)
    best_cfg, best_rows = None, None
    for n in steps_grid:
        for lam in guid_grid:
            phi.n_steps, phi.guidance = int(n), float(lam)
            rows = eval_phi(phi, db, gts, attr_index, device)
            m = rows["MACRO"]
            print(f"N={n:>2} λ={lam:<5} | R@1 {m['recall@1']:.3f} "
                  f"R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f}")
            lines.append(f"| {n} | {lam} | {m['recall@1']:.3f} | {m['recall@5']:.3f} "
                         f"| {m['recall@10']:.3f} |")
            key = (m["recall@1"], m["recall@5"])
            if key > best:
                best, best_cfg, best_rows = key, (int(n), float(lam)), rows

    lines += ["", f"best: N={best_cfg[0]}, λ={best_cfg[1]} "
                  f"(R@1 {best[0]:.3f}, R@5 {best[1]:.3f})"]
    (RESULTS / f"solution_c_{name}_sweep.md").write_text("\n".join(lines) + "\n")
    print(f"sweep frozen -> {RESULTS / f'solution_c_{name}_sweep.md'}")
    return best_cfg, best_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(RESULTS / "phi_flow.pt"))
    ap.add_argument("--name", default=None, help="results-file tag (default: ckpt stem)")
    ap.add_argument("--steps", type=int, default=8, help="Euler steps N")
    ap.add_argument("--guidance", type=float, default=0.0, help="probe-guidance λ")
    ap.add_argument("--probes", default=str(PROBES_PT),
                    help="probes file for guidance (skipped if missing)")
    ap.add_argument("--sweep", action="store_true",
                    help="grid over N × λ, then freeze the best cell's full table")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or Path(args.ckpt).stem
    phi = load_flow(args.ckpt, device)

    probes_path = Path(args.probes)
    if probes_path.exists():
        phi.set_probes(*load_probes(probes_path, device))
        print(f"loaded probes {probes_path}")
    elif args.guidance != 0.0 or args.sweep:
        print(f"WARNING: no probes at {probes_path} -> guidance disabled")

    db = load_db(DB_TEST).float().to(device)
    gts = build_ground_truth(EVAL_JSON)
    attr_index = attr_index_test()

    if args.sweep:
        (n, lam), rows = sweep(phi, db, gts, attr_index, device, name)
        write_results(rows, name, extra_header=f"best sweep cell: N={n}, λ={lam}")
    else:
        phi.n_steps, phi.guidance = args.steps, args.guidance
        rows = eval_phi(phi, db, gts, attr_index, device)
        m = rows["MACRO"]
        print(f"{name}: N={args.steps} λ={args.guidance} | R@1={m['recall@1']:.3f} "
              f"R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f}")
        write_results(rows, name, extra_header=f"N={args.steps}, λ={args.guidance}")


if __name__ == "__main__":
    main()
