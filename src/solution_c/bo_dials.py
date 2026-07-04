"""Solution C — TPE search (Optuna) over the inference dials, on the TRAIN bench.

The operating point of the flow was picked by exhaustive search on coarse grids
(N x lambda x T sweep on test, eta grid on train). This module asks whether
that optimum is a grid artifact: a TPE sampler explores the CONTINUOUS dial box

    N in {1..8}   Euler steps
    T in [0.5,1]  integration horizon
    lambda in [0, 0.5]        probe guidance on the query (known-dead, included
                              so the sampler can rediscover that on its own)
    eta   in log[1e-4, 3]     PoE re-rank weight (1e-4 is effectively 0: the
                              penalty is far below typical cosine gaps)
    weighting in {none, acc}  per-attribute reliability weighting

against the same pre-declared objective used everywhere else in Solution C
(0.5*R@1 + 0.5*R@5, macro on the TRAIN pseudo-benchmark — test is never
touched). Report use: if the BO converges to the grid cell, the grid optimum
is not an artifact of the grid resolution.

Run from repo root:
    python -m src.solution_c.bo_dials --smoke
    python -m src.solution_c.bo_dials --ckpt results/phi_flow_flow_v3.pt --trials 40
"""
import argparse

import torch

from src.common.metrics import evaluate_all
from src.common.paths import DB_TRAIN, RESULTS
from src.common.retrieval import load_db
from src.solution_b.run import attr_index_test
from src.solution_c.probes import PROBES_PT
from src.solution_c.rerank import (precompute_logits, compose_queries, rank_poe,
                                   _score)
from src.solution_c.run_flow import load_flow


def make_objective(phi, db, Z, gts, attr_index, device, acc_w, cache):
    """Objective closure: dials -> train-bench score. Compositions are cached
    per (N, T, lambda) so eta/weighting-only moves cost a re-rank, not a
    re-integration."""
    def objective(trial):
        n = trial.suggest_int("n_steps", 1, 8)
        T = trial.suggest_float("horizon", 0.5, 1.0)
        lam = trial.suggest_float("guidance", 0.0, 0.5)
        eta = trial.suggest_float("eta", 1e-4, 3.0, log=True)
        wtag = trial.suggest_categorical("weighting", ["none", "acc"])

        key = (n, round(T, 4), round(lam, 4))
        if key not in cache:
            phi.n_steps, phi.horizon, phi.guidance = n, T, lam
            cache[key] = compose_queries(phi, db, gts, attr_index, device)
        weights = acc_w if wtag == "acc" else None
        rows = evaluate_all(
            rank_poe(db, Z, gts, cache[key], attr_index, device, eta, weights), gts)
        return _score(rows)
    return objective


def run_study(objective, trials, seed, enqueue_grid=False):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if enqueue_grid:
        # seed the search with the grid operating point (η=1e-4 is the box's ≈0),
        # so best >= grid by construction and the study answers exactly one
        # question: is there anything better in the box?
        study.enqueue_trial({"n_steps": 8, "horizon": 1.0, "guidance": 0.0,
                             "eta": 1e-4, "weighting": "none"})
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study


def write_report(study, name, trials, seed, n_sources):
    lines = [f"# Solution C — TPE (Optuna) over the inference dials ({name})", "",
             f"{trials} trials, TPESampler(seed={seed}), objective 0.5·R@1+0.5·R@5 "
             f"on the TRAIN pseudo-benchmark ({n_sources} sources/query). "
             "Test is never touched; this is a grid-artifact check for the report.", "",
             "| trial | N | T | λ | η | weighting | score |",
             "|---|---|---|---|---|---|---|"]
    for t in study.trials:
        p = t.params
        lines.append(f"| {t.number} | {p['n_steps']} | {p['horizon']:.3f} "
                     f"| {p['guidance']:.3f} | {p['eta']:.4f} | {p['weighting']} "
                     f"| {t.value:.4f} |")
    b = study.best_trial
    lines += ["", f"**best: N={b.params['n_steps']}, T={b.params['horizon']:.3f}, "
                  f"λ={b.params['guidance']:.3f}, η={b.params['eta']:.4f}, "
                  f"weighting={b.params['weighting']} (train score {b.value:.4f})**", "",
              "Grid-search reference (same objective, same bench): N=8, T=1.0, "
              "λ=0, η=0 — if the BO optimum sits in that corner, the coarse grids "
              "did not miss a better operating point."]
    out = RESULTS / f"solution_c_{name}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"BO report -> {out}")
    print(f"best: {b.params} score={b.value:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(RESULTS / "phi_flow_flow_v3.pt"))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sources", type=int, default=200, help="train-bench sources/query")
    ap.add_argument("--bench-seed", type=int, default=0)
    ap.add_argument("--probes", default=str(PROBES_PT))
    ap.add_argument("--name", default="bo_dials")
    ap.add_argument("--enqueue-grid", action="store_true",
                    help="seed trial 0 with the grid operating point (N=8, T=1, λ=0, η≈0)")
    args = ap.parse_args()

    from src.solution_c.train_bench import build_train_gts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phi = load_flow(args.ckpt, device)
    blob = torch.load(args.probes, map_location=device)
    W, b = blob["W"].to(device).float(), blob["b"].to(device).float()
    acc_w = (2.0 * blob["acc_val"].to(device).float() - 1.0).clamp(0.0, 1.0)
    phi.set_probes(W, b)                                  # lambda dial needs them

    db = load_db(DB_TRAIN).float().to(device)
    gts = build_train_gts(n_sources=args.n_sources, seed=args.bench_seed,
                          device=device)
    attr_index = attr_index_test()
    Z = precompute_logits(db, W, b)

    objective = make_objective(phi, db, Z, gts, attr_index, device, acc_w, {})
    study = run_study(objective, args.trials, args.seed, args.enqueue_grid)
    write_report(study, args.name, args.trials, args.seed, args.n_sources)


# --------------------------------------------------------------------------- #
# Smoke: synthetic data + stub phi, a 5-trial study end to end, no files
# --------------------------------------------------------------------------- #
def _smoke():
    import torch.nn as nn
    import torch.nn.functional as F
    from src.common.groundtruth import QueryGT

    torch.manual_seed(0)
    N, n_attr = 80, 8
    db = F.normalize(torch.randn(N, 512), dim=1)
    W, b = torch.randn(n_attr, 512), torch.randn(n_attr)
    acc_w = torch.rand(n_attr)
    attr_index = {f"A{i}": i for i in range(n_attr)}
    gts = [QueryGT("+A1, -A3", pos=["A1"], neg=["A3"], gt={4: {7, 9, 11}}),
           QueryGT("+A2", pos=["A2"], gt={5: {30, 31, 32}})]

    class StubPhi(nn.Module):
        """Contract-shaped stub whose output depends on the dials (so distinct
        trials produce distinct scores)."""
        def __init__(self):
            super().__init__()
            self.n_steps, self.horizon, self.guidance = 8, 1.0, 0.0
            self.register_buffer("off", torch.randn(512))

        def forward(self, v_ref, cond_col, cond_sign, cond_mask):
            a = 0.01 * self.n_steps * self.horizon + self.guidance
            return F.normalize(v_ref + a * self.off, dim=1)

    Z = precompute_logits(db, W, b)
    cache = {}
    objective = make_objective(StubPhi(), db, Z, gts, attr_index, "cpu", acc_w, cache)
    study = run_study(objective, trials=5, seed=0)

    assert len(study.trials) == 5
    assert all(t.value is not None and 0.0 <= t.value <= 1.0 for t in study.trials)
    assert len(cache) >= 1, "composition cache never filled"
    for key in ("n_steps", "horizon", "guidance", "eta", "weighting"):
        assert key in study.best_trial.params
    print(f"  ok  5-trial study, best score {study.best_value:.4f}, "
          f"{len(cache)} cached compositions")
    print("smoke test passed.")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke()
    else:
        main()
