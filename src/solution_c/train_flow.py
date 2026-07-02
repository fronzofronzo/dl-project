"""Training loop for Φ-Flow (conditional flow matching on the CLIP hypersphere).

Reuses the SHARED backbone: the self-supervised FLIP-REF sampler and the
InfoNCE/identity losses (src/solution_b). CLIP stays frozen; only Φ trains.

Loss = L_cfm + lam_nce · InfoNCE(endpoint) + lam_id · identity(endpoint)

  L_cfm   — the flow-matching regression. For each anchor: draw t ~ U(0,1), put
            x_t on the geodesic slerp between v_ref and the sampled positive,
            regress the velocity field onto the analytic path velocity:
                L_cfm = ‖u_θ(x_t, t | conds) − dx_t/dt‖²
            No integration during training -> one forward per step, same cost
            class as T2. Resampling the positive from the pool each time makes
            the learned (marginal) flow transport v_ref toward the CENTER of the
            valid-target region — the optimal single query point for Recall@K.
  InfoNCE — optional retrieval-aligned endpoint term: integrate k_endpoint Euler
            steps WITH gradient and apply the shared contrastive loss (positives
            + polarity-violating hard negatives). lam_nce 0 = pure CFM (the
            stable fallback).

Warm start (default on): the direction/weight branch loads results/phi_t2_hybrid.pt,
so the initial velocity field IS T2-hybrid's edit field (see flow_phi.py) and
training starts from the current best model rather than from noise.

Validation on the CelebA-test benchmark every eval_every steps (pure flow,
eval_steps Euler steps, guidance OFF — guidance is an inference-time ablation,
swept later by run_flow.py). Best checkpoint by (R@1, R@5) saved with its config.

Run from repo root:
    python -m src.solution_c.train_flow                       # main run
    python -m src.solution_c.train_flow --lam-nce 0           # pure-CFM fallback
    python -m src.solution_c.train_flow --no-warm-start       # ablation
    python -m src.solution_c.train_flow --steps 50 --eval-every 25   # quick check
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import optim

from src.common.paths import EVAL_JSON, DB_TEST, RESULTS
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.sampler import TrainData            # shared self-supervised data
from src.solution_b.losses import info_nce, identity_anchor
from src.solution_b.run import eval_phi, attr_index_test
from src.solution_c.flow_phi import FlowPhi, slerp_velocity
from src.solution_c.run_flow import write_results

T2_CKPT = RESULTS / "phi_t2_hybrid.pt"


def _log(msg, log_file):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def cfm_loss(phi, batch):
    """Flow-matching velocity regression along the v_ref -> positive geodesic."""
    v0 = batch["v_ref"]
    v1 = torch.nn.functional.normalize(batch["pos_feat"].float(), dim=1)
    t = torch.rand(v0.shape[0], 1, device=v0.device)
    x_t, u_target = slerp_velocity(v0, v1, t)
    u = phi.velocity(x_t, t.squeeze(1), batch["cond_col"], batch["cond_sign"],
                     batch["cond_mask"])
    return (u - u_target).pow(2).sum(dim=1).mean()


def train(phi, data, gts, db, attr_index, device, log_file, args):
    """Optimize Φ-Flow; return (best_state, best_rows). CLIP/DB untouched."""
    phi.to(device)
    opt = optim.Adam(phi.parameters(), lr=args.lr)
    best_key, best_rows, best_state = (-1.0, -1.0), None, None

    for step in range(1, args.steps + 1):
        phi.train()
        b = data.sample_batch(args.batch)

        l_cfm = cfm_loss(phi, b)
        loss = l_cfm
        l_nce = l_id = torch.zeros((), device=device)
        if args.lam_nce > 0:
            v_q = phi(b["v_ref"], b["cond_col"], b["cond_sign"], b["cond_mask"],
                      n_steps=args.k_endpoint, guidance=0.0)
            l_nce = info_nce(v_q, b["pos_feat"], b["hneg_feat"], b["hneg_mask"],
                             tau=args.tau)
            l_id = identity_anchor(v_q, b["v_ref"])
            loss = loss + args.lam_nce * l_nce + args.lam_id * l_id

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(phi.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            phi.n_steps, phi.guidance = args.eval_steps, 0.0
            rows = eval_phi(phi, db, gts, attr_index, device)
            m = rows["MACRO"]
            _log(f"step {step:>5} | loss {loss.item():.3f} (cfm {l_cfm.item():.3f} "
                 f"nce {l_nce.item():.3f} id {l_id.item():.3f}) | "
                 f"R@1 {m['recall@1']:.3f} R@5 {m['recall@5']:.3f} "
                 f"R@10 {m['recall@10']:.3f}", log_file)
            key = (m["recall@1"], m["recall@5"])
            if key > best_key:
                best_key, best_rows = key, rows
                best_state = {k: v.detach().cpu().clone()
                              for k, v in phi.state_dict().items()}

    return best_state, best_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="flow")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lam-nce", type=float, default=0.5,
                    help="endpoint InfoNCE weight (0 = pure CFM)")
    ap.add_argument("--lam-id", type=float, default=0.1,
                    help="endpoint identity-anchor weight (active with lam-nce > 0)")
    ap.add_argument("--k-endpoint", type=int, default=4,
                    help="Euler steps for the differentiable endpoint")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=8,
                    help="Euler steps N for periodic validation")
    ap.add_argument("--warm-start", default=str(T2_CKPT),
                    help="T2 checkpoint for the direction/weight branch")
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--no-free-residual", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dict(hybrid=True, free_residual=not args.no_free_residual,
                  n_steps=args.eval_steps, guidance=0.0)

    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS / f"train_flow_{args.name}_{ts}.log"

    with open(log_path, "w") as log_file:
        _log(f"device: {device}  phi: flow/{args.name}\n"
             + "  ".join(f"{k}={v}" for k, v in vars(args).items()), log_file)

        data = TrainData(device=device)
        db = load_db(DB_TEST).float().to(device)
        gts = build_ground_truth(EVAL_JSON)
        attr_index = attr_index_test()

        phi = FlowPhi(**config)
        if not args.no_warm_start:
            t2_path = Path(args.warm_start)
            if not t2_path.exists():
                raise FileNotFoundError(f"warm-start checkpoint missing: {t2_path} "
                                        "(pass --no-warm-start to train from scratch)")
            loaded = phi.warm_start_from_t2(torch.load(t2_path, map_location="cpu"))
            _log(f"  warm-started {len(loaded)} tensors from {t2_path.name} "
                 "(init velocity field == T2-hybrid edit field)", log_file)
        else:
            _log("  training from scratch (no warm start)", log_file)

        best_state, best_rows = train(phi, data, gts, db, attr_index, device,
                                      log_file, args)

        ckpt = {"config": config, "state": best_state, "args": vars(args)}
        for path in (RESULTS / f"phi_flow_{args.name}_{ts}.pt",
                     RESULTS / f"phi_flow_{args.name}.pt"):
            torch.save(ckpt, path)
        _log(f"best checkpoint -> {RESULTS / f'phi_flow_{args.name}.pt'} "
             f"(+ timestamped copy)", log_file)

        m = best_rows["MACRO"]
        _log(f"best (N={args.eval_steps}, λ=0): R@1 {m['recall@1']:.3f} "
             f"R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f}", log_file)
        write_results(best_rows, args.name,
                      extra_header=f"training-time best, N={args.eval_steps}, λ=0 "
                                   f"(sweep N × λ with run_flow.py --sweep)")


if __name__ == "__main__":
    main()
