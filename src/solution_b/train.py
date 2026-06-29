"""Shared training loop for Φ (CLIP frozen, query-side adapter).

Φ is swappable: MLPPhi (baseline, here), later T1 (cross-attention) and T2
(learned directions) implement the same contract and drop straight in. Training
is self-supervised on CelebA-train attributes via the sampler; validation runs on
the CelebA-test benchmark every `eval_every` steps; the best checkpoint (by R@1,
tie R@5) is saved and its results table frozen.

Run from repo root:
  python -m src.solution_b.train          # MLP baseline (default)
  python -m src.solution_b.train t1       # T1 cross-attention
"""
import sys
import torch
from torch import optim
from datetime import datetime

from src.common.paths import EVAL_JSON, DB_TEST, RESULTS
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.sampler import TrainData
from src.solution_b.losses import total_loss
from src.solution_b.phi import MLPPhi
from src.solution_b.t1_attention import T1Phi
from src.solution_b.run import eval_phi, write_results, attr_index_test

# defaults (tunable; sweep later)
STEPS = 6000
BATCH = 256
LR = 1e-3
LR_MIN = 0.0
TAU = 0.07
LAM_ID = 0.3
K_HARDNEG = 8
EVAL_EVERY = 500


def _log(msg, log_file):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def train(phi, data, gts, db, attr_index, device, log_file, *, steps=STEPS, batch=BATCH,
          lr=LR, lr_min=LR_MIN, tau=TAU, lam_id=LAM_ID, eval_every=EVAL_EVERY):
    """Optimize Φ; return (best_state_dict, best_rows). CLIP/DB untouched."""
    phi.to(device)
    opt = optim.Adam(phi.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr_min)
    best_key, best_rows, best_state = (-1.0, -1.0), None, None

    for step in range(1, steps + 1):
        phi.train()
        b = data.sample_batch(batch)
        v_q = phi(b['v_ref'], b['cond_col'], b['cond_sign'], b['cond_mask'])
        loss, parts = total_loss(v_q, b['v_ref'], b['pos_feat'], b['hneg_feat'],
                                 b['hneg_mask'], tau=tau, lam_id=lam_id)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if step % eval_every == 0 or step == steps:
            rows = eval_phi(phi, db, gts, attr_index, device)
            m = rows["MACRO"]
            cur_lr = scheduler.get_last_lr()[0]
            msg = (f"step {step:>5} | lr {cur_lr:.2e} | loss {loss.item():.3f} "
                   f"(nce {parts['info_nce']:.3f} id {parts['identity']:.3f}) | "
                   f"R@1 {m['recall@1']:.3f} R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f}")
            _log(msg, log_file)
            key = (m["recall@1"], m["recall@5"])
            if key > best_key:
                best_key, best_rows = key, rows
                best_state = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}

    return best_state, best_rows


def build_phi(name):
    if name == "mlp":
        return MLPPhi()
    if name == "t1":
        return T1Phi()
    raise ValueError(f"unknown phi: {name!r} (choices: mlp, t1)")


def main(name="mlp"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams = dict(steps=STEPS, batch=BATCH, lr=LR, lr_min=LR_MIN, tau=TAU, lam_id=LAM_ID,
                   k_hardneg=K_HARDNEG, eval_every=EVAL_EVERY)

    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS / f"train_{name}_{ts}.log"
    with open(log_path, "w") as log_file:
        header = (f"device: {device}  phi: {name}\n"
                  + "  ".join(f"{k}={v}" for k, v in hparams.items()))
        _log(header, log_file)

        data = TrainData(device=device, k_hardneg=K_HARDNEG)
        db = load_db(DB_TEST).float().to(device)
        gts = build_ground_truth(EVAL_JSON)
        attr_index = attr_index_test()

        phi = build_phi(name)
        train_params = {k: v for k, v in hparams.items() if k != 'k_hardneg'}
        best_state, best_rows = train(phi, data, gts, db, attr_index, device, log_file,
                                      **train_params)

        ckpt = RESULTS / f"phi_{name}.pt"
        torch.save(best_state, ckpt)
        _log(f"best checkpoint -> {ckpt}", log_file)
        write_results(best_rows, name)
        _log(f"frozen -> {RESULTS / f'solution_b_{name}.md'}", log_file)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    main(name)
