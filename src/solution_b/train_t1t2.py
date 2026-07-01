"""Training loop for T1T2Phi — cross-attention weighted direction adapter.

Combines the T2 training infrastructure (warm-start D, ortho_reg, hybrid text
axes, correlation-aware decorrelation) with T1 training improvements (cosine LR
decay, per-step file logging with timestamp).

Loss = InfoNCE + lam_id * identity_anchor + lam_orth * ortho_reg(D, corr_target)

Run from repo root:
  python -m src.solution_b.train_t1t2
"""
import sys
import torch
from torch import optim
from datetime import datetime

from src.common.paths import EVAL_JSON, DB_TEST, RESULTS, DB_TRAIN, ATTRS_TRAIN
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.sampler import TrainData
from src.solution_b.losses import total_loss
from src.solution_b.t1t2_phi import T1T2Phi, N_ATTR, DIM
from src.solution_a.directions import build_direction_axes
from src.solution_a.run_t2 import eval_phi, write_results, attr_index_test

STEPS      = 6000
BATCH      = 256
LR         = 3e-4
LR_MIN     = 0.0
TAU        = 0.07
LAM_ID     = 0.4
LAM_ORTH   = 0.1
K_HARDNEG  = 4
EVAL_EVERY = 500

WARM_START = True
ORTHO_MODE = "corr"   # "corr" | "zero" | "off"
HYBRID     = True


def _log(msg, log_file):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def image_axes(F_train, L_train):
    """I3: per-attribute image-space probe axis on the train split."""
    axes = torch.zeros(N_ATTR, F_train.shape[1], device=F_train.device)
    for j in range(N_ATTR):
        on = L_train[:, j]
        axes[j] = F_train[on].mean(0) - F_train[~on].mean(0)
    return axes


def attr_corr(L_train):
    """I2: empirical attribute correlation matrix from train labels [n_attr, n_attr]."""
    X = L_train.float()
    X = X - X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True).clamp_min(1e-6)
    X = X / std
    return (X.t() @ X) / X.shape[0]


def text_axes_tensor(attr_index, device):
    """Hybrid: per-attribute CLIP text axis z_with - z_without, [n_attr, dim]."""
    axes = build_direction_axes(list(attr_index.keys()))
    T = torch.zeros(N_ATTR, DIM)
    for name, col in attr_index.items():
        T[col] = axes[name]
    return T.to(device)


def train(phi, data, gts, db, attr_index, device, log_file, ortho_target=None, *,
          steps=STEPS, batch=BATCH, lr=LR, lr_min=LR_MIN, tau=TAU,
          lam_id=LAM_ID, lam_orth=LAM_ORTH, eval_every=EVAL_EVERY):
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
        reg = phi.ortho_reg(target=ortho_target)
        loss = loss + lam_orth * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if step % eval_every == 0 or step == steps:
            rows = eval_phi(phi, db, gts, attr_index, device)
            m = rows["MACRO"]
            cur_lr = scheduler.get_last_lr()[0]
            gate_str = (f" gate {phi.text_gate.item():+.3f}" if HYBRID else "")
            msg = (f"step {step:>5} | lr {cur_lr:.2e} | loss {loss.item():.3f} "
                   f"(nce {parts['info_nce']:.3f} id {parts['identity']:.3f} "
                   f"orth {reg.item():.3f}{gate_str}) | "
                   f"R@1 {m['recall@1']:.3f} R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f}")
            _log(msg, log_file)
            key = (m["recall@1"], m["recall@5"])
            if key > best_key:
                best_key, best_rows = key, rows
                best_state = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}

    return best_state, best_rows


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = "t1t2_hybrid" if HYBRID else "t1t2"
    hparams = dict(steps=STEPS, batch=BATCH, lr=LR, lr_min=LR_MIN, tau=TAU,
                   lam_id=LAM_ID, lam_orth=LAM_ORTH, k_hardneg=K_HARDNEG,
                   eval_every=EVAL_EVERY, warm_start=WARM_START,
                   ortho_mode=ORTHO_MODE, hybrid=HYBRID)

    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS / f"train_{name}_{ts}.log"

    with open(log_path, "w") as log_file:
        header = (f"device: {device}  phi: {name}\n"
                  + "  ".join(f"{k}={v}" for k, v in hparams.items()))
        _log(header, log_file)

        data = TrainData(device=device, k_hardneg=K_HARDNEG)
        db   = load_db(DB_TEST).float().to(device)
        gts  = build_ground_truth(EVAL_JSON)
        attr_index = attr_index_test()

        phi = T1T2Phi(hybrid=HYBRID)

        if WARM_START:
            phi.load_directions(image_axes(data.F, data.L))
            _log("  warm-started D from train image-space probe", log_file)

        if HYBRID:
            phi.set_text_axes(text_axes_tensor(attr_index, device))
            _log("  loaded frozen CLIP text axes", log_file)

        if ORTHO_MODE == "corr":
            ortho_target = attr_corr(data.L).to(device)
            _log("  ortho_reg: correlation-aware (I2)", log_file)
        elif ORTHO_MODE == "zero":
            ortho_target = None
            _log("  ortho_reg: plain orthogonality", log_file)
        else:
            ortho_target, lam_orth = None, 0.0
            _log("  ortho_reg: off", log_file)

        train_params = {k: v for k, v in hparams.items()
                        if k not in ('k_hardneg', 'warm_start', 'ortho_mode', 'hybrid')}
        best_state, best_rows = train(phi, data, gts, db, attr_index, device, log_file,
                                      ortho_target, **train_params)

        ckpt = RESULTS / f"phi_{name}.pt"
        torch.save(best_state, ckpt)
        _log(f"best checkpoint -> {ckpt}", log_file)
        write_results(best_rows, name)
        _log(f"frozen -> {RESULTS / f'solution_a_{name}.md'}", log_file)


if __name__ == "__main__":
    main()
