"""Ablation study for T2 hybrid: runs I1, I3, I2 in sequence.

Each ablation disables exactly one improvement vs the T2 hybrid baseline
(tau=0.07, fixed LR 3e-4, EVAL_EVERY=500 for speed):

  I1  cond_dir=False    rigid global directions (no low-rank correction)
  I3  warm_start=False  random D init (no image-space probe)
  I2  ortho_mode=off    no decorrelation regularizer on D

Run from repo root:
  python -m src.solution_a.run_ablations
"""
import torch
from torch import optim
from datetime import datetime

from src.common.paths import EVAL_JSON, DB_TEST, RESULTS, DB_TRAIN, ATTRS_TRAIN
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.sampler import TrainData
from src.solution_b.losses import total_loss
from src.solution_a.t2_directions import T2Phi, N_ATTR, DIM
from src.solution_a.directions import build_direction_axes
from src.solution_a.run_t2 import eval_phi, write_results, attr_index_test
from src.solution_a.train_t2 import image_axes, attr_corr, text_axes_tensor, _log

# Shared baseline hyperparams (match T2 hybrid: tau=0.07, fixed LR)
STEPS      = 6000
BATCH      = 256
LR         = 3e-4
TAU        = 0.07
LAM_ID     = 0.4
LAM_ORTH   = 0.1
EVAL_EVERY = 500   # faster than baseline's 100 — ablations need final number, not best ckpt


def run_one(name, phi, data, db, gts, attr_index, device,
            ortho_target, lam_orth, log_file):
    phi.to(device)
    opt = optim.Adam(phi.parameters(), lr=LR)
    best_key, best_rows, best_state = (-1.0, -1.0), None, None

    for step in range(1, STEPS + 1):
        phi.train()
        b = data.sample_batch(BATCH)
        v_q = phi(b['v_ref'], b['cond_col'], b['cond_sign'], b['cond_mask'])
        loss, parts = total_loss(v_q, b['v_ref'], b['pos_feat'],
                                 b['hneg_feat'], b['hneg_mask'],
                                 tau=TAU, lam_id=LAM_ID)
        reg = phi.ortho_reg(target=ortho_target)
        loss = loss + lam_orth * reg
        opt.zero_grad(); loss.backward(); opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS:
            rows = eval_phi(phi, db, gts, attr_index, device)
            m = rows["MACRO"]
            gate = f" gate {phi.text_gate.item():+.3f}" if getattr(phi, "hybrid", False) else ""
            msg = (f"  step {step:>5} | loss {loss.item():.3f} "
                   f"(nce {parts['info_nce']:.3f} id {parts['identity']:.3f} "
                   f"orth {reg.item():.3f}{gate}) | "
                   f"R@1 {m['recall@1']:.3f} R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f}")
            _log(msg, log_file)
            key = (m["recall@1"], m["recall@5"])
            if key > best_key:
                best_key, best_rows = key, rows
                best_state = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}

    ckpt = RESULTS / f"phi_{name}.pt"
    torch.save(best_state, ckpt)
    _log(f"  checkpoint -> {ckpt}", log_file)
    write_results(best_rows, name)
    _log(f"  results   -> {RESULTS / f'solution_a_{name}.md'}\n", log_file)
    return best_rows["MACRO"]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS / f"ablations_{ts}.log"

    ablations = [
        # (name, cond_dir, warm_start, ortho_mode, hybrid)
        ("t2_hybrid_abl_i1", False, True,  "corr", True),   # I1 off: rigid directions
        ("t2_hybrid_abl_i3", True,  False, "corr", True),   # I3 off: no warm-start
        ("t2_hybrid_abl_i2", True,  True,  "off",  True),   # I2 off: no ortho_reg
    ]

    with open(log_path, "w") as log_file:
        _log(f"device: {device}  steps={STEPS} batch={BATCH} lr={LR} "
             f"tau={TAU} lam_id={LAM_ID} lam_orth={LAM_ORTH} eval_every={EVAL_EVERY}",
             log_file)
        _log(f"T2 hybrid baseline: R@1=0.090 R@5=0.240 R@10=0.349\n", log_file)

        data       = TrainData(device=device)
        db         = load_db(DB_TEST).float().to(device)
        gts        = build_ground_truth(EVAL_JSON)
        attr_index = attr_index_test()

        # pre-compute shared resources
        img_axes   = image_axes(data.F, data.L)
        corr_mat   = attr_corr(data.L).to(device)
        txt_axes   = text_axes_tensor(attr_index, device)

        summary = []
        for name, cond_dir, warm_start, ortho_mode, hybrid in ablations:
            _log(f"=== {name}  cond_dir={cond_dir} warm_start={warm_start} "
                 f"ortho={ortho_mode} hybrid={hybrid} ===", log_file)

            phi = T2Phi(cond_dir=cond_dir, hybrid=hybrid)

            if warm_start:
                phi.load_directions(img_axes)
                _log("  warm-started D", log_file)
            else:
                _log("  D initialized randomly", log_file)

            if hybrid:
                phi.set_text_axes(txt_axes)

            if ortho_mode == "corr":
                ortho_target, lam_orth = corr_mat, LAM_ORTH
                _log("  ortho_reg: correlation-aware", log_file)
            elif ortho_mode == "off":
                ortho_target, lam_orth = None, 0.0
                _log("  ortho_reg: off", log_file)

            macro = run_one(name, phi, data, db, gts, attr_index, device,
                            ortho_target, lam_orth, log_file)
            summary.append((name, macro))

        _log("=== ABLATION SUMMARY ===", log_file)
        _log(f"  {'model':<35} R@1    R@5    R@10", log_file)
        _log(f"  {'T2 hybrid (baseline)':<35} 0.090  0.240  0.349", log_file)
        for name, m in summary:
            tag = name.replace("t2_hybrid_abl_", "").upper()
            label = f"  ablation {tag} (disabled)"
            _log(f"  {label:<35} {m['recall@1']:.3f}  {m['recall@5']:.3f}  {m['recall@10']:.3f}",
                 log_file)


if __name__ == "__main__":
    main()
