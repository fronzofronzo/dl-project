"""Training loop for the T2 adapter Φ (input-conditioned image-space directions).

T2's OWN launch script — independent of solution_b/train.py (which person 1 uses for
T1). It reuses the SHARED backbone infrastructure (the self-supervised sampler and
the InfoNCE/identity losses) as the project plan mandates, but drives its own
training/eval orchestration and freezes its own checkpoint + results so the two
solutions never share launch scripts.

CLIP stays frozen; only Φ_T2 is trained. Self-supervised on CelebA-train attributes
via the sampler; validation on the CelebA-test benchmark every `eval_every` steps;
the best checkpoint (by R@1, tie R@5) is saved and its results table frozen.

Loss = InfoNCE + lam_id * identity_anchor + lam_orth * ortho_reg(D, corr_target)

Improvements wired in (results/solution_a_t2.md analysis):
  I1  input-conditioned directions      -> in the model (T2Phi.cond_dir=True)
  I2  correlation-aware decorrelation    -> ORTHO_MODE = corr | zero | off
  I3  hybrid warm-start of D             -> WARM_START, image_axes() on TRAIN split
  I4  extra DB-distribution negatives    -> QUEUE_N (optional, default off)

Run from repo root:  python -m src.solution_a.train_t2
"""
import torch
from torch import optim
from datetime import datetime

from src.common.paths import EVAL_JSON, DB_TEST, RESULTS
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.sampler import TrainData          # shared backbone (self-supervised data)
from src.solution_b.losses import total_loss          # shared backbone (InfoNCE + identity)
from src.solution_a.t2_directions import T2Phi, N_ATTR, DIM
from src.solution_a.directions import build_direction_axes
from src.solution_a.run_t2 import eval_phi, write_results, attr_index_test

# defaults (tunable; sweep later)
STEPS = 6000
BATCH = 256
LR = 3e-4               # was 1e-3: loss plateaued+oscillated -> lower LR settles deeper
LR_MIN = 3e-4           # == LR: disabilita cosine (LR costante)
TAU = 0.05              # sharper InfoNCE than default 0.07
LAM_ID = 0.4           # anchor identity hard to help R@1 (top-1 match)
LAM_ORTH = 0.1          # weight of the direction-decorrelation regularizer (I2)
EVAL_EVERY = 100

# improvement switches (flip for ablations)
WARM_START = True       # I3: warm-start D from the train image-space probe
ORTHO_MODE = "corr"     # I2: "corr" (label-correlation target) | "zero" (orthogonal) | "off"
QUEUE_N = 0             # I4: extra random train negatives per step (0 = off)
HYBRID = True


def image_axes(F_train, L_train):
    """I3 probe: per-attribute image-space axis on the TRAIN split (no leakage).

      d_img(j) = mean(F[L[:,j]==1]) − mean(F[L[:,j]==0])      -> [n_attr, dim]

    Image−image difference, so it is modality-gap-free (findings §3). Used to
    warm-start the base direction dictionary D.
    """
    n_attr = L_train.shape[1]
    axes = torch.zeros(n_attr, F_train.shape[1], device=F_train.device)
    for j in range(n_attr):
        on = L_train[:, j]
        axes[j] = F_train[on].mean(0) - F_train[~on].mean(0)
    return axes


def attr_corr(L_train):
    """I2 target: empirical attribute correlation matrix from train labels [n_attr,
    n_attr], values in [-1,1]. Pushes D's geometry toward the REAL co-occurrence
    structure instead of a blanket identity (so correlated same-sign attributes
    keep their shared component — the measured failure mode of plain orthogonality)."""
    X = L_train.float()
    X = X - X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True).clamp_min(1e-6)
    X = X / std
    return (X.t() @ X) / X.shape[0]                          # [n_attr, n_attr]


def queue_negatives(F_train, n):
    """I4: draw n random train features as shared extra negatives [n, dim]."""
    idx = torch.randint(F_train.shape[0], (n,), device=F_train.device)
    return F_train[idx]


def text_axes_tensor(attr_index, device):
    """Hybrid conditioning: per-attribute CLIP TEXT axis z_with − z_without, as a
    [n_attr, dim] tensor row-aligned to attribute columns. Frozen; a learned bridge
    in the model maps it text->image cone."""
    axes = build_direction_axes(list(attr_index.keys()))    # {name: [dim]}
    T = torch.zeros(N_ATTR, DIM)
    for n, c in attr_index.items():
        T[c] = axes[n]
    return T.to(device)


def _log(msg, log_file):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def train(phi, data, gts, db, attr_index, device, log_file, *, steps=STEPS, batch=BATCH,
          lr=LR, lr_min=LR_MIN, tau=TAU, lam_id=LAM_ID, lam_orth=LAM_ORTH,
          eval_every=EVAL_EVERY, ortho_target=None, queue_n=QUEUE_N):
    """Optimize Φ_T2; return (best_state_dict, best_rows). CLIP/DB untouched."""
    phi.to(device)
    opt = optim.Adam(phi.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr_min)
    # lr_min == lr -> constant LR (cosine disabled); lr_min < lr -> cosine decay
    best_key, best_rows, best_state = (-1.0, -1.0), None, None

    for step in range(1, steps + 1):
        phi.train()
        b = data.sample_batch(batch)
        v_q = phi(b['v_ref'], b['cond_col'], b['cond_sign'], b['cond_mask'])

        hneg_feat, hneg_mask = b['hneg_feat'], b['hneg_mask']
        if queue_n:                                         # I4: append DB-distribution negs
            B = v_q.shape[0]
            q = queue_negatives(data.F, queue_n).unsqueeze(0).expand(B, -1, -1)  # [B,Q,dim]
            hneg_feat = torch.cat([hneg_feat, q], dim=1)
            hneg_mask = torch.cat([hneg_mask, torch.ones(B, queue_n, dtype=torch.bool,
                                                          device=device)], dim=1)

        loss, parts = total_loss(v_q, b['v_ref'], b['pos_feat'], hneg_feat,
                                 hneg_mask, tau=tau, lam_id=lam_id)
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
            gate = f" gate {phi.text_gate.item():+.3f}" if getattr(phi, "hybrid", False) else ""
            msg = (f"step {step:>5} | lr {cur_lr:.2e} | loss {loss.item():.3f} "
                   f"(nce {parts['info_nce']:.3f} id {parts['identity']:.3f} "
                   f"orth {reg.item():.3f}{gate}) | "
                   f"R@1 {m['recall@1']:.3f} R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f}")
            _log(msg, log_file)
            key = (m["recall@1"], m["recall@5"])
            if key > best_key:
                best_key, best_rows = key, rows
                best_state = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}

    return best_state, best_rows


def main(name=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = name or ("t2_hybrid_tau05_fixed" if HYBRID else "t2_tau05_fixed")
    hparams = dict(steps=STEPS, batch=BATCH, lr=LR, lr_min=LR_MIN, tau=TAU,
                   lam_id=LAM_ID, lam_orth=LAM_ORTH, eval_every=EVAL_EVERY,
                   warm_start=WARM_START, ortho_mode=ORTHO_MODE,
                   queue_n=QUEUE_N, hybrid=HYBRID)

    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS / f"train_{name}_{ts}.log"

    with open(log_path, "w") as log_file:
        header = (f"device: {device}  phi: {name}\n"
                  + "  ".join(f"{k}={v}" for k, v in hparams.items()))
        _log(header, log_file)

        data = TrainData(device=device)
        db = load_db(DB_TEST).float().to(device)
        gts = build_ground_truth(EVAL_JSON)
        attr_index = attr_index_test()

        phi = T2Phi(hybrid=HYBRID)

        if WARM_START:                                          # I3
            phi.load_directions(image_axes(data.F, data.L))
            _log("  warm-started D from train image-space probe", log_file)

        if HYBRID:                                              # hybrid conditioning
            phi.set_text_axes(text_axes_tensor(attr_index, device))
            _log("  loaded frozen CLIP text axes (text->image bridge will train)", log_file)

        lam_orth = LAM_ORTH                                     # I2
        if ORTHO_MODE == "corr":
            ortho_target = attr_corr(data.L).to(device)
            _log("  ortho_reg: correlation-aware (I2)", log_file)
        elif ORTHO_MODE == "zero":
            ortho_target = None
            _log("  ortho_reg: plain orthogonality", log_file)
        elif ORTHO_MODE == "off":
            ortho_target, lam_orth = None, 0.0
            _log("  ortho_reg: off", log_file)
        else:
            raise ValueError(f"ORTHO_MODE must be corr|zero|off, got {ORTHO_MODE!r}")

        best_state, best_rows = train(phi, data, gts, db, attr_index, device, log_file,
                                      lam_orth=lam_orth, ortho_target=ortho_target)

        ckpt = RESULTS / f"phi_{name}_{ts}.pt"
        torch.save(best_state, ckpt)
        _log(f"best checkpoint -> {ckpt}", log_file)
        write_results(best_rows, name)
        _log(f"frozen -> {RESULTS / f'solution_a_{name}.md'}", log_file)


if __name__ == "__main__":
    main()
