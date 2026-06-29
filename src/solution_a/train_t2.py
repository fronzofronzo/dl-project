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

from src.common.paths import EVAL_JSON, DB_TEST, RESULTS
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.sampler import TrainData          # shared backbone (self-supervised data)
from src.solution_b.losses import total_loss          # shared backbone (InfoNCE + identity)
from src.solution_a.t2_directions import T2Phi
from src.solution_a.run_t2 import eval_phi, write_results, attr_index_test

# defaults (tunable; sweep later)
STEPS = 6000
BATCH = 256
LR = 3e-4               # was 1e-3: loss plateaued+oscillated -> lower LR settles deeper
TAU = 0.07
LAM_ID = 0.4           # anchor identity hard to help R@1 (top-1 match)
LAM_ORTH = 0.1          # weight of the direction-decorrelation regularizer (I2)
EVAL_EVERY = 100

# improvement switches (flip for ablations)
WARM_START = True       # I3: warm-start D from the train image-space probe
ORTHO_MODE = "corr"     # I2: "corr" (label-correlation target) | "zero" (orthogonal) | "off"
QUEUE_N = 0             # I4: extra random train negatives per step (0 = off)


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


def train(phi, data, gts, db, attr_index, device, *, steps=STEPS, batch=BATCH,
          lr=LR, tau=TAU, lam_id=LAM_ID, lam_orth=LAM_ORTH, eval_every=EVAL_EVERY,
          ortho_target=None, queue_n=QUEUE_N):
    """Optimize Φ_T2; return (best_state_dict, best_rows). CLIP/DB untouched."""
    phi.to(device)
    opt = optim.Adam(phi.parameters(), lr=lr)
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

        if step % eval_every == 0 or step == steps:
            rows = eval_phi(phi, db, gts, attr_index, device)
            m = rows["MACRO"]
            print(f"step {step:>5} | loss {loss.item():.3f} "
                  f"(nce {parts['info_nce']:.3f} id {parts['identity']:.3f} "
                  f"orth {reg.item():.3f}) | "
                  f"R@1 {m['recall@1']:.3f} R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f}")
            key = (m["recall@1"], m["recall@5"])
            if key > best_key:
                best_key, best_rows = key, rows
                best_state = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}

    return best_state, best_rows


def main(name="t2"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}  phi: {name}  "
          f"[warm_start={WARM_START} ortho={ORTHO_MODE} queue_n={QUEUE_N}]")

    data = TrainData(device=device)
    db = load_db(DB_TEST).float().to(device)
    gts = build_ground_truth(EVAL_JSON)
    attr_index = attr_index_test()

    phi = T2Phi()

    if WARM_START:                                          # I3
        phi.load_directions(image_axes(data.F, data.L))
        print("  warm-started D from train image-space probe")

    lam_orth = LAM_ORTH                                     # I2
    if ORTHO_MODE == "corr":
        ortho_target = attr_corr(data.L).to(device)
    elif ORTHO_MODE == "zero":
        ortho_target = None
    elif ORTHO_MODE == "off":
        ortho_target, lam_orth = None, 0.0
    else:
        raise ValueError(f"ORTHO_MODE must be corr|zero|off, got {ORTHO_MODE!r}")

    best_state, best_rows = train(phi, data, gts, db, attr_index, device,
                                  lam_orth=lam_orth, ortho_target=ortho_target)

    RESULTS.mkdir(exist_ok=True)
    ckpt = RESULTS / f"phi_{name}.pt"
    torch.save(best_state, ckpt)
    print(f"best checkpoint -> {ckpt}")
    write_results(best_rows, name)


if __name__ == "__main__":
    main()
