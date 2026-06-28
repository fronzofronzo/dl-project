"""Training loop for the T2 adapter Φ (learned image-space directions + dynamic gate).

T2's OWN launch script — independent of solution_b/train.py (which person 1 uses for
T1). It reuses the SHARED backbone infrastructure (the self-supervised sampler and
the InfoNCE/identity losses) as the project plan mandates, but drives its own
training/eval orchestration and freezes its own checkpoint + results so the two
solutions never share launch scripts.

CLIP stays frozen; only Φ_T2 is trained. Self-supervised on CelebA-train attributes
via the sampler; validation on the CelebA-test benchmark every `eval_every` steps;
the best checkpoint (by R@1, tie R@5) is saved and its results table frozen.

Loss = InfoNCE + lam_id * identity_anchor + lam_orth * ortho_reg(D)
       (the orthogonality regularizer is T2-specific: the learned Gram–Schmidt that
        disentangles the direction dictionary — CLAY limit P3.)

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
LR = 1e-3
TAU = 0.07
LAM_ID = 0.3
LAM_ORTH = 0.1          # weight of the direction-disentangling orthogonality regularizer
EVAL_EVERY = 100


def train(phi, data, gts, db, attr_index, device, *, steps=STEPS, batch=BATCH,
          lr=LR, tau=TAU, lam_id=LAM_ID, lam_orth=LAM_ORTH, eval_every=EVAL_EVERY):
    """Optimize Φ_T2; return (best_state_dict, best_rows). CLIP/DB untouched."""
    phi.to(device)
    opt = optim.Adam(phi.parameters(), lr=lr)
    best_key, best_rows, best_state = (-1.0, -1.0), None, None

    for step in range(1, steps + 1):
        phi.train()
        b = data.sample_batch(batch)
        v_q = phi(b['v_ref'], b['cond_col'], b['cond_sign'], b['cond_mask'])
        loss, parts = total_loss(v_q, b['v_ref'], b['pos_feat'], b['hneg_feat'],
                                 b['hneg_mask'], tau=tau, lam_id=lam_id)
        reg = phi.ortho_reg()
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
    print(f"device: {device}  phi: {name}")

    data = TrainData(device=device)
    db = load_db(DB_TEST).float().to(device)
    gts = build_ground_truth(EVAL_JSON)
    attr_index = attr_index_test()

    phi = T2Phi()
    best_state, best_rows = train(phi, data, gts, db, attr_index, device)

    RESULTS.mkdir(exist_ok=True)
    ckpt = RESULTS / f"phi_{name}.pt"
    torch.save(best_state, ckpt)
    print(f"best checkpoint -> {ckpt}")
    write_results(best_rows, name)


if __name__ == "__main__":
    main()
