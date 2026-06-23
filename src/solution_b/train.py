"""Shared training loop for Φ (CLIP frozen, query-side adapter).

Φ is swappable: MLPPhi (baseline, here), later T1 (cross-attention) and T2
(learned directions) implement the same contract and drop straight in. Training
is self-supervised on CelebA-train attributes via the sampler; validation runs on
the CelebA-test benchmark every `eval_every` steps; the best checkpoint (by R@1,
tie R@5) is saved and its results table frozen.

Run from repo root:  python -m src.solution_b.train
"""
import torch
from torch import optim
from torchvision.datasets import CelebA

from src.common.paths import PROJECT_ROOT as ROOT, EVAL_JSON, DB_TEST
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.sampler import TrainData
from src.solution_b.losses import total_loss
from src.solution_b.phi import MLPPhi
from src.solution_b.run import eval_phi, write_results, attr_index_test, RESULTS

# defaults (tunable; sweep later)
STEPS = 3000
BATCH = 256
LR = 1e-3
TAU = 0.07
LAM_ID = 0.1
EVAL_EVERY = 500


def train(phi, data, gts, db, attr_index, device, *, steps=STEPS, batch=BATCH,
          lr=LR, tau=TAU, lam_id=LAM_ID, eval_every=EVAL_EVERY):
    """Optimize Φ; return (best_state_dict, best_rows). CLIP/DB untouched."""
    phi.to(device)
    opt = optim.Adam(phi.parameters(), lr=lr)
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

        if step % eval_every == 0 or step == steps:
            rows = eval_phi(phi, db, gts, attr_index, device)
            m = rows["MACRO"]
            print(f"step {step:>5} | loss {loss.item():.3f} "
                  f"(nce {parts['info_nce']:.3f} id {parts['identity']:.3f}) | "
                  f"R@1 {m['recall@1']:.3f} R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f}")
            key = (m["recall@1"], m["recall@5"])
            if key > best_key:
                best_key, best_rows = key, rows
                best_state = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}

    return best_state, best_rows


def main(name="mlp"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    data = TrainData(device=device)
    db = load_db(DB_TEST).float().to(device)
    gts = build_ground_truth(EVAL_JSON)
    attr_index = attr_index_test()

    phi = MLPPhi()
    best_state, best_rows = train(phi, data, gts, db, attr_index, device)

    RESULTS.mkdir(exist_ok=True)
    ckpt = RESULTS / f"phi_{name}.pt"
    torch.save(best_state, ckpt)
    print(f"best checkpoint -> {ckpt}")
    write_results(best_rows, name)


if __name__ == "__main__":
    main()
