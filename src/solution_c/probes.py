"""Solution C — frozen linear attribute probes for hybrid classifier guidance.

One linear layer (512 -> 40) trained with BCE on the FROZEN CLIP features of the
CelebA TRAIN split (no test leakage; the eval DB is the test split). CLIP is
untouched — the probes read cached features, so this is the same "train a small
head on frozen features" regime as Φ itself.

Used by FlowPhi as inference-time guidance: the integration adds
λ · ∇_v Σ_i log p(constraint_i | v_t) (closed form, see FlowPhi.guidance_grad).
This is classifier guidance (Dhariwal & Nichol, 2021) transplanted from
diffusion sampling to retrieval-query construction: the flow provides the
learned prior, the probes provide an explicit constraint-satisfaction signal.
Purely query-side; the frozen DB is never touched.

Output: results/probes_linear.pt  {"W": [40, 512], "b": [40], "acc_val": [40]}

Run from repo root:  python -m src.solution_c.probes          (train + save)
                     python -m src.solution_c.probes --smoke  (synthetic check)
"""
import argparse

import torch
import torch.nn as nn

from src.common.paths import DB_TRAIN, ATTRS_TRAIN, RESULTS

N_ATTR = 40
DIM = 512
PROBES_PT = RESULTS / "probes_linear.pt"


def train_probes(F_feat, L, device, iters=1500, lr=1e-2, val_frac=0.1, log=print):
    """Fit the 40 logistic probes full-batch; return (W, b, acc_val per attr).

    F_feat [N, 512] L2-normalized frozen features, L [N, 40] bool labels.
    Last val_frac of a fixed shuffle is held out for honest accuracy only
    (the saved probes come from the train part).
    """
    N = F_feat.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g).to(F_feat.device)   # index on the data's device
    n_val = max(1, int(N * val_frac))
    tr, va = perm[:-n_val], perm[-n_val:]
    Xtr, Ytr = F_feat[tr].to(device), L[tr].float().to(device)
    Xva, Yva = F_feat[va].to(device), L[va].float().to(device)

    probe = nn.Linear(DIM, N_ATTR).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for it in range(1, iters + 1):
        loss = bce(probe(Xtr), Ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == iters:
            with torch.no_grad():
                acc = ((probe(Xva) > 0) == (Yva > 0.5)).float().mean()
            log(f"  iter {it:>4} | bce {loss.item():.4f} | val acc (mean) {acc:.3f}")

    with torch.no_grad():
        acc_val = ((probe(Xva) > 0) == (Yva > 0.5)).float().mean(dim=0).cpu()  # [40]
    return probe.weight.detach().cpu(), probe.bias.detach().cpu(), acc_val


def load_probes(path=PROBES_PT, device="cpu"):
    """Load saved probes -> (W [40,512], b [40]) on `device`."""
    blob = torch.load(path, map_location=device)
    return blob["W"].to(device), blob["b"].to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    F_feat = torch.load(DB_TRAIN).float()
    L = torch.load(ATTRS_TRAIN).bool()
    print(f"train features {tuple(F_feat.shape)}, labels {tuple(L.shape)}")

    W, b, acc_val = train_probes(F_feat, L, device)

    RESULTS.mkdir(exist_ok=True)
    torch.save({"W": W, "b": b, "acc_val": acc_val}, PROBES_PT)
    worst = acc_val.argsort()[:5].tolist()
    print(f"saved -> {PROBES_PT}")
    print(f"val accuracy: mean {acc_val.mean():.3f} | min {acc_val.min():.3f} "
          f"| worst attr columns {worst}")


# --------------------------------------------------------------------------- #
# Smoke: synthetic linearly-separable data, no files needed
# --------------------------------------------------------------------------- #
def _smoke():
    torch.manual_seed(0)
    W_true = torch.randn(N_ATTR, DIM)
    X = torch.nn.functional.normalize(torch.randn(4000, DIM), dim=1)
    L = (X @ W_true.t()) > 0
    W, b, acc = train_probes(X, L, "cpu", iters=2000)
    assert W.shape == (N_ATTR, DIM) and b.shape == (N_ATTR,)
    assert acc.mean() > 0.9, f"probes failed to fit separable data (acc {acc.mean():.3f})"
    print(f"smoke test passed (val acc {acc.mean():.3f}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic sanity check")
    if ap.parse_args().smoke:
        _smoke()
    else:
        main()
