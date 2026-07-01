"""Ensemble Φ: spherical average of T1 and T2hybrid query vectors.

For each query both adapters compute v_q independently, then:
    v_q_ens = normalize(v_q_t1 + v_q_t2hybrid)

The sum on the unit hypersphere is equivalent to taking the direction of the
angular bisector — it up-weights directions both models agree on and averages
out idiosyncratic errors.

Run from repo root:
  python -m src.solution_b.ensemble
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.paths import RESULTS, DB_TEST, EVAL_JSON
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.t1_attention import T1Phi
from src.solution_a.t2_directions import T2Phi
from src.solution_a.run_t2 import eval_phi, write_results, attr_index_test


class EnsemblePhi(nn.Module):
    """Spherical mean of two Φ adapters implementing the shared contract."""

    def __init__(self, phi_a, phi_b):
        super().__init__()
        self.phi_a = phi_a
        self.phi_b = phi_b

    def forward(self, v_ref, cond_col, cond_sign, cond_mask):
        v_a = self.phi_a(v_ref, cond_col, cond_sign, cond_mask)
        v_b = self.phi_b(v_ref, cond_col, cond_sign, cond_mask)
        return F.normalize(v_a + v_b, dim=1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    phi_t1 = T1Phi()
    phi_t1.load_state_dict(torch.load(RESULTS / "phi_t1.pt", map_location=device))
    phi_t1.to(device).eval()

    phi_t2 = T2Phi(hybrid=True)
    phi_t2.load_state_dict(torch.load(RESULTS / "phi_t2_hybrid.pt", map_location=device))
    phi_t2.to(device).eval()

    phi = EnsemblePhi(phi_t1, phi_t2).to(device)

    db   = load_db(DB_TEST).float().to(device)
    gts  = build_ground_truth(EVAL_JSON)
    attr_index = attr_index_test()

    rows = eval_phi(phi, db, gts, attr_index, device)
    m = rows["MACRO"]
    print(f"ensemble T1+T2hybrid: R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f}  R@10={m['recall@10']:.3f}")
    write_results(rows, "t1_t2_ensemble")
    print(f"results -> {RESULTS / 'solution_a_t1_t2_ensemble.md'}")


if __name__ == "__main__":
    main()
