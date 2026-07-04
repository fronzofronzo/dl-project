"""Solution C — GatedPhi: |C|-routed mixture of a one-shot editor and the flow.

The per-query tables show complementary strengths: the one-shot T2-hybrid wins
on single-attribute queries, the flow wins on composed ones (its sequential
trajectory only has something to sequence when |C| ≥ 2, and the one-shot editor
IS the flow's N=1 special case). GatedPhi routes each query by its condition
count — a property of the query itself, known before retrieval, with ZERO
learned/tuned parameters:

    v_q = Φ_flow(v_ref, C)      if |C| >= thresh   (default 2)
          Φ_t2(v_ref, C)        otherwise

Honesty protocol: the complementarity was first noticed on test tables, so the
gate must be CONFIRMED blind on the train pseudo-benchmark (--train-evidence
freezes that table) and stated as such in the report. Implements the shared Φ
contract, so it drops unchanged into eval_phi and into the PoE re-ranker
(--eta: the full system row).

Run from repo root:
    python -m src.solution_c.gated --smoke
    python -m src.solution_c.gated --flow-ckpt results/phi_flow_flow_v3.pt \
        --steps 8 --horizon 1.0 --train-evidence      # blind gate confirmation
    python -m src.solution_c.gated --flow-ckpt results/phi_flow_flow_v3.pt \
        --steps 8 --horizon 1.0                       # test row "gated"
    python -m src.solution_c.gated --flow-ckpt results/phi_flow_flow_v3.pt \
        --steps 8 --horizon 1.0 --eta 0.1             # test row "gated_poe"
"""
import argparse

import torch
import torch.nn as nn

from src.common.paths import EVAL_JSON, DB_TEST, DB_TRAIN, RESULTS
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from src.solution_b.run import eval_phi, attr_index_test
from src.solution_c.probes import PROBES_PT
from src.solution_c.rerank import eval_phi_poe, _load_t2
from src.solution_c.run_flow import load_flow, write_results

KS = (1, 5, 10)


class GatedPhi(nn.Module):
    """Mixture-of-experts router on the condition count |C| = cond_mask.sum(1)."""

    def __init__(self, phi_single, phi_multi, thresh=2):
        super().__init__()
        self.phi_single = phi_single
        self.phi_multi = phi_multi
        self.thresh = thresh

    def forward(self, v_ref, cond_col, cond_sign, cond_mask):
        v_s = self.phi_single(v_ref, cond_col, cond_sign, cond_mask)
        v_m = self.phi_multi(v_ref, cond_col, cond_sign, cond_mask)
        use_multi = cond_mask.sum(dim=1) >= self.thresh          # [B]
        return torch.where(use_multi.unsqueeze(1), v_m, v_s)


# --------------------------------------------------------------------------- #
# blind gate confirmation on the train pseudo-benchmark
# --------------------------------------------------------------------------- #
def train_evidence(phi_t2, phi_flow, device, n_sources, seed, thresh):
    """Both experts on the TRAIN bench, grouped by |C| -> frozen evidence table."""
    from src.solution_c.train_bench import build_train_gts
    db = load_db(DB_TRAIN).float().to(device)
    gts = build_train_gts(n_sources=n_sources, seed=seed, device=device)
    attr_index = attr_index_test()

    rows_t2 = eval_phi(phi_t2, db, gts, attr_index, device)
    rows_fl = eval_phi(phi_flow, db, gts, attr_index, device)

    n_cond = {q.query: len(q.pos) + len(q.neg) for q in gts}
    lines = [f"# Solution C — gate evidence on TRAIN pseudo-queries (thresh={thresh})", "",
             f"train split, {n_sources} sources/query, seed {seed}. The |C| "
             "complementarity was first noticed on test tables; this table is the "
             "blind confirmation used to freeze the gate.", "",
             "| query | C | t2 R@1 | flow R@1 | t2 R@5 | flow R@5 |",
             "|---|---|---|---|---|---|"]
    for q in sorted(n_cond, key=n_cond.get):
        t2, fl = rows_t2[q], rows_fl[q]
        lines.append(f"| {q} | {n_cond[q]} | {t2['recall@1']:.3f} | {fl['recall@1']:.3f} "
                     f"| {t2['recall@5']:.3f} | {fl['recall@5']:.3f} |")

    lines += ["", "| group | expert | R@1 | R@5 |", "|---|---|---|---|"]
    verdict = []
    for gname, cond in (("|C| < %d" % thresh, lambda c: c < thresh),
                        ("|C| >= %d" % thresh, lambda c: c >= thresh)):
        qs = [q for q, c in n_cond.items() if cond(c)]
        for tag, rows in (("t2", rows_t2), ("flow", rows_fl)):
            r1 = sum(rows[q]["recall@1"] for q in qs) / len(qs)
            r5 = sum(rows[q]["recall@5"] for q in qs) / len(qs)
            lines.append(f"| {gname} | {tag} | {r1:.3f} | {r5:.3f} |")
            verdict.append((gname, tag, 0.5 * r1 + 0.5 * r5))

    single_ok = verdict[0][2] >= verdict[1][2]     # t2 >= flow on |C|<thresh
    multi_ok = verdict[3][2] >= verdict[2][2]      # flow >= t2 on |C|>=thresh
    msg = ("CONFIRMED" if single_ok and multi_ok else "NOT CONFIRMED") + \
          f" (0.5·R@1+0.5·R@5 — single: t2 {verdict[0][2]:.4f} vs flow {verdict[1][2]:.4f}; " \
          f"multi: t2 {verdict[2][2]:.4f} vs flow {verdict[3][2]:.4f})"
    lines += ["", f"**gate thresh={thresh}: {msg}**"]

    out = RESULTS / "solution_c_gate_train_evidence.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"gate evidence -> {out}\n{msg}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-ckpt", required=True, help="FlowPhi checkpoint (multi expert)")
    ap.add_argument("--t2-ckpt", default=str(RESULTS / "phi_t2_hybrid.pt"),
                    help="T2Phi hybrid checkpoint (single expert)")
    ap.add_argument("--steps", type=int, default=8, help="flow Euler steps N")
    ap.add_argument("--horizon", type=float, default=1.0, help="flow horizon T")
    ap.add_argument("--thresh", type=int, default=2, help="route to flow when |C| >= thresh")
    ap.add_argument("--eta", type=float, default=None,
                    help="also apply PoE re-ranking with this train-tuned η")
    ap.add_argument("--weighting", choices=("none", "acc"), default="none")
    ap.add_argument("--probes", default=str(PROBES_PT))
    ap.add_argument("--name", default=None)
    ap.add_argument("--train-evidence", action="store_true",
                    help="freeze the blind |C|-group table on the train bench and exit")
    ap.add_argument("--n-sources", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    phi_flow = load_flow(args.flow_ckpt, device)
    phi_flow.n_steps, phi_flow.horizon, phi_flow.guidance = args.steps, args.horizon, 0.0
    phi_t2 = _load_t2(args.t2_ckpt, device)

    if args.train_evidence:
        train_evidence(phi_t2, phi_flow, device, args.n_sources, args.seed, args.thresh)
        return

    phi = GatedPhi(phi_t2, phi_flow, thresh=args.thresh).to(device).eval()
    dial = (f"gate: |C|>={args.thresh} -> flow(N={args.steps}, T={args.horizon}), "
            f"else t2-hybrid")
    db = load_db(DB_TEST).float().to(device)
    gts = build_ground_truth(EVAL_JSON)
    attr_index = attr_index_test()

    if args.eta is None:
        name = args.name or "gated"
        rows = eval_phi(phi, db, gts, attr_index, device)
        header = dial
    else:
        name = args.name or "gated_poe"
        blob = torch.load(args.probes, map_location=device)
        W, b = blob["W"].to(device).float(), blob["b"].to(device).float()
        weights = ((2.0 * blob["acc_val"].to(device).float() - 1.0).clamp(0.0, 1.0)
                   if args.weighting == "acc" else None)
        rows = eval_phi_poe(phi, db, gts, attr_index, device, W, b, args.eta, weights)
        header = f"{dial}; PoE η={args.eta}, weighting={args.weighting}"

    m = rows["MACRO"]
    print(f"{name}: R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} "
          f"R@10={m['recall@10']:.3f}")
    write_results(rows, name, extra_header=header)


# --------------------------------------------------------------------------- #
# Smoke: synthetic, no files needed
# --------------------------------------------------------------------------- #
def _smoke():
    import torch.nn.functional as F

    torch.manual_seed(0)
    B, C, DIM = 6, 3, 512

    class OffsetPhi(nn.Module):
        """Deterministic contract-shaped stub: v_q = normalize(v_ref + offset)."""
        def __init__(self, offset):
            super().__init__()
            self.register_buffer("off", offset)

        def forward(self, v_ref, cond_col, cond_sign, cond_mask):
            return F.normalize(v_ref + self.off, dim=1)

    v_ref = F.normalize(torch.randn(B, DIM), dim=1)
    cond_col = torch.randint(0, 40, (B, C))
    cond_sign = torch.where(torch.rand(B, C) > 0.5, 1.0, -1.0)
    cond_mask = torch.zeros(B, C, dtype=torch.bool)
    n_cond = [1, 2, 3, 1, 3, 2]                       # per-row |C|
    for i, n in enumerate(n_cond):
        cond_mask[i, :n] = True

    a, bmod = OffsetPhi(torch.randn(DIM) * 0.1), OffsetPhi(torch.randn(DIM) * 0.1)

    # 1. same module in both slots == the module alone
    same = GatedPhi(a, a, thresh=2)(v_ref, cond_col, cond_sign, cond_mask)
    assert torch.allclose(same, a(v_ref, cond_col, cond_sign, cond_mask))
    print("  ok  same expert in both slots == expert alone")

    # 2. row-wise routing by |C|
    gated = GatedPhi(a, bmod, thresh=2)
    out = gated(v_ref, cond_col, cond_sign, cond_mask)
    va, vb = a(v_ref, cond_col, cond_sign, cond_mask), bmod(v_ref, cond_col, cond_sign, cond_mask)
    for i, n in enumerate(n_cond):
        want = vb[i] if n >= 2 else va[i]
        assert torch.allclose(out[i], want), f"row {i} (|C|={n}) routed wrong"
    assert torch.allclose(out.norm(dim=1), torch.ones(B), atol=1e-5)
    print("  ok  rows routed by |C| (thresh=2), unit-norm output")

    # 3. thresh honored
    out3 = GatedPhi(a, bmod, thresh=3)(v_ref, cond_col, cond_sign, cond_mask)
    for i, n in enumerate(n_cond):
        want = vb[i] if n >= 3 else va[i]
        assert torch.allclose(out3[i], want)
    print("  ok  thresh=3 routing")

    print("smoke test passed.")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke()
    else:
        main()
