"""Solution C — product-of-experts re-ranking (Bayesian posterior readout).

Probe guidance on the QUERY (λ) pushes v_q through CLIP's entangled attribute
space and never beats λ=0 at the best sweep cell. Same probes, other side of
the similarity: score every DB candidate by how well IT satisfies the signed
constraints, read from the frozen linear probes on the frozen DB features:

    score(x) = cos(v_q, x) + η · Σ_i w_i · logσ(s_i · (W[c_i]·x + b[c_i]))

Log-linear opinion pooling (PoE-inspired; Hinton 2002): the composed-query
similarity is the identity prior, each signed constraint contributes an
independent per-candidate log-likelihood. logσ saturates at 0 once a
constraint is satisfied and decays ~linearly in a violated logit — a SOFT VETO.
Antiparallel constraint pairs (`+Wearing_Lipstick, −Heavy_Makeup`) cancel when
summed as direction vectors on the query side; here each candidate is judged
per-attribute, so they cannot cancel.

Query-side legality: the per-image logits Z = DB @ Wᵀ + b are precomputed ONCE
offline (same regime as the frozen DB itself, and as Solution A's negative
re-ranking); per query the penalty is a signed sum of |C| ≤ 3 precomputed
columns. The DB is never re-encoded.

Tuning discipline: η and the reliability weighting are tuned on the TRAIN
pseudo-benchmark (src/solution_c/train_bench.py) with the same weighted metric
as the v3 checkpoint rule (0.5·R@1 + 0.5·R@5), then applied ONCE to test; the
test η-sweep is a sensitivity table for the report, not a selection.

Run from repo root:
    python -m src.solution_c.rerank --smoke
    python -m src.solution_c.rerank --ckpt results/phi_flow_flow_v3.pt \
        --steps 8 --horizon 1.0 --tune                 # train-side η choice
    python -m src.solution_c.rerank --ckpt results/phi_flow_flow_v3.pt \
        --steps 8 --horizon 1.0 --eta 0.1 --name flow_v3_poe [--eta-sweep]
    python -m src.solution_c.rerank --t2-ckpt results/phi_t2_hybrid.pt \
        --eta 0.1 --name t2_hybrid_poe
    python -m src.solution_c.rerank --no-phi --eta 0.1 --name poe_only
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.common.paths import EVAL_JSON, DB_TEST, DB_TRAIN, RESULTS
from src.common.groundtruth import build_ground_truth
from src.common.metrics import evaluate_all
from src.common.retrieval import load_db
from src.solution_b.run import query_conditions, attr_index_test
from src.solution_c.probes import PROBES_PT
from src.solution_c.run_flow import load_flow, write_results

KS = (1, 5, 10)
ETA_GRID = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


# --------------------------------------------------------------------------- #
# PoE scoring
# --------------------------------------------------------------------------- #
def precompute_logits(db, W, b):
    """Per-image probe logits Z = DB @ Wᵀ + b, computed once per DB. [N, 40]"""
    return db.float() @ W.t().float() + b.float()


def constraint_logprob(Z, cols, signs, weights=None):
    """Σ_i w_i · logσ(s_i · z_{c_i}) per DB image -> [N].

    cols [C] long, signs [C] ±1 float, weights optional [n_attr] per-attribute
    reliability (None = 1).
    """
    lp = F.logsigmoid(signs * Z[:, cols])                    # [N, C]
    if weights is not None:
        lp = lp * weights[cols]
    return lp.sum(dim=1)


@torch.no_grad()
def compose_queries(phi, db, gts, attr_index, device):
    """Run Φ once per unique query -> {query: (sources, v_q [B,512] unit)}.

    phi=None composes nothing (v_q = v_ref): the PoE-only ablation row.
    """
    if phi is not None:
        phi.eval()
    out = {}
    for qgt in gts:
        if not qgt.gt or qgt.query in out:                   # eval JSON repeats -Young
            continue
        cc, cs = query_conditions(qgt, attr_index, device)
        sources = list(qgt.gt)
        v_ref = db[sources]
        B, C = len(sources), cc.numel()
        if phi is None:
            v_q = v_ref.float()
        else:
            v_q = phi(v_ref,
                      cc.unsqueeze(0).expand(B, C),
                      cs.unsqueeze(0).expand(B, C),
                      torch.ones(B, C, dtype=torch.bool, device=device))
        out[qgt.query] = (sources, F.normalize(v_q.float(), dim=1))
    return out


@torch.no_grad()
def rank_poe(db, Z, gts, composed, attr_index, device, eta, weights=None,
             ks=KS, chunk=1024):
    """Rankings with the PoE penalty added to the cosine scores.

    η=0 reproduces the plain cosine ranking of src.common.retrieval.rank (the
    db is NOT re-normalized, exactly like rank()). Sources are chunked to bound
    the [N_db, chunk] score matrix; the source row is set to −inf (the eval
    protocol's exclude={s}). Recomputing for a new η costs only this function.
    """
    dbf = db.float()
    kmax = max(ks)
    rankings = {}
    for qgt in gts:
        if qgt.query not in composed or qgt.query in rankings:
            continue
        sources, v_q = composed[qgt.query]
        cc, cs = query_conditions(qgt, attr_index, device)
        p = constraint_logprob(Z, cc, cs, weights)           # [N]
        ranks_q = {}
        for lo in range(0, len(sources), chunk):
            vq = v_q[lo:lo + chunk]                          # [b, 512]
            scores = dbf @ vq.t()                            # [N, b]
            if eta != 0.0:
                scores = scores + eta * p.unsqueeze(1)
            for j, s in enumerate(sources[lo:lo + chunk]):
                scores[s, j] = float("-inf")
            top = scores.topk(kmax, dim=0).indices           # [kmax, b]
            for j, s in enumerate(sources[lo:lo + chunk]):
                ranks_q[s] = top[:, j].tolist()
        rankings[qgt.query] = ranks_q
    return rankings


@torch.no_grad()
def eval_phi_poe(phi, db, gts, attr_index, device, W, b, eta, weights=None,
                 ks=KS, chunk=1024):
    """eval_phi with the PoE penalty: compose -> rank(+η·logp) -> metrics rows."""
    Z = precompute_logits(db, W, b)
    composed = compose_queries(phi, db, gts, attr_index, device)
    rankings = rank_poe(db, Z, gts, composed, attr_index, device, eta, weights,
                        ks=ks, chunk=chunk)
    return evaluate_all(rankings, gts, ks=ks)


# --------------------------------------------------------------------------- #
# η selection (train split) and test sensitivity
# --------------------------------------------------------------------------- #
def _score(rows, w_r1=0.5):
    m = rows["MACRO"]
    return w_r1 * m["recall@1"] + (1 - w_r1) * m["recall@5"]


def _grid_lines(tag, db, Z, gts, composed, attr_index, device, weight_opts, chunk):
    """(md table lines, best (score, eta, wtag)) over ETA_GRID × weightings."""
    lines = ["| η | weighting | R@1 | R@5 | R@10 | 0.5·R@1+0.5·R@5 |",
             "|---|---|---|---|---|---|"]
    best = (-1.0, None, None)
    for wtag, weights in weight_opts:
        for eta in ETA_GRID:
            rows = evaluate_all(rank_poe(db, Z, gts, composed, attr_index,
                                         device, eta, weights, chunk=chunk), gts)
            m, s = rows["MACRO"], _score(rows)
            print(f"  [{tag}] η={eta:<6} w={wtag:<4} | R@1 {m['recall@1']:.3f} "
                  f"R@5 {m['recall@5']:.3f} R@10 {m['recall@10']:.3f} | score {s:.4f}")
            lines.append(f"| {eta} | {wtag} | {m['recall@1']:.3f} | {m['recall@5']:.3f} "
                         f"| {m['recall@10']:.3f} | {s:.4f} |")
            if s > best[0]:
                best = (s, eta, wtag)
    return lines, best


def tune_on_train(phi, name, W, b, acc_w, device, n_sources, seed, chunk):
    """Grid η × weighting on the TRAIN pseudo-benchmark; freeze the choice."""
    from src.solution_c.train_bench import build_train_gts
    db = load_db(DB_TRAIN).float().to(device)
    gts = build_train_gts(n_sources=n_sources, seed=seed, device=device)
    attr_index = attr_index_test()
    Z = precompute_logits(db, W, b)
    composed = compose_queries(phi, db, gts, attr_index, device)

    lines, best = _grid_lines("train", db, Z, gts, composed, attr_index, device,
                              [("none", None), ("acc", acc_w)], chunk)
    out = RESULTS / f"solution_c_poe_tune_{name}.md"
    header = [f"# Solution C — PoE η tuning on TRAIN pseudo-queries ({name})", "",
              f"train split, {n_sources} sources/query, seed {seed}; "
              f"objective 0.5·R@1+0.5·R@5 (same as v3 ckpt rule).", ""]
    footer = ["", f"**frozen choice: η={best[1]}, weighting={best[2]} "
                  f"(train score {best[0]:.4f})** — apply once to test."]
    out.write_text("\n".join(header + lines + footer) + "\n")
    print(f"tune table -> {out}\nfrozen: η={best[1]} weighting={best[2]}")
    return best


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_t2(path, device):
    from src.solution_a.t2_directions import T2Phi
    phi = T2Phi(hybrid=True)
    phi.load_state_dict(torch.load(path, map_location=device))
    return phi.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="FlowPhi checkpoint (train_flow format)")
    src.add_argument("--t2-ckpt", help="T2Phi hybrid state-dict checkpoint")
    src.add_argument("--no-phi", action="store_true",
                     help="no composition (v_q = v_ref): PoE-only ablation")
    ap.add_argument("--steps", type=int, default=8, help="flow Euler steps N")
    ap.add_argument("--horizon", type=float, default=1.0, help="flow horizon T")
    ap.add_argument("--probes", default=str(PROBES_PT))
    ap.add_argument("--eta", type=float, default=None,
                    help="PoE weight (train-tuned); required for a test eval")
    ap.add_argument("--weighting", choices=("none", "acc"), default="none")
    ap.add_argument("--tune", action="store_true",
                    help="grid η × weighting on the TRAIN pseudo-benchmark and exit")
    ap.add_argument("--eta-sweep", action="store_true",
                    help="also freeze a test η-sensitivity table (report only)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--n-sources", type=int, default=200, help="train-bench sources/query")
    ap.add_argument("--seed", type=int, default=0, help="train-bench seed")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.probes, map_location=device)
    W, b = blob["W"].to(device).float(), blob["b"].to(device).float()
    acc_w = (2.0 * blob["acc_val"].to(device).float() - 1.0).clamp(0.0, 1.0)

    if args.ckpt:
        phi = load_flow(args.ckpt, device)
        phi.n_steps, phi.horizon, phi.guidance = args.steps, args.horizon, 0.0
        tag, dial = Path(args.ckpt).stem, f"flow N={args.steps} T={args.horizon}"
    elif args.t2_ckpt:
        phi, tag, dial = _load_t2(args.t2_ckpt, device), Path(args.t2_ckpt).stem, "t2"
    else:
        phi, tag, dial = None, "ref_only", "no composition (v_q = v_ref)"
    name = args.name or f"{tag}_poe"

    if args.tune:
        tune_on_train(phi, name, W, b, acc_w, device,
                      args.n_sources, args.seed, args.chunk)
        return

    if args.eta is None:
        ap.error("--eta is required for a test eval (choose it with --tune first)")
    weights = acc_w if args.weighting == "acc" else None

    db = load_db(DB_TEST).float().to(device)
    gts = build_ground_truth(EVAL_JSON)
    attr_index = attr_index_test()
    Z = precompute_logits(db, W, b)
    composed = compose_queries(phi, db, gts, attr_index, device)

    rows = evaluate_all(rank_poe(db, Z, gts, composed, attr_index, device,
                                 args.eta, weights, chunk=args.chunk), gts)
    m = rows["MACRO"]
    print(f"{name}: η={args.eta} w={args.weighting} ({dial}) | "
          f"R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f}")
    write_results(rows, name,
                  extra_header=f"PoE re-rank: η={args.eta}, weighting={args.weighting}, "
                               f"composition: {dial} (train-tuned η, single test run)")

    if args.eta_sweep:
        lines, _ = _grid_lines("test", db, Z, gts, composed, attr_index, device,
                               [(args.weighting, weights)], args.chunk)
        out = RESULTS / f"solution_c_{name}_eta_sweep.md"
        out.write_text("\n".join(
            [f"# Solution C — PoE test η-sensitivity ({name})", "",
             f"composition: {dial}, weighting={args.weighting}. SENSITIVITY table "
             f"(η was frozen on train, not selected here).", ""] + lines) + "\n")
        print(f"η sensitivity -> {out}")


# --------------------------------------------------------------------------- #
# Smoke: synthetic, no files needed
# --------------------------------------------------------------------------- #
def _smoke():
    import torch.nn as nn
    from src.common.groundtruth import QueryGT
    from src.solution_b.run import eval_phi

    torch.manual_seed(0)
    N, n_attr, C = 60, 8, 2
    db = F.normalize(torch.randn(N, 512), dim=1)
    W, b = torch.randn(n_attr, 512), torch.randn(n_attr)
    attr_index = {f"A{i}": i for i in range(n_attr)}
    gts = [QueryGT("+A1, -A3", pos=["A1"], neg=["A3"],
                   gt={4: {7, 9, 11}, 20: {21, 22, 23, 24}}),
           QueryGT("+A2", pos=["A2"], gt={5: {30, 31}})]

    # 1. constraint_logprob == explicit loop (with and without weights)
    Z = precompute_logits(db, W, b)
    cols = torch.tensor([1, 3]); signs = torch.tensor([1.0, -1.0])
    wts = torch.rand(n_attr)
    for weights in (None, wts):
        p = constraint_logprob(Z, cols, signs, weights)
        for i in (0, 17):
            want = sum((1.0 if weights is None else float(weights[c]))
                       * float(F.logsigmoid(s * (db[i] @ W[c] + b[c])))
                       for c, s in zip(cols.tolist(), signs.tolist()))
            assert abs(float(p[i]) - want) < 1e-4
    print("  ok  constraint_logprob matches explicit loop (± weighting)")

    class IdentityPhi(nn.Module):
        def forward(self, v_ref, cond_col, cond_sign, cond_mask):
            return F.normalize(v_ref, dim=1)

    # 2. η=0 reproduces eval_phi rows exactly (same phi, same rankings)
    phi = IdentityPhi()
    rows_ref = eval_phi(phi, db, gts, attr_index, "cpu")
    rows_poe = eval_phi_poe(phi, db, gts, attr_index, "cpu", W, b, eta=0.0)
    assert rows_ref == rows_poe, "η=0 diverges from eval_phi"
    rows_nophi = eval_phi_poe(None, db, gts, attr_index, "cpu", W, b, eta=0.0)
    assert rows_nophi == rows_ref, "phi=None != identity phi at η=0"
    print("  ok  η=0 reproduces eval_phi rows (IdentityPhi and phi=None)")

    # 3. chunked == unchunked
    a = eval_phi_poe(phi, db, gts, attr_index, "cpu", W, b, eta=0.3, chunk=1)
    c = eval_phi_poe(phi, db, gts, attr_index, "cpu", W, b, eta=0.3, chunk=4096)
    assert a == c, "chunking changes results"
    print("  ok  chunked == unchunked")

    # 4. source exclusion honored at huge η (constant penalty can't rescue it)
    composed = compose_queries(phi, db, gts, attr_index, "cpu")
    rk = rank_poe(db, Z, gts, composed, attr_index, "cpu", eta=1e6)
    for q, per_src in rk.items():
        for s, ranked in per_src.items():
            assert s not in ranked
    print("  ok  source excluded from its own ranking")

    # 5. soft veto: with large η the top-1 is the best-satisfying candidate
    qgt = gts[1]
    cc, cs = query_conditions(qgt, attr_index, "cpu")
    p = constraint_logprob(Z, cc, cs)
    p_masked = p.clone(); p_masked[5] = -torch.inf          # source of query "+A2"
    top1 = rank_poe(db, Z, [qgt], composed, attr_index, "cpu", eta=1e6)[qgt.query][5][0]
    assert top1 == int(p_masked.argmax()), "huge η should rank by constraint logprob"
    print("  ok  η→∞ ranks by constraint satisfaction (soft veto)")

    print("smoke test passed.")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke()
    else:
        main()
