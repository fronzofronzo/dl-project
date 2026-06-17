import sys
from pathlib import Path

import torch

from baselines import build_direction_axes, contrastive_query
from demo_single import K, parse_query, satisfies, render
from groundtruth import build_ground_truth
from load_data import load_data, wire_attributes
from retrieval import load_db, rank

ROOT = Path(__file__).resolve().parent.parent

def compute_dynamic_weights(v_ref, pos_names, neg_names, axes):
      """
      pos_names: list[str]  — attribute names to add
      neg_names: list[str]  — attribute names to remove
      axes: dict[str, Tensor]  — from build_direction_axes
      Returns: list of (weight, signed_direction) tuples
      """
      cos = torch.nn.CosineSimilarity(dim=0)
      result = []
      for name in pos_names:
          d = axes[name]
          w = max(0.0, 1.0 - cos(v_ref, d).item())   # absent → high weight
          result.append((w, d))
      for name in neg_names:
          d = axes[name]
          w = max(0.0, cos(v_ref, d).item())           # present → high weight
          result.append((w, -d))                        # negative: flip sign
      return result

def apply_edit(v_ref, weighted_dirs, alpha=1.0, cap=2.0):
    v = v_ref.float().clone()
    edit = sum(w * d for w, d in weighted_dirs)
    edit_norm = edit.norm()
    if edit_norm > cap:
        edit = edit * cap / edit_norm
    v = v + alpha * edit
    return v / v.norm().clamp_min(1e-12)

def dynamic_query(v_ref, pos_names, neg_names, axes, alpha=1.0, cap=2.0):
    contributions = compute_dynamic_weights(v_ref, pos_names, neg_names, axes)
    if not contributions:
        return v_ref / v_ref.norm().clamp_min(1e-12)
    v = v_ref.float().clone()
    edit = sum(w * d for w, d in contributions)
    edit_norm = edit.norm()
    if edit_norm > cap:
        edit = edit * (cap / edit_norm)
    v = v + alpha * edit
    return v / v.norm().clamp_min(1e-12)


def combined_query(v_ref, pos_names, neg_names, axes, alpha=1.0, cap=2.0):
    """Contrastive shift first, then dynamic weights conditioned on the shifted vector."""
    from baselines import contrastive_query
    v1 = contrastive_query(v_ref, pos_names, neg_names, axes, alpha=alpha)
    return dynamic_query(v1, pos_names, neg_names, axes, alpha=alpha, cap=cap)



def main():
    query  = sys.argv[1] if len(sys.argv) > 1 else "+Eyeglasses"
    ref_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    alpha  = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0

    pos, neg = parse_query(query)
    celeba = load_data(ROOT / "data", split="test", download=False)
    attr_dict = wire_attributes(celeba)
    pos_idx = [attr_dict[p] for p in pos]
    neg_idx = [attr_dict[n] for n in neg]
    male_idx = attr_dict["Male"]

    if ref_idx is None:
        gts = {g.query: g for g in build_ground_truth(ROOT / "data" / "celeba_evaluation.json")}
        if query not in gts:
            raise SystemExit(f"Query {query!r} non nel JSON; passa un ref_idx esplicito.")
        for s in gts[query].gt:
            if int(celeba[s][1][male_idx]) == 0:
                ref_idx = s
                break
        ref_idx = ref_idx if ref_idx is not None else next(iter(gts[query].gt))

    db   = load_db(ROOT / "data" / "clip_features_test.pt")
    axes = build_direction_axes(set(pos) | set(neg))

    v_ref      = db[ref_idx]
    no_edit    = rank(v_ref, db, exclude={ref_idx}, k=K)
    v_contr    = contrastive_query(v_ref, pos, neg, axes, alpha=alpha)
    contr_edit = rank(v_contr, db, exclude={ref_idx}, k=K)
    v_dyn      = dynamic_query(v_ref, pos, neg, axes, alpha=alpha)
    dyn_edit   = rank(v_dyn, db, exclude={ref_idx}, k=K)

    def hit_count(ids):
        return sum(satisfies(celeba[j][1], pos_idx, neg_idx) for j in ids)

    ref_ok = satisfies(celeba[ref_idx][1], pos_idx, neg_idx)
    print(f"query={query!r}  ref=#{ref_idx} (soddisfa gia'? {ref_ok})  alpha={alpha}  K={K}")
    print(f"  no-edit    : {hit_count(no_edit)}/{K}  {no_edit}")
    print(f"  contrastive: {hit_count(contr_edit)}/{K}  {contr_edit}")
    print(f"  dynamic-w  : {hit_count(dyn_edit)}/{K}  {dyn_edit}")

    contributions = compute_dynamic_weights(v_ref, pos, neg, axes)
    labels = [f"+{n}" for n in pos] + [f"-{n}" for n in neg]
    print("  dynamic weights:")
    for label, (w, _) in zip(labels, contributions):
        print(f"    {label}: w={w:.4f}")

    render(celeba, ref_idx, no_edit, dyn_edit, pos_idx, neg_idx, query, alpha, out_name="demo_dynamic_weights.png")


if __name__ == "__main__":
    main()
