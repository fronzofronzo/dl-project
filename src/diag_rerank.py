"""Per-query effect of negative re-rank on NEGATED queries only."""
import torch
from pathlib import Path
from torchvision.datasets import CelebA
from baselines import build_direction_axes, contrastive_query
from groundtruth import build_ground_truth
from metrics import evaluate_all
from rerank import build_image_probes, negative_rerank
from retrieval import load_db

ROOT = Path(__file__).resolve().parent.parent
gts = build_ground_truth(ROOT/"data"/"celeba_evaluation.json")
names = sorted({n for q in gts for n in (*q.pos,*q.neg)})
axes = build_direction_axes(names)
db = load_db(ROOT/"data"/"clip_features_test.pt").float()
ds = CelebA(root="data", split="test", download=False)
ai = {n:i for i,n in enumerate(ds.attr_names) if n}
probes = build_image_probes(db, ds.attr.float(), ai, names)

neg_gts = [q for q in gts if q.neg]
print(f"negated queries: {len(neg_gts)}\n")

def run(lam, alpha=4.0):
    rpq={}
    for q in neg_gts:
        d=rpq.setdefault(q.query,{})
        for s in q.gt:
            v=contrastive_query(db[s],q.pos,q.neg,axes,alpha=alpha)
            d[s]=negative_rerank(v,db,q.neg,probes,exclude={s},k=10,pool=200,lam=lam)
    return evaluate_all(rpq,neg_gts,ks=(1,5,10))

r0=run(0.0); r2=run(2.0)
print(f"{'query':<42} {'R@5 lam0':>9} {'R@5 lam2':>9} {'delta':>7}")
for q in neg_gts:
    a,b=r0[q.query]['recall@5'], r2[q.query]['recall@5']
    print(f"{q.query:<42} {a:>9.3f} {b:>9.3f} {b-a:>+7.3f}")
print(f"{'MACRO(neg only)':<42} {r0['MACRO']['recall@5']:>9.3f} {r2['MACRO']['recall@5']:>9.3f} "
      f"{r2['MACRO']['recall@5']-r0['MACRO']['recall@5']:>+7.3f}")

# probe sanity: do GT targets really have LOW presence on negated attr vs pool?
print("\nprobe sanity (mean presence score, GT targets vs random DB):")
for q in neg_gts:
    for n in q.neg:
        gtset=set().union(*q.gt.values())
        gt_idx=torch.tensor(sorted(gtset))
        s_gt=(db[gt_idx]@probes[n]).mean().item()
        s_all=(db@probes[n]).mean().item()
        print(f"  {q.query:<40} neg={n:<14} GT={s_gt:+.3f} DBmean={s_all:+.3f}")
