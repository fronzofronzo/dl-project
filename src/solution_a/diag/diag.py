"""Diagnostic: measure the real magnitudes that drive Solution A tangent.

Checks the three things that silently kill results:
  1. ‖z_ref‖ = log_map(mu, v_ref) angle distribution (how far images sit from mu)
  2. ‖d‖ ambient direction norm + ‖d_t‖ after tangent projection
  3. cos(v_ref, d) distribution per attribute -> drives dynamic weights w
"""
import torch
import torch.nn.functional as F

from src.solution_a.directions import build_direction_axes
from src.common.geometry import log_map, sphere_mean
from src.common.groundtruth import build_ground_truth
from src.common.retrieval import load_db
from pathlib import Path

from src.common.paths import PROJECT_ROOT as ROOT
db = load_db(ROOT / "data" / "clip_features_test.pt")
db = F.normalize(db.float(), dim=-1)
mu = sphere_mean(db)
gts = build_ground_truth(ROOT / "data" / "celeba_evaluation.json")
names = sorted({n for q in gts for n in (*q.pos, *q.neg)})
axes = build_direction_axes(names)

print(f"DB {tuple(db.shape)}  ‖mu‖={mu.norm():.4f}")

# 1. how far do images sit from the visual mean? (z_ref norm = geodesic angle)
sample = db[torch.randperm(len(db))[:2000]]
angles = torch.acos((sample @ mu).clamp(-1+1e-7, 1-1e-7))
print(f"\n[1] image->mu angle (=‖z_ref‖): mean={angles.mean():.3f} "
      f"min={angles.min():.3f} max={angles.max():.3f}  (radians)")

# 2 & 3. per-attribute direction norms and cos(image, d)
print(f"\n[2/3] per-attribute:")
print(f"{'attr':<16} {'‖d‖':>7} {'‖d_t‖':>7} {'cos(mu,d)':>10} "
      f"{'cos(img,d) mean':>16} {'min':>7} {'max':>7}")
for n in names:
    d = axes[n].float()
    d_t = d - mu * (mu * d).sum()
    cos_mu_d = (F.normalize(mu, dim=0) * F.normalize(d, dim=0)).sum()
    # cos(image, d) over a sample
    dn = F.normalize(d, dim=0)
    cos_img = sample @ dn
    print(f"{n:<16} {d.norm():>7.3f} {d_t.norm():>7.3f} {cos_mu_d:>10.3f} "
          f"{cos_img.mean():>16.4f} {cos_img.min():>7.3f} {cos_img.max():>7.3f}")

# 4. weight behavior: for positives w=1-cos, for negatives w=cos
print(f"\n[4] resulting dynamic weights (sample mean):")
for n in names:
    d = axes[n].float()
    dn = F.normalize(d, dim=0)
    cos_img = sample @ dn
    w_pos = (1.0 - cos_img).clamp_min(0).mean()
    w_neg = cos_img.clamp_min(0).mean()
    print(f"{n:<16} w_pos≈{w_pos:.3f}  w_neg≈{w_neg:.3f}")

# 5. ceiling check: what does NO edit (pure v_ref) retrieve vs edited?
from src.common.metrics import evaluate_all
from src.common.retrieval import rank

def eval_fn(query_fn, label):
    rpq = {}
    for qgt in gts:
        d = rpq.setdefault(qgt.query, {})
        for s in qgt.gt:
            v = query_fn(db[s], qgt.pos, qgt.neg)
            d[s] = rank(v, db, exclude={s}, k=10)
    rows = evaluate_all(rpq, gts, ks=(1,5,10))
    m = rows["MACRO"]
    print(f"{label:<28} R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} R@10={m['recall@10']:.3f}")

print("\n[5] ceiling / sanity:")
eval_fn(lambda v,p,n: v, "no edit (pure v_ref)")

# 6. fix candidates: precompute per-attr DB stats for cos(image, d)
print("\n[6] fix candidates (tangent variants):")
dn_cache = {n: F.normalize(axes[n].float(), dim=0) for n in names}
stats = {}
for n in names:
    c = db @ dn_cache[n]
    stats[n] = (c.mean().item(), c.std().item() + 1e-8)

from src.common.geometry import exp_map

def tangent(v_ref, pos, neg, alpha, weight):
    z = log_map(mu, v_ref)
    edit = torch.zeros_like(z)
    for n in pos:
        d_t = F.normalize(axes[n].float() - mu*(mu*axes[n].float()).sum(), dim=0)
        edit = edit + weight(v_ref, n, +1) * d_t
    for n in neg:
        d_t = F.normalize(axes[n].float() - mu*(mu*axes[n].float()).sum(), dim=0)
        edit = edit - weight(v_ref, n, -1) * d_t
    tot = z + alpha*edit
    if tot.norm() > 1.0: tot = tot * (1.0/tot.norm())
    return exp_map(mu, tot)

# variant A: constant weight 1
eval_fn(lambda v,p,n: tangent(v,p,n,0.8, lambda v,nm,s:1.0), "tangent w=1 const a=0.8")

# variant B: standardized z-score sigmoid weights
def w_std(v, nm, s):
    c = (v * dn_cache[nm]).sum().item()
    mu_s, sd_s = stats[nm]
    z = (c - mu_s)/sd_s
    return torch.sigmoid(torch.tensor(-z if s>0 else z)).item()
for a in (0.3,0.5,0.8,1.2):
    eval_fn(lambda v,p,n,a=a: tangent(v,p,n,a, w_std), f"tangent w=std a={a}")

# 7. tangent with NATIVE direction norms (no unit-normalize) — preserve magnitude signal
print("\n[7] tangent native-norm directions, w=1 const:")
def tangent_native(v_ref, pos, neg, alpha):
    z = log_map(mu, v_ref)
    edit = torch.zeros_like(z)
    for n in pos:
        d = axes[n].float(); edit = edit + (d - mu*(mu*d).sum())
    for n in neg:
        d = axes[n].float(); edit = edit - (d - mu*(mu*d).sum())
    tot = z + alpha*edit
    if tot.norm() > 1.0: tot = tot * (1.0/tot.norm())
    return exp_map(mu, tot)
for a in (1.0,2.0,3.0,4.0):
    eval_fn(lambda v,p,n,a=a: tangent_native(v,p,n,a), f"tangent native w=1 a={a}")

# 8. reference: ambient contrastive (the current champion)
print("\n[8] ambient contrastive reference:")
from src.solution_a.directions import contrastive_query
for a in (2.0,3.0,4.0):
    eval_fn(lambda v,p,n,a=a: contrastive_query(v,p,n,axes,alpha=a), f"ambient contrastive a={a}")

# 9. image-space presence probe for dynamic weights (validation of headline feature)
print("\n[9] image-space dynamic weights:")
from torchvision.datasets import CelebA
ds = CelebA(root=str(ROOT / "data"), split="test", download=False)
A = ds.attr.float()              # [N,40] in {0,1}
an = {nm:i for i,nm in enumerate(ds.attr_names) if nm}
# image-space direction per attr: mean(img with) - mean(img without)
dimg = {}
for n in names:
    col = A[:, an[n]]
    pos_m = db[col==1].mean(0); neg_m = db[col==0].mean(0)
    dimg[n] = F.normalize(pos_m - neg_m, dim=0)
# presence score stats over DB
istats = {n: ((db@dimg[n]).mean().item(), (db@dimg[n]).std().item()+1e-8) for n in names}

def w_img(v, nm, s):
    z = ((v*dimg[nm]).sum().item() - istats[nm][0]) / istats[nm][1]
    return torch.sigmoid(torch.tensor(-z if s>0 else z)).item()

# tangent native-norm directions for edit, image-space dynamic weight
def tangent_imgw(v_ref, pos, neg, alpha):
    z = log_map(mu, v_ref); edit = torch.zeros_like(z)
    for n in pos:
        d=axes[n].float(); edit=edit + w_img(v_ref,n,+1)*(d-mu*(mu*d).sum())
    for n in neg:
        d=axes[n].float(); edit=edit - w_img(v_ref,n,-1)*(d-mu*(mu*d).sum())
    tot=z+alpha*edit
    if tot.norm()>1.0: tot=tot*(1.0/tot.norm())
    return exp_map(mu, tot)
for a in (3.0,4.0,6.0):
    eval_fn(lambda v,p,n,a=a: tangent_imgw(v,p,n,a), f"tangent img-w a={a}")

# 10. modulation weights around 1 (preserve constant magnitude + add dynamics)
print("\n[10] modulation w=1+k*tanh, + higher-alpha ambient:")
def w_mod(v, nm, s, k):
    z = ((v*dimg[nm]).sum().item() - istats[nm][0]) / istats[nm][1]
    t = -z if s>0 else z
    import math
    return 1.0 + k*math.tanh(t)
def tangent_mod(v_ref, pos, neg, alpha, k):
    z=log_map(mu,v_ref); edit=torch.zeros_like(z)
    for n in pos:
        d=axes[n].float(); edit=edit+w_mod(v_ref,n,+1,k)*(d-mu*(mu*d).sum())
    for n in neg:
        d=axes[n].float(); edit=edit-w_mod(v_ref,n,-1,k)*(d-mu*(mu*d).sum())
    tot=z+alpha*edit
    if tot.norm()>1.0: tot=tot*(1.0/tot.norm())
    return exp_map(mu,tot)
for k in (0.3,0.5):
    eval_fn(lambda v,p,n,k=k: tangent_mod(v,p,n,4.0,k), f"tangent mod k={k} a=4")
for a in (5.0,6.0,8.0):
    eval_fn(lambda v,p,n,a=a: contrastive_query(v,p,n,axes,alpha=a), f"ambient a={a}")
