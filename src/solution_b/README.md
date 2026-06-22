# Solution B — Training-based fusion Φ (Part B)

CLIP ViT-B/32 **frozen**; we train only the small fusion module Φ, query-side,
DB frozen. Two complementary architectures (see `../../next_steps.md` for the
full plan and the work split):

- **T1 — Conditional cross-attention adapter.** Signed condition tokens +
  learned polarity embedding, cross-attention over the condition set, FiLM-gated
  residual on `v_ref`. Owner: person 1.
- **T2 — Learned image-space directions + dynamic gate.** Learned attribute
  direction dictionary `D ∈ ℝ^{40×512}` (sign-aware), MLP weight head conditioned
  on `v_ref`, orthogonality regularizer. The trained twin of Solution A.
  Owner: person 2.

Shared spine to build first (in `../common/`): train-split CLIP features,
self-supervised sampler, InfoNCE loss. Eval reuses `../common/metrics.py` etc.

Planned files:
```
solution_b/
├── sampler.py     # self-supervised (ref, ±constraints, positives, hard-negs) from CelebA train
├── losses.py      # InfoNCE + identity anchor + (T2) orthogonality reg
├── t1_attention.py
├── t2_directions.py
├── train.py       # shared training loop
└── run.py         # eval trained Φ vs baselines
```
