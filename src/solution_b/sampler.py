"""Self-supervised sampler for training Φ (shared by T1 and T2).

Generates (v_ref, signed constraints, positive, hard-negatives) from the frozen
CelebA-train CLIP features + 40 binary attribute labels. No extra human labels.

Ground-truth rule mirrors the eval protocol (src/common/groundtruth.py), but is
regenerated on the fly on the TRAIN split:
  positive  = strictly satisfies the constraints AND Hamming <= 2 on the remaining
              attributes (same identity-preservation rule as the benchmark).
  hard-neg  = violates EXACTLY ONE queried constraint (the rest satisfied,
              Hamming <= 2) -> forces polarity (CLAY limit P1).

Sign strategy = FLIP-REF: constraints flip the reference's actual attributes, so
the edit is always non-trivial and positives are guaranteed to exist.
"""
import torch

from src.common.paths import DB_TRAIN, ATTRS_TRAIN

N_ATTR = 40


class TrainData:
    """Holds the frozen train features + labels and draws training batches."""

    def __init__(self, feat_path=DB_TRAIN, attr_path=ATTRS_TRAIN, device='cpu',
                 cmax=3, k_hardneg=4, max_ham=2):
        self.F = torch.load(feat_path).float().to(device)        # [N, 512] L2-norm
        self.L = torch.load(attr_path).bool().to(device)         # [N, 40] {0,1}
        self.N, self.n_attr = self.L.shape
        self.device = device
        self.cmax = cmax
        self.k = k_hardneg
        self.max_ham = max_ham

    def _one(self, i):
        """Build one training example anchored at index i, or None if no positive."""
        a = self.L[i]                                            # [40] bool
        n_cond = int(torch.randint(1, self.cmax + 1, (1,)))
        queried = torch.randperm(self.n_attr, device=self.device)[:n_cond]
        req = ~a[queried]                                        # FLIP-REF target on queried

        qmask = torch.zeros(self.n_attr, dtype=torch.bool, device=self.device)
        qmask[queried] = True
        rest = ~qmask
        ham_ok = (self.L[:, rest] != a[rest]).sum(1) <= self.max_ham   # [N]

        # positive: matches the flipped target on all queried + Hamming<=2 on rest
        pos_mask = (self.L[:, queried] == req).all(1) & ham_ok
        pos_mask[i] = False
        pos_cand = pos_mask.nonzero(as_tuple=False).squeeze(1)
        if pos_cand.numel() == 0:
            return None
        pos_idx = int(pos_cand[torch.randint(len(pos_cand), (1,))])

        # hard negatives: violate EXACTLY ONE queried constraint
        hneg = []
        for p in range(n_cond):
            req_v = req.clone()
            req_v[p] = ~req_v[p]                                 # flip the p-th back -> violate it
            m = (self.L[:, queried] == req_v).all(1) & ham_ok
            m[i] = False
            hneg.append(m.nonzero(as_tuple=False).squeeze(1))
        hneg_cand = torch.cat(hneg) if hneg else torch.empty(0, dtype=torch.long)
        if hneg_cand.numel() > self.k:
            sel = torch.randperm(hneg_cand.numel(), device=self.device)[:self.k]
            hneg_cand = hneg_cand[sel]

        sign = torch.where(req, 1.0, -1.0)                       # +1 present / -1 absent
        return {'ref': i, 'queried': queried, 'sign': sign,
                'pos': pos_idx, 'hneg': hneg_cand}

    def sample_batch(self, B, max_tries=20):
        """Draw B valid anchors (resampling those without a positive)."""
        rows, tries = [], 0
        while len(rows) < B and tries < B * max_tries:
            i = int(torch.randint(self.N, (1,)))
            ex = self._one(i)
            tries += 1
            if ex is not None:
                rows.append(ex)
        if len(rows) < B:
            raise RuntimeError(f"only {len(rows)}/{B} anchors got a positive in {tries} tries")

        cmax, k = self.cmax, self.k
        cond_col = torch.zeros(B, cmax, dtype=torch.long, device=self.device)
        cond_sign = torch.zeros(B, cmax, device=self.device)
        cond_mask = torch.zeros(B, cmax, dtype=torch.bool, device=self.device)
        hneg_idx = torch.full((B, k), -1, dtype=torch.long, device=self.device)
        ref_idx = torch.tensor([r['ref'] for r in rows], device=self.device)
        pos_idx = torch.tensor([r['pos'] for r in rows], device=self.device)

        for b, r in enumerate(rows):
            c = r['queried'].numel()
            cond_col[b, :c] = r['queried']
            cond_sign[b, :c] = r['sign']
            cond_mask[b, :c] = True
            h = r['hneg'].numel()
            if h:
                hneg_idx[b, :h] = r['hneg']

        hneg_mask = hneg_idx >= 0
        hneg_safe = hneg_idx.clamp_min(0)                        # gather-safe (pads masked out)
        return {
            'v_ref': self.F[ref_idx],                            # [B, 512]
            'cond_col': cond_col, 'cond_sign': cond_sign, 'cond_mask': cond_mask,
            'pos_feat': self.F[pos_idx],                         # [B, 512]
            'hneg_feat': self.F[hneg_safe],                      # [B, K, 512]
            'hneg_mask': hneg_mask,                              # [B, K]
            'ref_idx': ref_idx, 'pos_idx': pos_idx, 'hneg_idx': hneg_idx,
        }


# --------------------------------------------------------------------------- #
# Smoke test: python -m src.solution_b.sampler
# --------------------------------------------------------------------------- #
def _smoke():
    from src.solution_b.phi import MLPPhi
    from src.solution_b.losses import total_loss

    data = TrainData(device='cpu')
    print(f"loaded F={tuple(data.F.shape)} L={tuple(data.L.shape)}")

    B = 8
    batch = data.sample_batch(B)
    for key in ('v_ref', 'cond_col', 'cond_sign', 'cond_mask', 'pos_feat',
                'hneg_feat', 'hneg_mask'):
        print(f"  {key:10s} {tuple(batch[key].shape)}")
    assert batch['cond_col'].shape[1] == data.cmax

    L = data.L
    # 2. positive correctness
    for b in range(B):
        ref, pos = int(batch['ref_idx'][b]), int(batch['pos_idx'][b])
        m = batch['cond_mask'][b]
        cols = batch['cond_col'][b][m]
        want = batch['cond_sign'][b][m] > 0                     # True=present required
        assert torch.equal(L[pos][cols], want), f"anchor {b}: positive violates constraint"
        qmask = torch.zeros(data.n_attr, dtype=torch.bool); qmask[cols] = True
        ham = (L[pos][~qmask] != L[ref][~qmask]).sum().item()
        assert ham <= data.max_ham, f"anchor {b}: positive Hamming {ham} > {data.max_ham}"
    print("  ok  positives satisfy constraints + Hamming<=2")

    # 3. hard-neg correctness: violates exactly one queried constraint
    n_hneg = 0
    for b in range(B):
        m = batch['cond_mask'][b]
        cols = batch['cond_col'][b][m]
        want = batch['cond_sign'][b][m] > 0
        qmask = torch.zeros(data.n_attr, dtype=torch.bool); qmask[cols] = True
        for j in range(batch['hneg_mask'].shape[1]):
            if not batch['hneg_mask'][b, j]:
                continue
            h = int(batch['hneg_idx'][b, j])
            viol = (L[h][cols] != want).sum().item()
            assert viol == 1, f"anchor {b} hneg {j}: violates {viol} constraints (want 1)"
            ham = (L[h][~qmask] != L[int(batch['ref_idx'][b])][~qmask]).sum().item()
            assert ham <= data.max_ham
            n_hneg += 1
    print(f"  ok  {n_hneg} hard-negs violate exactly one constraint + Hamming<=2")

    # 4. loss path + backward
    phi = MLPPhi()
    v_q = phi(batch['v_ref'], batch['cond_col'], batch['cond_sign'], batch['cond_mask'])
    loss, parts = total_loss(v_q, batch['v_ref'], batch['pos_feat'],
                             batch['hneg_feat'], batch['hneg_mask'])
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in phi.parameters() if p.grad is not None)
    assert torch.isfinite(loss) and gnorm > 0, "loss not finite or no gradient"
    print(f"  ok  loss={loss.item():.4f} (nce={parts['info_nce']:.4f} "
          f"id={parts['identity']:.4f}) grad_norm={gnorm:.3f}")

    # 5. positive-found rate over random anchors
    found = sum(data._one(int(torch.randint(data.N, (1,)))) is not None for _ in range(200))
    print(f"  positive-found rate: {found}/200")


if __name__ == "__main__":
    _smoke()
