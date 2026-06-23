"""Spherical geometry primitives for CLIP embedding space (S^{d-1}).

All vectors assumed L2-normalized before calling these functions.
"""
import torch
import torch.nn.functional as F


def sphere_mean(features: torch.Tensor) -> torch.Tensor:
    """Normalized mean direction of a set of unit vectors — Fréchet mean approx."""
    return F.normalize(features.mean(dim=0), dim=0)


def log_map(mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Project x ∈ S^{d-1} to tangent plane T_μ via the Riemannian Log map.

    Log_μ(x) = (x − μ·(x·μ)) · θ/sin(θ),   θ = arccos(x·μ)
    """
    mu = mu.float()
    x = x.float()
    cos_t = (mu * x).sum().clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_t)
    if theta.abs() < 1e-7:
        return torch.zeros_like(x)
    x_perp = x - mu * cos_t
    return x_perp * (theta / theta.sin())


def exp_map(mu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Map tangent vector v ∈ T_μ back onto S^{d-1} via the Riemannian Exp map.

    Exp_μ(v) = cos(‖v‖)·μ + sin(‖v‖)·(v/‖v‖)
    """
    mu = mu.float()
    v = v.float()
    r = v.norm()
    if r < 1e-7:
        return mu.clone()
    return mu * r.cos() + (v / r) * r.sin()


def gram_schmidt(directions):
    """Sequentially orthogonalize a list of vectors (fixes CLAY's P3 interaction bug).

    Each direction keeps only its UNIQUE component, with the part already covered by
    the previously processed directions projected out:

        uₖ = dₖ − Σ_{j<k} (dₖ·uⱼ / uⱼ·uⱼ) · uⱼ

    Two active directions sharing an axis (e.g. +Blond / −Red_Hair both lie on the
    hair-colour axis) no longer double-count or silently cancel after summation.

    ORDER MATTERS: the first direction is kept whole; later ones lose the shared
    part. Pass the most important constraint first. `directions` are expected with
    the sign already applied (sign only scales the projection, so orientation is
    preserved). Near-degenerate residuals (a direction fully spanned by earlier
    ones) collapse to ~0 and contribute nothing — the intended behaviour.

    Returns a list of orthogonal vectors, same length and order as the input.
    """
    basis = []
    out = []
    for d in directions:
        u = d.float().clone()
        for q in basis:
            denom = (q * q).sum().clamp_min(1e-12)
            u = u - ((d * q).sum() / denom) * q
        out.append(u)
        if u.norm() > 1e-7:
            basis.append(u)
    return out


def gram_schmidt_conflict(directions, signs):
    """Sign-conflict-aware Gram–Schmidt: orthogonalize a direction ONLY against the
    previously kept directions of OPPOSITE sign.

    Rationale (measured, see results/solution_a_sweep): plain GS helps opposite-sign
    conflicts (e.g. −Smiling vs +Eyeglasses+Hat: +0.013 R@5) but HURTS same-sign
    correlated pairs (+Eyeglasses & +Smiling: −0.023) because it strips the shared
    component that there carries real signal. So we project out the shared axis only
    when two constraints PULL APART (s_i ≠ s_j); same-sign directions keep their
    overlap (raw sum, the better-performing behaviour).

    `directions` are signed vectors (sign already baked in); `signs` ∈ {+1,−1} the
    matching polarity labels, same order. Returns vectors in input order.
    """
    basis = []                      # list of (orthogonal_vec, sign)
    out = []
    for d, s in zip(directions, signs):
        u = d.float().clone()
        for q, sq in basis:
            if sq == s:
                continue            # same sign -> keep shared component
            denom = (q * q).sum().clamp_min(1e-12)
            u = u - ((d * q).sum() / denom) * q
        out.append(u)
        if u.norm() > 1e-7:
            basis.append((u, s))
    return out
