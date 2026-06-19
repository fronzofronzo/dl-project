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
