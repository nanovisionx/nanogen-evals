"""MIND (Monge Inception Distance). Adopted from torch-fidelity:
https://github.com/toshas/torch-fidelity. Paper: https://arxiv.org/abs/2605.06797"""

from __future__ import annotations

import numpy as np
import torch


def mind_from_features(
    gen_features: torch.Tensor | np.ndarray,
    ref_features: torch.Tensor | np.ndarray,
    num_projections: int = 1000,
    rng_seed: int = 2020,
) -> float:
    X = _as_float_tensor(gen_features)
    Y = _as_float_tensor(ref_features)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(f"shape mismatch: X={tuple(X.shape)} Y={tuple(Y.shape)}")

    d = X.shape[1]
    alpha = 3 * d
    g = torch.Generator(device="cpu").manual_seed(rng_seed)

    n_gen, n_ref = X.shape[0], Y.shape[0]
    if n_gen != n_ref:
        n = min(n_gen, n_ref)
        if n_gen > n:
            idx = torch.randperm(n_gen, generator=g)[:n]
            X = X[idx]
        if n_ref > n:
            idx = torch.randperm(n_ref, generator=g)[:n]
            Y = Y[idx]

    U = torch.randn(num_projections, d, generator=g, dtype=X.dtype)
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
    P1, _ = (X @ U.T).sort(dim=0)
    P2, _ = (Y @ U.T).sort(dim=0)
    sliced_w2_sq = ((P1 - P2) ** 2).mean(dim=0)
    return float(alpha * sliced_w2_sq.mean().item())


def _as_float_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", dtype=torch.float32)
    return torch.from_numpy(np.asarray(x, dtype=np.float32))
