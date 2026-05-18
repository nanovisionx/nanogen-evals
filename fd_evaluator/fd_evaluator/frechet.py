"""Frechet distance. Adopted from FD-loss: https://github.com/Jiawei-Yang/FD-loss"""

from __future__ import annotations

import numpy as np
import torch
from scipy import linalg


class FrechetAccumulator:
    """Stream-friendly. Update with feature batches, finalize to (mu, sigma)."""

    def __init__(self, feat_dim: int):
        self.feat_dim = feat_dim
        self._sum = np.zeros(feat_dim, dtype=np.float64)
        self._outer = np.zeros((feat_dim, feat_dim), dtype=np.float64)
        self._n = 0

    def update(self, feats: torch.Tensor | np.ndarray) -> None:
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().to("cpu", dtype=torch.float64).numpy()
        feats = np.asarray(feats, dtype=np.float64)
        if feats.ndim != 2 or feats.shape[1] != self.feat_dim:
            raise ValueError(
                f"expected [B, {self.feat_dim}], got {feats.shape}"
            )
        self._sum += feats.sum(axis=0)
        self._outer += feats.T @ feats
        self._n += feats.shape[0]

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self._n < 2:
            raise RuntimeError(f"need >=2 samples to compute mu/sigma; got {self._n}")
        mu = self._sum / self._n
        sigma = (self._outer - self._n * np.outer(mu, mu)) / (self._n - 1)
        return mu, sigma

    @property
    def n(self) -> int:
        return self._n


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = float(np.max(np.abs(covmean.imag)))
            raise RuntimeError(f"imaginary component {m}")
        covmean = covmean.real

    tr_covmean = float(np.trace(covmean))
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
