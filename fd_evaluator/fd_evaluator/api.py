"""Orchestrator: compute FID / FDR6 / MIND on a generated image set.

Metric names:
    fid               FID (inception-v3-compat) vs imagenet_256_fid_stats
    inception_score   Inception Score
    fdr6              bundle: FD / norm_const across 6 extractors (FD-loss)
    mind6             bundle: MIND across 6 extractors (torch-fidelity)
    fdr_<ext>         single-extractor FDR
    mind_<ext>        single-extractor MIND
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .extractors import (
    Extractor,
    ExtractorSpec,
    InceptionExtractor,
    build_extractor,
)
from .frechet import FrechetAccumulator, frechet_distance
from .mind import mind_from_features
from .stats import load_moments


def list_extractors() -> list[str]:
    return list(_load_catalogue()["extractors"].keys())


def list_bundles() -> list[str]:
    return list(_load_catalogue()["bundles"].keys())


def compute_metrics(
    images: np.ndarray | str,
    metrics: Sequence[str],
    *,
    fid_reference: str = "imagenet_256_fid_stats",
    reference_images: Optional[np.ndarray | str] = None,
    device: str = "cuda",
    batch_size: int = 64,
    num_projections: int = 1000,
    rng_seed: int = 2020,
    feature_cache_dir: Optional[str] = None,
    feature_cache_key: Optional[str] = None,
    reference_feature_cache_key: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, float]:
    cat = _load_catalogue()
    requested = _expand_metrics(metrics, cat)
    needed = _extractors_for(requested, cat)

    gen_arr = _load_uint8_nhwc(images)
    ref_arr = (
        _load_uint8_nhwc(reference_images) if reference_images is not None else None
    )

    if any(m.startswith("mind") for m in requested) and ref_arr is None:
        raise ValueError(
            "MIND metrics require reference_images (path to a uint8 NHWC npz, "
            "e.g. imagenet-256-val.npz). Got reference_images=None."
        )

    results: dict[str, float] = {}
    for ext_name in needed:
        spec = _spec_for(ext_name, cat)
        if verbose:
            print(f"[fd_evaluator] extractor {ext_name} ({spec.model})")

        need_features = any(
            m == f"mind_{ext_name}" or m in {"mind6"} for m in requested
        )
        need_frechet = any(
            m == f"fdr_{ext_name}"
            or m == f"fd_{ext_name}"
            or m == "fdr6"
            or (ext_name == "inception" and m in {"fid", "inception_score"})
            for m in requested
        )

        extractor = build_extractor(spec).to(device).eval()

        gen_feats, gen_mu, gen_sigma, gen_logits = _extract_with_cache(
            extractor=extractor,
            spec=spec,
            arr=gen_arr,
            batch_size=batch_size,
            device=device,
            keep_features=need_features,
            keep_logits=(ext_name == "inception" and "inception_score" in requested),
            cache_dir=feature_cache_dir,
            cache_key=(
                f"{feature_cache_key}-{ext_name}" if feature_cache_key else None
            ),
            verbose=verbose,
        )

        if ext_name == "inception" and "inception_score" in requested:
            results["inception_score"] = float(_inception_score(gen_logits))

        if need_frechet:
            need_fdr_branch = (
                f"fdr_{ext_name}" in requested
                or f"fd_{ext_name}" in requested
                or "fdr6" in requested
            )
            if need_fdr_branch:
                mu_ref, sig_ref = load_moments(spec.default_reference)
                fd = frechet_distance(gen_mu, gen_sigma, mu_ref, sig_ref)
                if spec.norm_const > 0:
                    results[f"fdr_{ext_name}"] = fd / spec.norm_const

            if ext_name == "inception" and "fid" in requested:
                mu_g, sig_g = load_moments(fid_reference)
                results["fid"] = frechet_distance(gen_mu, gen_sigma, mu_g, sig_g)

        if need_features:
            ref_feats = _extract_reference_features(
                extractor=extractor,
                spec=spec,
                ref_arr=ref_arr,
                batch_size=batch_size,
                device=device,
                cache_dir=feature_cache_dir,
                cache_key=(
                    f"{reference_feature_cache_key}-{ext_name}"
                    if reference_feature_cache_key
                    else None
                ),
                verbose=verbose,
            )
            mind = mind_from_features(
                gen_feats, ref_feats, num_projections=num_projections, rng_seed=rng_seed
            )
            results[f"mind_{ext_name}"] = mind

        del extractor
        torch.cuda.empty_cache()

    if "fdr6" in requested:
        bundle = cat["bundles"]["fdr6"]
        fdrs = [
            results[f"fdr_{x}"]
            for x in bundle
            if f"fdr_{x}" in results and math.isfinite(results[f"fdr_{x}"])
        ]
        if len(fdrs) == len(bundle):
            results["fdr6"] = float(np.mean(fdrs))

    return results


def _load_catalogue() -> dict:
    path = Path(__file__).with_name("catalogue.yaml")
    with path.open() as f:
        return yaml.safe_load(f)


def _spec_for(name: str, cat: dict) -> ExtractorSpec:
    cfg = cat["extractors"].get(name)
    if cfg is None:
        raise KeyError(f"unknown extractor {name!r}; known: {list(cat['extractors'])}")
    return ExtractorSpec(name=name, **cfg)


def _expand_metrics(metrics: Sequence[str], cat: dict) -> set[str]:
    out: set[str] = set()
    for m in metrics:
        if m in cat["bundles"]:
            out.add(m)
            for ext in cat["bundles"][m]:
                if m == "fdr6":
                    out.add(f"fdr_{ext}")
                elif m == "mind6":
                    out.add(f"mind_{ext}")
        else:
            out.add(m)
    return out


def _extractors_for(metrics: set[str], cat: dict) -> list[str]:
    needed: list[str] = []
    catalogue_order = list(cat["extractors"].keys())
    for ext in catalogue_order:
        if any(
            m == f"fdr_{ext}"
            or m == f"fd_{ext}"
            or m == f"mind_{ext}"
            or (ext == "inception" and m in {"fid", "inception_score"})
            for m in metrics
        ):
            needed.append(ext)
    return needed


def _load_uint8_nhwc(src: np.ndarray | str) -> np.ndarray:
    if isinstance(src, np.ndarray):
        arr = src
    else:
        data = np.load(src, mmap_mode="r")
        key = "arr_0" if "arr_0" in data else list(data.keys())[0]
        arr = data[key]
    if arr.dtype != np.uint8 or arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"expected uint8 NHWC RGB, got dtype={arr.dtype} shape={arr.shape}"
        )
    return arr


def _extract_with_cache(
    *,
    extractor: Extractor,
    spec: ExtractorSpec,
    arr: np.ndarray,
    batch_size: int,
    device: str,
    keep_features: bool,
    keep_logits: bool,
    cache_dir: Optional[str],
    cache_key: Optional[str],
    verbose: bool,
) -> tuple[Optional[torch.Tensor], np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    cached_feats = _maybe_load_feature_cache(cache_dir, cache_key, "features")
    cached_logits = _maybe_load_feature_cache(cache_dir, cache_key, "logits")

    if cached_feats is not None and (not keep_logits or cached_logits is not None):
        if verbose:
            print(f"[fd_evaluator]  using cached features {cache_key!r}")
        feats = cached_feats
        logits = cached_logits if keep_logits else None
    else:
        feats_list: list[torch.Tensor] = []
        logits_list: list[torch.Tensor] = []
        iterator = _batched(arr, batch_size, total=True)
        if verbose:
            iterator = tqdm(iterator, desc=f"  extract[{spec.name}]")
        for batch in iterator:
            x = (
                torch.from_numpy(np.ascontiguousarray(batch))
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(device, non_blocking=True)
            )
            if keep_logits and isinstance(extractor, InceptionExtractor):
                f, lg = extractor.forward_with_logits(x)
                logits_list.append(lg.detach().cpu())
            else:
                f = extractor(x)
            feats_list.append(f.detach().cpu())
        feats = torch.cat(feats_list, dim=0)
        logits = torch.cat(logits_list, dim=0) if logits_list else None
        _maybe_save_feature_cache(cache_dir, cache_key, "features", feats)
        if logits is not None:
            _maybe_save_feature_cache(cache_dir, cache_key, "logits", logits)

    acc = FrechetAccumulator(spec.feat_dim)
    acc.update(feats)
    mu, sigma = acc.finalize()
    return (feats if keep_features else None, mu, sigma, logits)


def _extract_reference_features(
    *,
    extractor: Extractor,
    spec: ExtractorSpec,
    ref_arr: np.ndarray,
    batch_size: int,
    device: str,
    cache_dir: Optional[str],
    cache_key: Optional[str],
    verbose: bool,
) -> torch.Tensor:
    cached = _maybe_load_feature_cache(cache_dir, cache_key, "features")
    if cached is not None:
        if verbose:
            print(f"[fd_evaluator]  using cached ref features {cache_key!r}")
        return cached

    feats_list: list[torch.Tensor] = []
    iterator = _batched(ref_arr, batch_size, total=True)
    if verbose:
        iterator = tqdm(iterator, desc=f"  extract-ref[{spec.name}]")
    for batch in iterator:
        x = (
            torch.from_numpy(np.ascontiguousarray(batch))
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(device, non_blocking=True)
        )
        feats_list.append(extractor(x).detach().cpu())
    feats = torch.cat(feats_list, dim=0)
    _maybe_save_feature_cache(cache_dir, cache_key, "features", feats)
    return feats


def _batched(arr: np.ndarray, bs: int, total: bool = False) -> Iterable[np.ndarray]:
    n = arr.shape[0]
    for i in range(0, n, bs):
        yield arr[i : i + bs]


def _maybe_load_feature_cache(
    cache_dir: Optional[str], cache_key: Optional[str], kind: str
) -> Optional[torch.Tensor]:
    if not (cache_dir and cache_key):
        return None
    p = Path(cache_dir) / f"{cache_key}.{kind}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu")


def _maybe_save_feature_cache(
    cache_dir: Optional[str], cache_key: Optional[str], kind: str, tensor: torch.Tensor
) -> None:
    if not (cache_dir and cache_key):
        return
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, p / f"{cache_key}.{kind}.pt")


def _inception_score(logits: torch.Tensor, splits: int = 10) -> float:
    """Standard Inception Score from inception-v3-compat unbiased logits.

    Matches the in-tree calculate_fid path: seed-0 random permutation of
    samples before splitting, so a class-ordered gen npz doesn't deflate IS.
    """
    if logits is None:
        return float("nan")
    probs = torch.softmax(logits.float(), dim=1).numpy()
    n = probs.shape[0]
    probs = probs[np.random.RandomState(0).permutation(n)]
    chunk = n // splits
    scores = []
    for k in range(splits):
        part = probs[k * chunk : (k + 1) * chunk]
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(scores))
