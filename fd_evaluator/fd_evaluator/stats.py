"""Reference stats: name -> path. HF download cached under ~/.cache/nanogen-evals/stats/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def _catalogue() -> dict:
    path = Path(__file__).with_name("catalogue.yaml")
    with path.open() as f:
        return yaml.safe_load(f)


def list_references() -> list[str]:
    return sorted(_catalogue()["references"].keys())


def resolve(name: str) -> str:
    cat = _catalogue()
    refs = cat["references"]
    if name not in refs:
        raise KeyError(
            f"unknown reference {name!r}; known: {sorted(refs)}. "
            "Pass an absolute *.npz path to skip resolution."
        )
    filename = refs[name]

    local_dir = os.environ.get("NANOGEN_EVALS_STATS_DIR")
    if local_dir:
        p = Path(local_dir) / filename
        if not p.exists():
            raise FileNotFoundError(
                f"NANOGEN_EVALS_STATS_DIR={local_dir} but {p} not found"
            )
        return str(p)

    from huggingface_hub import hf_hub_download

    cache_dir = Path(os.environ.get("NANOGEN_EVALS_CACHE_DIR", _default_cache_dir()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=cat["hf_repo_id"],
        filename=filename,
        repo_type=cat.get("hf_repo_type", "dataset"),
        cache_dir=str(cache_dir),
    )
    return path


def _default_cache_dir() -> str:
    return str(Path.home() / ".cache" / "nanogen-evals" / "stats")


def load_moments(name_or_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = name_or_path
    if not (name_or_path.endswith(".npz") and os.path.isfile(name_or_path)):
        path = resolve(name_or_path)
    data = np.load(path)
    if "mu" in data and "sigma" in data:
        mu, sigma = data["mu"], data["sigma"]
    elif "ref_mu" in data and "ref_sigma" in data:
        mu, sigma = data["ref_mu"], data["ref_sigma"]
    else:
        raise KeyError(
            f"{path}: expected keys ('mu','sigma') or ('ref_mu','ref_sigma'); "
            f"got {sorted(data.keys())}"
        )
    return np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)


def load_features(name_or_path: str) -> Optional[np.ndarray]:
    path = name_or_path
    if not (name_or_path.endswith(".npz") and os.path.isfile(name_or_path)):
        path = resolve(name_or_path)
    data = np.load(path)
    for key in ("features", "feats", "X"):
        if key in data:
            return np.asarray(data[key])
    return None
