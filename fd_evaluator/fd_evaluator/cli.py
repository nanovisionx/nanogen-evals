"""fdeval CLI: `fdeval gen.npz [--metrics ...] [--out ...]`."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from .api import compute_metrics
from .format import format_results

_DEFAULT_REF_IMAGES = "./data/imagenet/imagenet-256-val.npz"
_DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "nanogen-evals" / "features")
_DEFAULT_METRICS = "fid,inception_score,fdr6,mind6"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fdeval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=(
            "Env overrides (optional):\n"
            "  NANOGEN_EVALS_STATS_DIR   point at a local dir of *.npz stats "
            "(disables HF download)\n"
            "  NANOGEN_EVALS_REF_IMAGES  path to the raw imagenet val npz; "
            "default %s\n"
            "  NANOGEN_EVALS_CACHE_DIR   feature cache; default %s\n"
        )
        % (_DEFAULT_REF_IMAGES, _DEFAULT_CACHE_DIR),
    )
    parser.add_argument("npz", help="generated uint8 NHWC RGB npz (key arr_0)")
    parser.add_argument(
        "--metrics",
        default=_DEFAULT_METRICS,
        help=f"comma-separated. default: {_DEFAULT_METRICS}. bundles: fdr6, mind6. "
        "singles: fid, inception_score, fd_<ext>, fdr_<ext>, mind_<ext>.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="optional CSV path; prints to stdout regardless.",
    )

    args = parser.parse_args(argv)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    ref_images = os.environ.get("NANOGEN_EVALS_REF_IMAGES", _DEFAULT_REF_IMAGES)
    cache_dir = os.environ.get("NANOGEN_EVALS_CACHE_DIR", _DEFAULT_CACHE_DIR)
    needs_mind = any(m.startswith("mind") for m in metrics)
    if needs_mind and not Path(ref_images).exists():
        raise SystemExit(
            f"[fdeval] MIND metric requested but raw-image reference not found:\n"
            f"  {ref_images}\n"
            f"Set NANOGEN_EVALS_REF_IMAGES to override, or drop mind6 from --metrics."
        )

    print(f"[fdeval] npz: {args.npz}")
    print(f"[fdeval] metrics: {metrics}")

    results = compute_metrics(
        images=args.npz,
        metrics=metrics,
        reference_images=ref_images if needs_mind else None,
        feature_cache_dir=cache_dir,
        feature_cache_key=Path(args.npz).stem,
        reference_feature_cache_key="imagenet256_val",
    )

    print()
    print(format_results(results))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fieldnames = ["npz"] + sorted(results.keys())
        row = {"npz": Path(args.npz).name, **results}
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        print(f"\n[fdeval] wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
