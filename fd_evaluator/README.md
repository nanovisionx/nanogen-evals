# fd_evaluator

FID, FDR6, and MIND across a fixed six-extractor zoo.

- FDR6: from [FD-loss](https://github.com/Jiawei-Yang/FD-loss)
- MIND: from [torch-fidelity](https://github.com/toshas/torch-fidelity) ([paper](https://arxiv.org/abs/2605.06797))

## Install

```toml
[tool.uv.sources]
fd-evaluator = { git = "ssh://git@github.com/nanovisionx/nanogen-evals.git",
                 subdirectory = "fd_evaluator", branch = "main" }
```

## Usage

```bash
fdeval gen.npz                              # default: fid,inception_score,fdr6,mind6
fdeval gen.npz --metrics fdr6 --out out.csv
```

```python
from fd_evaluator import compute_metrics
result = compute_metrics(images="gen.npz", metrics=["fid", "fdr6", "mind6"],
                        reference_images="imagenet-256-val.npz")
```

Reference moments (~140 MB across 7 files) auto-download from
[`nanovisionx/nanogen-evals-stats`](https://huggingface.co/datasets/nanovisionx/nanogen-evals-stats)
to `~/.cache/nanogen-evals/stats/`. MIND additionally needs raw images
(`imagenet-256-val.npz`) for the reference feature pass.
