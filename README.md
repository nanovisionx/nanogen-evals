# nanogen-evals

Eval shims for [nanogen](https://github.com/nanovisionx/nanogen). Three independent text-to-image evaluators consolidated into one repo, each installable as its own package.

| Subdir | Package | Metric | Upstream |
|---|---|---|---|
| `dpgbench/` | `dpg_evaluator` | DPG-Bench (VQA-based composition) | [TencentQQGYLab/ELLA](https://github.com/TencentQQGYLab/ELLA) |
| `geneval/` | `geneval_evaluator` | GenEval (object presence, count, color, position) | [djghosh13/geneval](https://github.com/djghosh13/geneval) |
| `t2v_metrics/` | `t2v_metrics` | VQAScore (image-text alignment) | [linzhiqiu/t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) |
| `fd_evaluator/` | `fd_evaluator` | FID / FDR6 / MIND across six representation spaces | [Jiawei-Yang/FD-loss](https://github.com/Jiawei-Yang/FD-loss), [toshas/torch-fidelity](https://github.com/toshas/torch-fidelity) |

Each fork strips upstream to evaluation-only code, removes heavy deps (mmdet, modelscope, video stack), and standardizes on `pyproject.toml` + uv. Numerical scores closely match the originals — see per-subdir READMEs for any deviations.

## Install

Via uv with `subdirectory`:

```toml
[tool.uv.sources]
dpg-evaluator     = { git = "ssh://git@github.com/nanovisionx/nanogen-evals.git", subdirectory = "dpgbench",     branch = "main" }
geneval-evaluator = { git = "ssh://git@github.com/nanovisionx/nanogen-evals.git", subdirectory = "geneval",      branch = "main" }
t2v-metrics       = { git = "ssh://git@github.com/nanovisionx/nanogen-evals.git", subdirectory = "t2v_metrics",  branch = "main" }
fd-evaluator      = { git = "ssh://git@github.com/nanovisionx/nanogen-evals.git", subdirectory = "fd_evaluator", branch = "main" }
```

Or pip:

```bash
pip install "git+https://github.com/nanovisionx/nanogen-evals.git#subdirectory=dpgbench"
```

Usage and CLI examples: see each subdir's README.

## Citation

Please cite the original benchmarks and nanogen:

```bibtex
@article{nanogen,
  title   = {nanogen: A Unified Codebase for Scaling State-of-the-Art Diffusion Transformers},
  author  = {nanogen team},
  journal = {arXiv preprint},
  year    = {2026}
}
```
