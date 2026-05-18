# GenEval Evaluator (Minimal)

A minimal, self-contained reimplementation of the [GenEval](https://github.com/djghosh13/geneval) object-centric evaluation pipeline for text-to-image generation.

## What this repo does

- Provides a lightweight Python package (`geneval_evaluator`) that scores generated images against compositional prompts (object presence, counting, colors, spatial relations).
- Uses **Mask2Former** (instance segmentation) and **CLIP** (zero-shot classification) from Hugging Face Transformers — no mmdet or other heavy dependencies required.
- Exposes simple APIs: `load_models`, `evaluate_image`, `evaluate_batch`, `evaluate_pairs`, `summarize_results`.
- Ships evaluation metadata and class names as package data.

## What changed from the original GenEval

- Removed the image generation code — this repo is evaluation only.
- Replaced the mmdet-based detector with Hugging Face `Mask2FormerForUniversalSegmentation`, eliminating the mmdet/mmcv dependency chain.
- Reimplemented `post_process_instance_segmentation` locally to avoid version-sensitive transformers internals.
- Reduced dependencies to 6 direct packages (numpy, pandas, pillow, torch, tqdm, transformers).
- Switched from `requirements.txt` (pip freeze) to a proper `pyproject.toml` with uv for dependency management.
- Scores now closely match the original implementation.

## Installation

```bash
uv pip install git+https://github.com/G-REPA/geneval_minimal.git
```

## Quick start

```python
from geneval_evaluator import load_models, evaluate_image, summarize_results, fetch_metadata

models = load_models(device="cuda")
metadata_list = fetch_metadata()

# Evaluate a single image against its metadata
from PIL import Image
image = Image.open("path/to/generated.png")
result = evaluate_image(image, metadata_list[0], models)

# Summarize a batch of results
summary = summarize_results([result])
print(summary["overall_score"])
```

## CLI usage

```bash
# Single GPU
python test/evaluation_cli.py --imagedir /path/to/images --outfile results.jsonl

# Multi-GPU
torchrun --nproc_per_node=4 test/evaluation_cli.py --imagedir /path/to/images --outfile results.jsonl

# Summarize scores
python test/summary_scores.py results.jsonl
```

## Project structure

```
geneval_evaluator/
  __init__.py          # Public API
  evaluator.py         # Detection, evaluation, summarization
  utils.py             # Mask processing, CLIP classification, post-processing
  constants.py         # Model configs, thresholds, label mappings
  object_names.txt     # COCO class names
  metadata.json        # GenEval evaluation metadata
test/
  evaluation_cli.py    # CLI for batch evaluation (supports torchrun)
  summary_scores.py    # Print score summary from JSONL results
```

## Acknowledgements

This project is based on [GenEval](https://github.com/djghosh13/geneval) by Dhruba Ghosh et al.
