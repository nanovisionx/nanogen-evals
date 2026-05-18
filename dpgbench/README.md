# DPG-Bench Evaluator (Minimal)

A minimal, self-contained reimplementation of the [DPG-Bench](https://github.com/TencentQQGYLab/ELLA/blob/main/dpg_bench/compute_dpg_bench.py) evaluation pipeline for text-to-image generation.

## What this repo does

- Provides a lightweight Python package (`dpg_evaluator`) that scores generated images against compositional prompts using Visual Question Answering.
- Uses **mPLUG** (VQA model) with a CLIP vision backbone from Hugging Face Transformers — auto-downloads weights from HuggingFace Hub.
- Exposes simple APIs: `MPLUG`, `evaluate`, `evaluate_batch`, `aggregate_results`, `load_prompt2id`, `load_dpg_metadata`.
- Ships evaluation metadata (prompts, question mappings) as package data.

## What changed from the original DPG-Bench

- Extracted evaluation-only code — no image generation, no ELLA model code.
- Bundled the mPLUG model definition locally to avoid external dependency on modelscope.
- Reduced dependencies to 8 direct packages (numpy, pandas, pillow, pyyaml, torch, torchvision, tqdm, transformers).
- Switched from `requirements.txt` (pip freeze) to a proper `pyproject.toml` with uv for dependency management.
- Category scores use all samples instead of only the last crop's raw scores (see Limitations).

## Limitations

Differences from the [reference implementation](https://github.com/TencentQQGYLab/ELLA/blob/main/dpg_bench/compute_dpg_bench.py):

- **Category scores use all samples**: The reference computes L1/L2 category scores using only the last crop's raw scores (when `pic_num > 1`). This implementation uses raw scores from all samples for category aggregation.
- Tiny (~0.05-0.06%) numerical differences in the scores.

## Installation

```bash
uv pip install git+https://github.com/G-REPA/dpgbench_minimal.git
```

## Quick start

```python
from dpg_evaluator import MPLUG, load_prompt2id, load_dpg_metadata, evaluate
from PIL import Image

# 1. Initialize the VQA model
model = MPLUG()  # auto-downloads from HuggingFace

# 2. Load metadata
prompt2id = load_prompt2id()
question_dict = load_dpg_metadata()

# 3. Prepare your images and prompts
images = [Image.open("0.png"), Image.open("1.png")]
prompts = ["A cat sitting on a mat.", "A dog running in the park."]

# 4. Evaluate (aggregated scores)
mean_score, l1_scores, l2_scores = evaluate(
    images, prompts, prompt2id, question_dict, model.batch_vqa
)
```

## CLI usage

```bash
python test/dpg_evaluation_cli.py --imagedir /path/to/images --outfile results.csv
```

Images should be named as `{id}.png` where id corresponds to keys in `id2prompt.json`.

## Project structure

```
dpg_evaluator/
  __init__.py          # Public API
  evaluator.py         # Scoring, evaluation, aggregation
  utils.py             # MPLUG wrapper, metadata loading
  mplug/               # mPLUG VQA model (local copy)
    modeling_mplug.py   # Model architecture
    configuration_mplug.py  # Config classes
    predictor.py        # Text generation
    clip/               # CLIP vision backbone
  dpg_bench.csv        # DPG-Bench metadata
  id2prompt.json       # ID-to-prompt mapping
  prompt2id.json       # Prompt-to-ID mapping
test/
  dpg_evaluation_cli.py  # CLI for batch evaluation
```

## Acknowledgements

This project is based on [DPG-Bench](https://github.com/TencentQQGYLab/ELLA) by the TencentQQGYLab team.
