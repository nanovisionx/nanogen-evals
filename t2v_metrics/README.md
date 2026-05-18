# t2v_metrics

Minimal [VQAScore](https://github.com/linzhiqiu/t2v_metrics) implementation for text-to-image evaluation.

This is a stripped-down fork of [linzhiqiu/t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) that keeps only what's needed to compute VQAScore — no video support, no extra metrics, no heavy dependencies.

## What's included

- **VQAScore** computation for image-text alignment
- Batched forward pass support for efficient evaluation
- Model families:
  - **CLIP-FlanT5**: `clip-flant5-xxl`, `clip-flant5-xl`
  - **Qwen2.5-VL**: `qwen2.5-vl-3b`, `qwen2.5-vl-7b`, `qwen2.5-vl-32b`, `qwen2.5-vl-72b`
  - **Qwen3-VL**: `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b`, `qwen3-vl-32b`
  - **Qwen3.5**: `qwen3.5-4b`, `qwen3.5-9b`, `qwen3.5-27b`

## What's removed (vs upstream)

- Video scoring (CLIPScore, PickScore, ImageReward, etc.)
- LLaVA-based models
- Dataset downloading / GenAI-Bench evaluation harness
- All non-VQAScore metrics

## Installation

```bash
pip install git+https://github.com/G-REPA/t2v_metrics_minimal.git
```

## Usage

```python
import t2v_metrics

score_func = t2v_metrics.VQAScore(model="clip-flant5-xxl")

# Single pair
score = score_func(images="image.png", texts="a cat sitting on a couch")

# Batch of paired images and texts
scores = score_func(
    images=["img1.png", "img2.png"],
    texts=["a cat sitting on a couch", "a dog in the park"],
)
```

## Acknowledgements

This project is a minimal fork of [t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) by [Zhiqiu Lin](https://github.com/linzhiqiu).
