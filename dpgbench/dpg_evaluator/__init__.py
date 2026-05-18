"""
DPG-Bench Evaluator for text-to-image generation.

Usage:
    >>> from dpg_evaluator import MPLUG, load_prompt2id, load_dpg_metadata, evaluate
    >>> from PIL import Image
    >>>
    >>> # 1. Initialize the VQA model
    >>> model = MPLUG()  # auto-downloads from HuggingFace
    >>>
    >>> # 2. Load metadata
    >>> prompt2id = load_prompt2id()
    >>> question_dict = load_dpg_metadata()
    >>>
    >>> # 3. Prepare your images and prompts
    >>> images = [Image.open('0.png'), Image.open('1.png')]
    >>> prompts = ['A cat sitting on a mat.', 'A dog running in the park.']
    >>>
    >>> # 4. Evaluate (aggregated scores)
    >>> mean_score, l1_scores, l2_scores = evaluate(
    ...     images, prompts, prompt2id, question_dict, model.batch_vqa
    ... )
    >>>
    >>> # Or get per-sample results
    >>> results = evaluate_batch(
    ...     images, prompts, prompt2id, question_dict, model.batch_vqa
    ... )
    >>> # results[i] = {'score': float, 'qid2tuple': dict, 'raw_scores': dict}
"""

from .evaluator import evaluate, evaluate_batch, aggregate_results, compute_sample_score
from .utils import MPLUG, load_prompt2id, load_dpg_metadata

__all__ = [
    "evaluate",
    "evaluate_batch",
    "aggregate_results",
    "compute_sample_score",
    "MPLUG",
    "load_prompt2id",
    "load_dpg_metadata",
]
