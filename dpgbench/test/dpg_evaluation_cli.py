#!/usr/bin/env python
"""DPG-Bench CLI - Evaluate generated images.

Usage:
    python dpg_evaluation_cli.py --imagedir /path/to/images --outfile results.csv

Images should be named as {id}.png where id corresponds to keys in id2prompt.json.
"""

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Any, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

from dpg_evaluator.utils import MPLUG, load_prompt2id, load_dpg_metadata
from dpg_evaluator.evaluator import evaluate_batch, aggregate_results


DEFAULT_IMAGEDIR = 'data/self-attention-w2048-3b-res256-e2evaeflux-mjhq512bs128fix-lr2e-5-400k-dpgbench_ema_100000'
DEFAULT_OUTFILE = 'assets/self-attention-w2048-3b-res256-e2evaeflux-mjhq512bs128fix-lr2e-5-400k-dpgbench_ema_100000_ours.csv'
DEFAULT_ID2PROMPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dpg_evaluator/id2prompt.json')


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench Evaluation CLI")
    parser.add_argument("--imagedir", type=str, default=DEFAULT_IMAGEDIR, help="Path to image directory")
    parser.add_argument("--outfile", type=str, default=DEFAULT_OUTFILE, help="Output CSV file")
    parser.add_argument("--id2prompt", type=str, default=DEFAULT_ID2PROMPT, help="Path to id2prompt.json")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to mPLUG checkpoint (auto-downloads if not specified)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for VQA inference")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution for image cropping")
    parser.add_argument("--n-images", type=int, default=4, help="Number of crops per image (1-4)")
    return parser.parse_args()


def load_id2prompt(json_path: str) -> Dict[str, str]:
    """Load id to prompt mapping."""
    with open(json_path, 'r') as f:
        return json.load(f)


def crop_image(image: Image.Image, resolution: int, n_images: int) -> List[Image.Image]:
    """Crop image into quadrants matching reference implementation.

    Args:
        image: PIL image (expected to be resolution*2 x resolution*2 for n_images=4)
        resolution: Size of each crop
        n_images: Number of crops (1-4)

    Returns:
        List of cropped PIL images
    """
    crop_tuples = [
        (0, 0, resolution, resolution),                             # top-left
        (resolution, 0, resolution * 2, resolution),                # top-right
        (0, resolution, resolution, resolution * 2),                # bottom-left
        (resolution, resolution, resolution * 2, resolution * 2),   # bottom-right
    ][:n_images]

    return [image.crop(crop_tuple) for crop_tuple in crop_tuples]


def collect_samples(imagedir: str, id2prompt: Dict[str, str]) -> List[Dict[str, str]]:
    """Collect all valid image samples from directory.

    Args:
        imagedir: Path to directory containing {id}.png images
        id2prompt: Mapping from image ID to prompt text

    Returns:
        List of dicts with keys: image_path, id, prompt
    """
    samples = []
    for filename in sorted(os.listdir(imagedir)):
        if not filename.endswith('.png'):
            continue

        image_id = os.path.splitext(filename)[0]
        if image_id not in id2prompt:
            print(f"Warning: No prompt found for ID {image_id}, skipping", file=sys.stderr)
            continue

        samples.append({
            'image_path': os.path.join(imagedir, filename),
            'id': image_id,
            'prompt': id2prompt[image_id],
        })
    return samples


def load_batch_images(
    samples: List[Dict[str, str]],
    resolution: int,
    n_images: int,
) -> Tuple[List[Image.Image], List[str], List[Dict[str, str]], List[Dict[str, Any]], List[int]]:
    """Load images for a batch of samples, with cropping.

    Args:
        samples: List of sample dicts with image_path, id, prompt
        resolution: Resolution for cropping
        n_images: Number of crops per image

    Returns:
        Tuple of (images, prompts, valid_samples, failed_results, crop_counts)
        - images: Flattened list of all cropped images
        - prompts: Flattened list of prompts (repeated for each crop)
        - valid_samples: Original samples that loaded successfully
        - failed_results: Results for samples that failed to load
        - crop_counts: Number of crops for each valid sample (for averaging)
    """
    images = []
    prompts = []
    valid_samples = []
    failed_results = []
    crop_counts = []

    for sample in samples:
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            cropped_images = crop_image(image, resolution, n_images)
            images.extend(cropped_images)
            prompts.extend([sample['prompt']] * len(cropped_images))
            valid_samples.append(sample)
            crop_counts.append(len(cropped_images))
        except Exception as e:
            print(f"Error loading {sample['image_path']}: {e}", file=sys.stderr)
            failed_results.append({
                'id': sample['id'],
                'prompt': sample['prompt'],
                'score': None,
                'image_path': sample['image_path'],
            })

    return images, prompts, valid_samples, failed_results, crop_counts


def process_batch(
    samples: List[Dict[str, str]],
    model: MPLUG,
    prompt2id: Dict[str, str],
    question_dict: Dict[str, Dict],
    batch_size: int,
    resolution: int,
    n_images: int,
) -> List[Dict[str, Any]]:
    """Process a batch of samples: load images, crop, evaluate, return results.

    Args:
        samples: List of sample dicts with image_path, id, prompt
        model: MPLUG model instance
        prompt2id: Prompt text to item_id mapping
        question_dict: DPG-Bench metadata
        batch_size: Batch size for VQA inference
        resolution: Resolution for image cropping
        n_images: Number of crops per image

    Returns:
        List of result dicts with id, prompt, score, image_path, qid2tuple, raw_scores
    """
    # Load and crop images
    images, prompts, valid_samples, failed_results, crop_counts = load_batch_images(
        samples, resolution, n_images
    )

    if not images:
        return failed_results

    # Batched VQA function
    def batched_vqa_fn(vqa_images: List[Image.Image], questions: List[str]) -> List[str]:
        all_answers = []
        for i in range(0, len(vqa_images), batch_size):
            batch_imgs = vqa_images[i:i + batch_size]
            batch_qs = questions[i:i + batch_size]
            answers = model.batch_vqa(batch_imgs, batch_qs)
            all_answers.extend(answers)
        return all_answers

    # Evaluate all cropped images
    eval_results = evaluate_batch(images, prompts, prompt2id, question_dict, batched_vqa_fn)

    # Average scores across crops for each original sample
    results = []
    result_idx = 0
    for sample, crop_count in zip(valid_samples, crop_counts):
        crop_results = eval_results[result_idx:result_idx + crop_count]
        result_idx += crop_count

        # Average scores across crops (matching reference implementation)
        avg_score = sum(r['score'] for r in crop_results) / len(crop_results)

        # Use raw_scores and qid2tuple from last crop (consistent with reference)
        results.append({
            'id': sample['id'],
            'prompt': sample['prompt'],
            'image_path': sample['image_path'],
            'score': avg_score,
            'qid2tuple': crop_results[-1]['qid2tuple'],
            'raw_scores': crop_results[-1]['raw_scores'],
        })

    return failed_results + results


def gather_results(
    all_results: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], float]:
    """Aggregate results and compute category scores.

    Args:
        all_results: List of all result dicts from process_batch

    Returns:
        Tuple of (results_df, l1_category_scores, l2_category_scores, mean_score)
    """
    # Filter valid results for aggregation
    valid_results = [r for r in all_results if r.get('score') is not None]

    # Use evaluator's aggregate_results
    mean_score, l1_category_scores, l2_category_scores = aggregate_results(valid_results)

    # Build output dataframe (exclude internal fields)
    df = pd.DataFrame([{
        'id': r['id'],
        'prompt': r['prompt'],
        'score': r.get('score'),
        'image_path': r['image_path'],
    } for r in all_results])

    return df, l1_category_scores, l2_category_scores, mean_score


def main():
    args = parse_args()

    # Load id2prompt mapping
    print(f"Loading id2prompt mapping from {args.id2prompt}...", file=sys.stderr)
    id2prompt = load_id2prompt(args.id2prompt)
    print(f"Loaded {len(id2prompt)} prompt mappings", file=sys.stderr)

    # Collect samples
    print(f"Scanning images in {args.imagedir}...", file=sys.stderr)
    samples = collect_samples(args.imagedir, id2prompt)
    print(f"Found {len(samples)} images", file=sys.stderr)
    print(f"Using {args.n_images} crops per image at resolution {args.resolution}", file=sys.stderr)

    if not samples:
        print("No images found!", file=sys.stderr)
        return

    # Load model and metadata
    print(f"Loading mPLUG model on {args.device}...", file=sys.stderr)
    model = MPLUG(model_dir=args.model_dir, device=args.device)

    print("Loading DPG-Bench metadata...", file=sys.stderr)
    prompt2id = load_prompt2id()
    question_dict = load_dpg_metadata()

    # Evaluate in batches
    all_results = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Evaluating batches"):
        max_idx = min(i + args.batch_size, len(samples))
        batch_samples = samples[i:max_idx]
        batch_results = process_batch(
            batch_samples, model, prompt2id, question_dict, args.batch_size,
            args.resolution, args.n_images
        )
        all_results.extend(batch_results)

    # Gather results
    df, l1_category_scores, l2_category_scores, mean_score = gather_results(all_results)

    # Print summary
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"DPG-Bench Score: {mean_score * 100:.2f}", file=sys.stderr)
    print(f"\nL1 Category Scores:", file=sys.stderr)
    for cat, score in sorted(l1_category_scores.items()):
        print(f"  {cat}: {score * 100:.2f}", file=sys.stderr)
    print(f"\nL2 Category Scores:", file=sys.stderr)
    for cat, score in sorted(l2_category_scores.items()):
        print(f"  {cat}: {score * 100:.2f}", file=sys.stderr)
    print(f"{'='*50}\n", file=sys.stderr)

    # Save summary only
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    summary_rows = [{'category': 'Overall', 'level': 'mean', 'score': mean_score}]
    for cat, score in l1_category_scores.items():
        summary_rows.append({'category': cat, 'level': 'L1', 'score': score})
    for cat, score in l2_category_scores.items():
        summary_rows.append({'category': cat, 'level': 'L2', 'score': score})
    pd.DataFrame(summary_rows).to_csv(args.outfile, index=False)
    print(f"Saved summary to {args.outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
