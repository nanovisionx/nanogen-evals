#!/usr/bin/env python
"""GenEval CLI - Evaluate generated images.

Usage:
    # Single GPU
    python evaluation_cli.py --imagedir /path/to/images --outfile results.jsonl

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 evaluation_cli.py --imagedir /path/to/images --outfile results.jsonl
"""

import argparse
import json
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import torch.distributed as dist

from geneval_evaluator import (
    load_models,
    evaluate_image,
    DEFAULT_OPTIONS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GenEval Evaluation CLI")
    parser.add_argument("--imagedir", type=str, required=True, help="Path to geneval-structured directory")
    parser.add_argument("--outfile", type=str, default="results.jsonl", help="Output JSONL file")
    parser.add_argument("--detector-model", type=str, default=None)
    parser.add_argument("--clip-model", type=str, default=None)
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    return args


def get_all_samples(imagedir):
    """Get list of all (subfolder, imagename) pairs."""
    samples = []
    for subfolder in sorted(os.listdir(imagedir)):
        folderpath = os.path.join(imagedir, subfolder)
        if not os.path.isdir(folderpath) or not subfolder.isdigit():
            continue
        metadata_path = os.path.join(folderpath, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            continue
        samples_dir = os.path.join(folderpath, "samples")
        if not os.path.isdir(samples_dir):
            continue
        for imagename in sorted(os.listdir(samples_dir)):
            if re.match(r"\d+\.png", imagename):
                samples.append((subfolder, imagename))
    return samples


def main():
    args = parse_args()
    
    # Get distributed info from torchrun env vars
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    
    # Initialize distributed if multi-GPU
    if world_size > 1:
        dist.init_process_group(backend="gloo")
    
    # Get and shard samples
    all_samples = get_all_samples(args.imagedir)
    my_samples = [s for i, s in enumerate(all_samples) if i % world_size == rank]
    
    if world_size > 1:
        print(f"[Rank {rank}/{world_size}] {len(my_samples)} samples on {device}", file=sys.stderr)
    
    results = []
    
    if my_samples:
        # Load models
        print(f"Loading models on {device}...", file=sys.stderr)
        models = load_models(device=device, detector_model=args.detector_model, clip_model=args.clip_model)
        options = {**DEFAULT_OPTIONS, **args.options}
        
        # Evaluate
        metadata_cache = {}
        iterator = tqdm(my_samples, desc=f"Rank {rank}") if rank == 0 else my_samples
        
        for subfolder, imagename in iterator:
            if subfolder not in metadata_cache:
                with open(os.path.join(args.imagedir, subfolder, "metadata.jsonl")) as f:
                    metadata_cache[subfolder] = json.load(f)
            
            imagepath = os.path.join(args.imagedir, subfolder, "samples", imagename)
            image = ImageOps.exif_transpose(Image.open(imagepath))
            result = evaluate_image(image, metadata_cache[subfolder], models, device, options)
            result["filename"] = imagepath
            result["sample_id"] = f"{subfolder}/{imagename}"
            results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    
    if world_size > 1:
        # Save each rank's results separately
        base, ext = os.path.splitext(args.outfile)
        rank_file = f"{base}_rank{rank}{ext}"
        pd.DataFrame(results).to_json(rank_file, orient="records", lines=True)
        
        # Wait for all ranks to finish saving
        dist.barrier()
        
        # Rank 0 gathers and saves final results
        if rank == 0:
            all_results = []
            for r in range(world_size):
                rf = f"{base}_rank{r}{ext}"
                with open(rf) as f:
                    for line in f:
                        all_results.append(json.loads(line))
                os.remove(rf)
            pd.DataFrame(all_results).to_json(args.outfile, orient="records", lines=True)
            print(f"Saved {len(all_results)} results to {args.outfile}", file=sys.stderr)
        
        dist.destroy_process_group()
    else:
        pd.DataFrame(results).to_json(args.outfile, orient="records", lines=True)
        print(f"Saved {len(results)} results to {args.outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
