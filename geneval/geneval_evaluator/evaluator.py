"""GenEval evaluation functions."""

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from .constants import DEFAULT_DETECTOR_MODEL, DEFAULT_CLIP_MODEL, CLIP_MODEL_MAPPING, LABEL_MAPPING, DEFAULT_OPTIONS, COLORS
from .utils import _mask_to_bbox, _color_classification, _relative_position, _compute_iou, post_process_instance_segmentation


def load_models(device="cuda", detector_model=None, clip_model=None):
    """Load Mask2Former and CLIP models."""
    detector_model = detector_model or DEFAULT_DETECTOR_MODEL
    clip_model = clip_model or DEFAULT_CLIP_MODEL
    clip_model = CLIP_MODEL_MAPPING.get(clip_model, clip_model)

    object_detector = Mask2FormerForUniversalSegmentation.from_pretrained(detector_model).to(device)
    detector_processor = Mask2FormerImageProcessor.from_pretrained(detector_model)
    clip_model_instance = CLIPModel.from_pretrained(clip_model).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model)

    with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]

    return {
        "object_detector": object_detector,
        "detector_processor": detector_processor,
        "clip_model": clip_model_instance,
        "clip_processor": clip_processor,
        "classnames": classnames,
    }


def _evaluate_detections(image, objects, metadata, models, device, options, color_classifiers):
    position_threshold = float(options.get("position_threshold", 0.1))
    correct = True
    reason = []
    matched_groups = []

    for req in metadata.get("include", []):
        classname = req["class"]
        matched = True
        found_objects = objects.get(classname, [])[:req["count"]]
        if len(found_objects) < req["count"]:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if "color" in req:
                colors = _color_classification(image, found_objects, classname, models, device, options, color_classifiers)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found " +
                        f"{colors.count(req['color'])} {req['color']}; and " +
                        ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if "position" in req and matched:
                expected_rel, target_group = req["position"]
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = _relative_position(obj, target_obj, position_threshold)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found " +
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        matched_groups.append(found_objects if matched else None)

    for req in metadata.get("exclude", []):
        classname = req["class"]
        if len(objects.get(classname, [])) >= req["count"]:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")

    return correct, "\n".join(reason)


def _detect_objects(image, models, device, options):
    object_detector = models["object_detector"]
    detector_processor = models["detector_processor"]
    classnames = models["classnames"]

    inputs = detector_processor(
        images=image,
        return_tensors="pt",
        size={"shortest_edge": 800, "longest_edge": 1333},
    ).to(device)

    with torch.no_grad():
        outputs = object_detector(**inputs)

    result = post_process_instance_segmentation(
        outputs, inputs['pixel_values'].shape[2:], target_sizes=[image.size[::-1]],
        threshold=float(options.get("threshold", 0.3)),
        return_binary_maps=True
    )[0]

    class_detections = {classname: [] for classname in classnames}
    segmentation_map = result["segmentation"].cpu().numpy()

    for segment in result["segments_info"]:
        label_id = segment["label_id"]
        score = segment["score"]
        # Map transformers label to standard COCO label
        class_label = object_detector.config.id2label[label_id]
        class_label = LABEL_MAPPING.get(class_label, class_label)
        if class_label not in class_detections:
            continue
        mask = (segmentation_map[segment["id"]] * 255).astype(np.uint8)
        bbox = _mask_to_bbox(mask)
        if bbox is None:
            continue
        bbox_with_score = np.append(bbox, score)
        class_detections[class_label].append((bbox_with_score, mask, score))

    return class_detections, classnames


def _filter_detections(class_detections, classnames, metadata, options):
    threshold = float(options.get("threshold", 0.3))
    counting_threshold = float(options.get("counting_threshold", 0.9))
    max_objects = int(options.get("max_objects", 16))
    nms_threshold = float(options.get("max_overlap", 1.0))

    confidence_threshold = threshold if metadata.get("tag") != "counting" else counting_threshold
    detected = {}

    for classname in classnames:
        detections = class_detections[classname]
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        detections = [d for d in detections if d[2] > confidence_threshold]
        detections = detections[:max_objects]

        kept = []
        for det in detections:
            bbox, mask, _ = det
            should_keep = True
            for kept_det in kept:
                if nms_threshold < 1 and _compute_iou(bbox, kept_det[0]) >= nms_threshold:
                    should_keep = False
                    break
            if should_keep:
                kept.append((bbox, mask))
        if kept:
            detected[classname] = kept

    return detected


def evaluate_image(image, metadata, models, device="cuda", options=None):
    """Evaluate a single image."""
    options = {**DEFAULT_OPTIONS, **(options or {})}
    color_classifiers = {}

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    class_detections, classnames = _detect_objects(image, models, device, options)
    detected = _filter_detections(class_detections, classnames, metadata, options)
    is_correct, reason = _evaluate_detections(image, detected, metadata, models, device, options, color_classifiers)

    return {
        "tag": metadata.get("tag", ""),
        "prompt": metadata.get("prompt", ""),
        "correct": is_correct,
        "reason": reason,
        "metadata": json.dumps(metadata),
        "details": json.dumps({
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        })
    }


def evaluate_batch(images, metadata_list, models, device="cuda", options=None, show_progress=True):
    """
    Evaluate a batch of images.

    Args:
        images: list of list, images[i] = samples for metadata_list[i]
        metadata_list: list of metadata dicts
        models: loaded models dict
        device: cuda/cpu device
        options: evaluation options
        show_progress: whether to show tqdm progress bar

    Returns:
        list of result dicts with prompt_idx, sample_idx, tag, prompt, correct, reason, etc.
    """
    from tqdm import tqdm

    options = {**DEFAULT_OPTIONS, **(options or {})}
    color_classifiers = {}
    results = []

    # Count total images for progress bar
    total_images = sum(len(img_list) for img_list in images)

    # Create iterator with optional progress bar
    def iterate_images():
        for prompt_idx, (image_list, metadata) in enumerate(zip(images, metadata_list)):
            for sample_idx, image in enumerate(image_list):
                yield prompt_idx, sample_idx, image, metadata

    iterator = iterate_images()
    if show_progress:
        iterator = tqdm(iterator, total=total_images, desc="Evaluating")

    for prompt_idx, sample_idx, image, metadata in iterator:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        class_detections, classnames = _detect_objects(image, models, device, options)
        detected = _filter_detections(class_detections, classnames, metadata, options)
        is_correct, reason = _evaluate_detections(image, detected, metadata, models, device, options, color_classifiers)

        results.append({
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "tag": metadata.get("tag", ""),
            "prompt": metadata.get("prompt", ""),
            "correct": is_correct,
            "reason": reason,
            "metadata": json.dumps(metadata),
            "details": json.dumps({
                key: [box.tolist() for box, _ in value]
                for key, value in detected.items()
            })
        })

    return results


def evaluate_pairs(images, metadata_list, models, device="cuda", options=None, show_progress=True):
    """
    Evaluate a list of (image, metadata) pairs (1 image per metadata).

    Args:
        images: list of images (PIL.Image.Image or np.ndarray)
        metadata_list: list of metadata dicts (same length as images)
        models: loaded models dict
        device: cuda/cpu device
        options: evaluation options
        show_progress: whether to show tqdm progress bar

    Returns:
        list of result dicts, one per input pair. (prompt_idx matches the input index, sample_idx is 0)
    """
    if len(images) != len(metadata_list):
        raise ValueError(f"images and metadata_list must have the same length, got {len(images)} vs {len(metadata_list)}")

    # Reuse evaluate_batch by treating each image as a single-sample list for its corresponding metadata.
    grouped_images = [[img] for img in images]
    return evaluate_batch(
        grouped_images,
        metadata_list,
        models=models,
        device=device,
        options=options,
        show_progress=show_progress,
    )


def summarize_results(results):
    """
    Compute summary metrics from evaluation results (equivalent to test/summary_scores.py).

    Args:
        results: list[dict] as returned by evaluate_image/evaluate_batch/evaluate_pairs,
                 or a pandas DataFrame with the same columns.

    Returns:
        dict with:
          - total_images
          - total_prompts
          - correct_images (fraction)
          - correct_prompts (fraction)
          - task_scores: {tag: fraction_correct}
          - overall_score: average of task_scores over tags (float)
    """
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    if df.empty:
        return {
            "total_images": 0,
            "total_prompts": 0,
            "correct_images": 0.0,
            "correct_prompts": 0.0,
            "task_scores": {},
            "overall_score": 0.0,
        }

    # Measure overall success
    total_images = int(len(df))
    total_prompts = int(df["metadata"].nunique()) if "metadata" in df.columns else int(len(df))
    correct_images = float(df["correct"].mean()) if "correct" in df.columns else 0.0

    if "metadata" in df.columns and "correct" in df.columns:
        correct_prompts = float(df.groupby("metadata")["correct"].any().mean())
    else:
        correct_prompts = 0.0

    # By group (tag)
    task_scores = {}
    task_score_list = []
    if "tag" in df.columns and "correct" in df.columns:
        for tag, task_df in df.groupby("tag", sort=False):
            score = float(task_df["correct"].mean())
            task_scores[str(tag)] = score
            task_score_list.append(score)

    overall_score = float(np.mean(task_score_list)) if task_score_list else 0.0

    return {
        "total_images": total_images,
        "total_prompts": total_prompts,
        "correct_images": correct_images,
        "correct_prompts": correct_prompts,
        "task_scores": task_scores,
        "overall_score": overall_score,
    }


def gather_results(results_dir, outfile=None):
    """
    Gather results from multiple JSONL files into a single DataFrame.
    
    Args:
        results_dir: Directory containing JSONL result files
        outfile: Optional output file path to save combined results
        
    Returns:
        Combined DataFrame of all results
    """
    all_results = []
    
    for filename in glob(os.path.join(results_dir, "*.jsonl")):
        with open(filename) as f:
            for line in f:
                all_results.append(json.loads(line))
    
    combined = pd.DataFrame(all_results)
    
    if outfile:
        if os.path.dirname(outfile):
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, "w") as f:
            combined.to_json(f, orient="records", lines=True)
    
    return combined


def fetch_metadata():
    with open(os.path.join(os.path.dirname(__file__), "metadata.json"), "r") as f:
        rtn = json.load(f)
    return rtn
