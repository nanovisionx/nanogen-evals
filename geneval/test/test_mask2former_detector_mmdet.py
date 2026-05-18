import argparse
import json
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List

import numpy as np
from PIL import Image
import torch
import mmdet
from mmdet.apis import inference_detector, init_detector

from geneval_evaluator.evaluator import LABEL_MAPPING, _mask_to_bbox


DEFAULT_JSONL = "assets/geneval-FLUX.1-dev-cfg3.5-steps30-res1024-seed42_ours.jsonl"
DEFAULT_SAMPLE_ID = "00280/00000.png"
DEFAULT_OUTPUT_DIR = "test_outputs/mask2former_debug_mmdet"
DEFAULT_MODEL_NAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"


def _load_record(jsonl_path: Path, sample_id: str) -> dict:
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("sample_id") == sample_id:
                return rec
    raise ValueError(f"sample_id {sample_id} not found in {jsonl_path}")


def _colorize_segmentation(segmentation_map: np.ndarray) -> Image.Image:
    h, w = segmentation_map.shape
    rng = np.random.RandomState(0)
    unique_ids = np.unique(segmentation_map)
    color_map = {}
    for seg_id in unique_ids:
        if seg_id == 0:
            color_map[int(seg_id)] = (0, 0, 0)
        else:
            color_map[int(seg_id)] = tuple(rng.randint(0, 256, size=3).tolist())
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for seg_id, color in color_map.items():
        color_img[segmentation_map == seg_id] = color
    return Image.fromarray(color_img, mode="RGB")


def _save_mask(mask: np.ndarray, path: Path) -> None:
    mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
    mask_img.save(path)


def _save_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    base = np.array(image.convert("RGB"))
    overlay = base.copy()
    overlay[mask > 0] = [255, 0, 0]
    blended = (0.7 * base + 0.3 * overlay).astype(np.uint8)
    Image.fromarray(blended, mode="RGB").save(path)


def _default_config_path() -> str:
    return os.path.join(
        os.path.dirname(mmdet.__file__),
        "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    )


def _load_classnames() -> List[str]:
    with open(os.path.join(os.path.dirname(__file__), "..", "geneval_evaluator", "object_names.txt")) as cls_file:
        return [line.strip() for line in cls_file]


def _extract_masks(segm, class_index):
    if segm is None:
        return None
    masks = segm[class_index]
    if hasattr(masks, "to_ndarray"):
        return masks.to_ndarray()
    return masks


def _nms_keep(detections, nms_threshold):
    kept = []
    for det in detections:
        bbox = det[:4]
        should_keep = True
        for kept_det in kept:
            if nms_threshold < 1 and _compute_iou(bbox, kept_det[:4]) >= nms_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(det)
    return kept


def _compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--counting-threshold", type=float, default=0.9)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--max-overlap", type=float, default=1.0)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--model-path", default="../geneval/checkpoints")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    record = _load_record(jsonl_path, args.sample_id)
    image_path = Path(record["filename"])
    if not image_path.is_absolute():
        image_path = Path(__file__).resolve().parents[1] / image_path

    output_dir = Path(args.output_dir) / args.sample_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    image.save(output_dir / "input.png")

    model_config = args.model_config or _default_config_path()
    checkpoint_path = os.path.join(args.model_path, f"{args.model_name}.pth")
    detector = init_detector(model_config, checkpoint_path, device=args.device)

    result = inference_detector(detector, str(image_path))
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None

    classnames = _load_classnames()
    confidence_threshold = args.threshold if record.get("tag") != "counting" else args.counting_threshold
    segmentation_map = np.zeros((image.height, image.width), dtype=np.int32)

    segments = []
    segment_index = 0
    for class_index, classname in enumerate(classnames):
        if class_index >= len(bbox):
            break
        class_label = LABEL_MAPPING.get(classname, classname)
        class_dets = bbox[class_index]
        if class_dets.size == 0:
            continue
        ordering = np.argsort(class_dets[:, 4])[::-1]
        ordering = ordering[class_dets[ordering, 4] > confidence_threshold]
        ordering = ordering[: args.max_objects].tolist()

        kept = _nms_keep(class_dets[ordering], args.max_overlap)
        masks_for_class = _extract_masks(segm, class_index)
        for det in kept:
            score = float(det[4])
            mask = None
            if masks_for_class is not None and len(masks_for_class) > 0:
                mask_index = int(np.where((class_dets == det).all(axis=1))[0][0])
                mask = masks_for_class[mask_index]
                if hasattr(mask, "astype"):
                    mask = mask.astype(np.uint8) * 255
                else:
                    mask = np.array(mask, dtype=np.uint8) * 255

            if mask is not None:
                bbox_from_mask = _mask_to_bbox(mask)
                bbox_out = None if bbox_from_mask is None else bbox_from_mask.tolist()
            else:
                bbox_out = det[:4].tolist()

            segments.append({
                "segment_index": segment_index,
                "segment_id": segment_index + 1,
                "label_id": int(class_index),
                "label": class_label,
                "score": score,
                "bbox": bbox_out,
            })

            if mask is not None:
                mask_name = f"mask_{segment_index:03d}_{class_label.replace(' ', '_')}_{score:.3f}.png"
                _save_mask(mask, output_dir / mask_name)
                overlay_name = f"overlay_{segment_index:03d}_{class_label.replace(' ', '_')}_{score:.3f}.png"
                _save_overlay(image, mask, output_dir / overlay_name)
                segmentation_map[mask > 0] = segment_index + 1
            segment_index += 1

    if np.any(segmentation_map):
        _colorize_segmentation(segmentation_map).save(output_dir / "segmentation_map.png")

    with (output_dir / "segments.json").open("w") as f:
        json.dump({
            "sample_id": args.sample_id,
            "prompt": record.get("prompt"),
            "reason": record.get("reason"),
            "details": record.get("details"),
            "segments": segments,
            "options": {
                "threshold": args.threshold,
                "counting_threshold": args.counting_threshold,
                "max_objects": args.max_objects,
                "max_overlap": args.max_overlap,
                "model_name": args.model_name,
                "model_config": model_config,
                "model_path": args.model_path,
            },
        }, f, indent=2)

    print(f"Saved debug outputs to {output_dir}")


if __name__ == "__main__":
    main()
