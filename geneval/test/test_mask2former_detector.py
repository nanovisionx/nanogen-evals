import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from geneval_evaluator.evaluator import DEFAULT_DETECTOR_MODEL, LABEL_MAPPING, _mask_to_bbox


DEFAULT_JSONL = "assets/geneval-FLUX.1-dev-cfg3.5-steps30-res1024-seed42_ours.jsonl"
DEFAULT_SAMPLE_ID = "00280/00000.png"
DEFAULT_OUTPUT_DIR = "test_outputs/mask2former_debug"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--model", default=DEFAULT_DETECTOR_MODEL)
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

    processor = Mask2FormerImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(args.device)

    inputs = processor(images=image, return_tensors="pt").to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=args.threshold
    )[0]

    segmentation_map = result["segmentation"].cpu().numpy()
    _colorize_segmentation(segmentation_map).save(output_dir / "segmentation_map.png")

    segments = []
    for idx, segment in enumerate(result["segments_info"]):
        label_id = segment["label_id"]
        score = segment["score"]
        class_label = model.config.id2label[label_id]
        class_label = LABEL_MAPPING.get(class_label, class_label)
        mask = (segmentation_map == segment["id"]).astype(np.uint8) * 255
        bbox = _mask_to_bbox(mask)
        segments.append({
            "segment_index": idx,
            "segment_id": int(segment["id"]),
            "label_id": int(label_id),
            "label": class_label,
            "score": float(score),
            "bbox": None if bbox is None else bbox.tolist(),
        })
        mask_name = f"mask_{idx:03d}_{class_label.replace(' ', '_')}_{score:.3f}.png"
        _save_mask(mask, output_dir / mask_name)
        overlay_name = f"overlay_{idx:03d}_{class_label.replace(' ', '_')}_{score:.3f}.png"
        _save_overlay(image, mask, output_dir / overlay_name)

    with (output_dir / "segments.json").open("w") as f:
        json.dump({
            "sample_id": args.sample_id,
            "prompt": record.get("prompt"),
            "reason": record.get("reason"),
            "details": record.get("details"),
            "segments": segments,
        }, f, indent=2)

    print(f"Saved debug outputs to {output_dir}")


if __name__ == "__main__":
    main()
