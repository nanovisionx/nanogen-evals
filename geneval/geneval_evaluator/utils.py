from typing import Optional
from transformers.utils import TensorType, is_torch_tensor
import numpy as np
import torch
from PIL import Image
from .constants import DEFAULT_OPTIONS, COLORS


def binary_mask_to_rle(mask):
    """
    Converts given binary mask of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        mask (`torch.Tensor` or `numpy.array`):
            A binary mask tensor of shape `(height, width)` where 0 denotes background and 1 denotes the target
            segment_id or class_id.
    Returns:
        `List`: Run-length encoded list of the binary mask. Refer to COCO API for more information about the RLE
        format.
    """
    if is_torch_tensor(mask):
        mask = mask.numpy()

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return list(runs)

def convert_segmentation_to_rle(segmentation):
    """
    Converts given segmentation map of shape `(height, width)` to the run-length encoding (RLE) format.

    Args:
        segmentation (`torch.Tensor` or `numpy.array`):
            A segmentation map of shape `(height, width)` where each value denotes a segment or class id.
    Returns:
        `list[List]`: A list of lists, where each list is the run-length encoding of a segment / class id.
    """
    segment_ids = torch.unique(segmentation)

    run_length_encodings = []
    for idx in segment_ids:
        mask = torch.where(segmentation == idx, 1, 0)
        rle = binary_mask_to_rle(mask)
        run_length_encodings.append(rle)

    return run_length_encodings


def post_process_instance_segmentation(
    outputs,
    image_size: tuple[int, int],
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    target_sizes: Optional[list[tuple[int, int]]] = None,
    return_coco_annotation: Optional[bool] = False,
    return_binary_maps: Optional[bool] = False,
) -> list[dict]:
    if return_coco_annotation and return_binary_maps:
        raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

    # [batch_size, num_queries, num_classes+1]
    class_queries_logits = outputs.class_queries_logits
    # [batch_size, num_queries, height, width]
    masks_queries_logits = outputs.masks_queries_logits

    # Scale back to preprocessed image size
    masks_queries_logits = torch.nn.functional.interpolate(
        masks_queries_logits, size=image_size, mode="bilinear", align_corners=False
    )

    device = masks_queries_logits.device
    num_classes = class_queries_logits.shape[-1] - 1
    num_queries = class_queries_logits.shape[-2]

    # Loop over items in batch size
    results: list[dict[str, TensorType]] = []

    for i in range(class_queries_logits.shape[0]):
        mask_pred = masks_queries_logits[i]
        mask_cls = class_queries_logits[i]

        scores = torch.nn.functional.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
        mask_pred = mask_pred[topk_indices]
        pred_masks = (mask_pred > 0).float()

        # Calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
            pred_masks.flatten(1).sum(1) + 1e-6
        )
        pred_scores = scores_per_image * mask_scores_per_image
        pred_classes = labels_per_image

        segmentation = torch.zeros(image_size) - 1
        if target_sizes is not None:
            segmentation = torch.zeros(target_sizes[i]) - 1
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.unsqueeze(0), size=target_sizes[i], mode="nearest"
            )[0]

        instance_maps, segments = [], []
        current_segment_id = 0
        for j in range(num_queries):
            score = pred_scores[j].item()

            if not torch.all(pred_masks[j] == 0) and score >= threshold:
                segmentation[pred_masks[j] == 1] = current_segment_id
                segments.append(
                    {
                        "id": current_segment_id,
                        "label_id": pred_classes[j].item(),
                        "was_fused": False,
                        "score": round(score, 6),
                    }
                )
                current_segment_id += 1
                instance_maps.append(pred_masks[j])

        # Return segmentation map in run-length encoding (RLE) format
        if return_coco_annotation:
            segmentation = convert_segmentation_to_rle(segmentation)

        # Return a concatenated tensor of binary instance maps
        if return_binary_maps and len(instance_maps) != 0:
            segmentation = torch.stack(instance_maps, dim=0)

        results.append({"segmentation": segmentation, "segments_info": segments})
    return results


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, options=None):
        options = options or DEFAULT_OPTIONS
        self._image = image.convert("RGB")
        bgcolor = options.get("bgcolor", "#999")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._crop = options.get("crop", "1") == "1"

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if self._crop:
            image = image.crop(box[:4])
        return image


def _zero_shot_classifier(classnames, templates, clip_model, clip_processor, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(c=classname) for template in templates]
            inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
            class_embeddings = clip_model.get_text_features(**inputs)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def _run_classification(classifier, dataloader, clip_model, clip_processor, device):
    with torch.no_grad():
        all_logits = []
        # Get logit_scale from the model (crucial for proper classification!)
        logit_scale = clip_model.logit_scale.exp()
        
        for images in dataloader:
            inputs = clip_processor(
                images=images, return_tensors="pt", input_data_format="channels_last"
            ).to(device)
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * (image_features @ classifier)
            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)
    return logits


def _color_classification(image, bboxes, classname, models, device, options, color_classifiers):
    clip_model = models["clip_model"]
    clip_processor = models["clip_processor"]

    if classname not in color_classifiers:
        color_classifiers[classname] = _zero_shot_classifier(
            COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            clip_model, clip_processor, device
        )
    clf = color_classifiers[classname]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes, options),
        batch_size=16, num_workers=1,
        collate_fn=lambda x: x,
    )
    pred = _run_classification(clf, dataloader, clip_model, clip_processor, device)
    return [COLORS[index.item()] for index in pred.argmax(1)]


def _compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def _relative_position(obj_a, obj_b, position_threshold):
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def _mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2, y2], dtype=np.float32)
