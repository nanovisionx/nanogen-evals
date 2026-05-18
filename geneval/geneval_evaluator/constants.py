CLIP_MODEL_MAPPING = {
    "ViT-L-14": "openai/clip-vit-large-patch14",
    "ViT-B-32": "openai/clip-vit-base-patch32",
    "ViT-B-16": "openai/clip-vit-base-patch16",
}

DEFAULT_DETECTOR_MODEL = "facebook/mask2former-swin-small-coco-instance"
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]

# Mapping from transformers Mask2Former labels to standard COCO labels used in metadata
LABEL_MAPPING = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "diningtable": "dining table",
    "tvmonitor": "tv",
    "mouse": "computer mouse",
    "remote": "tv remote",
    "keyboard": "computer keyboard",
}

DEFAULT_OPTIONS = {
    "threshold": 0.3,
    "counting_threshold": 0.9,
    "max_objects": 16,
    "max_overlap": 1.0,
    "position_threshold": 0.1,
    "bgcolor": "#999",
    "crop": "1",
}
