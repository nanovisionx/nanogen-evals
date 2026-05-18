try:
    from .clip_t5_model import CLIP_T5_MODELS, CLIPT5Model
except ImportError as e:
    print(f"Error importing CLIP_T5_MODELS: {e}")
    CLIP_T5_MODELS = []
    CLIPT5Model = None
try:
    from .qwen2vl_model import QWEN2_VL_MODELS, Qwen2VLModel
except ImportError as e:
    print(f"Error importing QWEN2_VL_MODELS: {e}")
    QWEN2_VL_MODELS = []
    Qwen2VLModel = None
try:
    from .qwen3vl_model import QWEN3_VL_MODELS, Qwen3VLModel
except ImportError as e:
    print(f"Error importing QWEN3_VL_MODELS: {e}")
    QWEN3_VL_MODELS = []
    Qwen3VLModel = None
try:
    from .qwen3p5_model import QWEN3p5_VL_MODELS, Qwen3p5VLModel
except ImportError as e:
    print(f"Error importing QWEN3p5_VL_MODELS: {e}")
    QWEN3p5_VL_MODELS = []
    Qwen3p5VLModel = None
from ..constants import HF_CACHE_DIR

ALL_VQA_MODELS = [
    CLIP_T5_MODELS,
    QWEN2_VL_MODELS,
    QWEN3_VL_MODELS,
    QWEN3p5_VL_MODELS,
]


def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]

def get_vqascore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models()
    if model_name in CLIP_T5_MODELS:
        return CLIPT5Model(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in QWEN2_VL_MODELS:
        return Qwen2VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in QWEN3_VL_MODELS:
        return Qwen3VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    elif model_name in QWEN3p5_VL_MODELS:
        return Qwen3p5VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    else:
        raise NotImplementedError()