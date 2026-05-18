"""Feature extractors. Adopted from FD-loss: https://github.com/Jiawei-Yang/FD-loss"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ExtractorSpec:
    name: str
    backend: str
    model: str
    input_size: int
    feat_dim: int
    pool: str
    norm_const: float
    default_reference: str


class Extractor(nn.Module):
    spec: ExtractorSpec

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class InceptionExtractor(Extractor):
    def __init__(self, spec: ExtractorSpec):
        super().__init__()
        from torch_fidelity.feature_extractor_inceptionv3 import (
            FeatureExtractorInceptionV3,
        )

        self.spec = spec
        self._inception = FeatureExtractorInceptionV3(
            name="inception-v3-compat",
            features_list=["2048", "logits_unbiased"],
        )
        self._inception.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_uint8_nchw(x)
        feats, _ = self._inception(x)
        return feats.float()

    @torch.inference_mode()
    def forward_with_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _check_uint8_nchw(x)
        feats, logits = self._inception(x)
        return feats.float(), logits.float()


class ConvNextExtractor(Extractor):
    def __init__(self, spec: ExtractorSpec):
        super().__init__()
        import timm

        self.spec = spec
        self.model = timm.create_model(spec.model, pretrained=True, num_classes=0)
        self.model.eval().requires_grad_(False)
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1) * 255.0
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_uint8_nchw(x)
        x = _resize(x.float(), self.spec.input_size)
        x = (x - self.mean) / self.std
        feats = self.model.forward_features(x)
        pooled = feats.mean(dim=[2, 3]) if feats.ndim == 4 else feats.mean(dim=1)
        return pooled.float()


class TimmClsExtractor(Extractor):
    def __init__(self, spec: ExtractorSpec):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        self.spec = spec
        try:
            self.model = timm.create_model(
                spec.model,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )
        except TypeError:
            self.model = timm.create_model(spec.model, pretrained=True, num_classes=0)
        self.model.eval().requires_grad_(False)

        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 0))
        self.has_attn_pool = (
            hasattr(self.model, "attn_pool") and self.model.attn_pool is not None
        )

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        mean = torch.tensor(data_cfg["mean"]).view(1, 3, 1, 1) * 255.0
        std = torch.tensor(data_cfg["std"]).view(1, 3, 1, 1) * 255.0
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_uint8_nchw(x)
        x = _resize(x.float(), self.spec.input_size)
        x = (x - self.mean) / self.std
        with torch.autocast("cuda", enabled=x.is_cuda, dtype=torch.bfloat16):
            feats = self.model.forward_features(x)
        if feats.ndim == 4:
            cls = feats.mean(dim=[2, 3])
        elif self.spec.pool == "attn_pool" and self.has_attn_pool:
            cls = self.model.attn_pool(feats)
        elif self.num_prefix_tokens > 0:
            cls = feats[:, 0]
        else:
            cls = feats.mean(dim=1)
        return cls.float()


def build_extractor(spec: ExtractorSpec) -> Extractor:
    if spec.backend == "torch_fidelity":
        return InceptionExtractor(spec)
    if spec.backend == "timm":
        if spec.pool == "spatial_mean":
            return ConvNextExtractor(spec)
        return TimmClsExtractor(spec)
    raise ValueError(f"unknown backend {spec.backend!r} for extractor {spec.name!r}")


def _resize(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(
        x, size=(size, size), mode="bicubic", align_corners=False, antialias=True
    )


def _check_uint8_nchw(x: torch.Tensor) -> None:
    if x.dtype != torch.uint8 or x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(
            f"expected uint8 NCHW with 3 channels, got dtype={x.dtype} shape={tuple(x.shape)}"
        )
