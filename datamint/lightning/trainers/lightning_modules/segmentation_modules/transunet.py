"""TransUNet segmentation module.

Architecture: R50-ViT-B/16 encoder (hybrid CNN+Transformer) + cascaded upsampler decoder.
Reference: Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image
           Segmentation", arXiv:2102.04306 (2021).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from ..segmentation_module import SegmentationModule


# ---------------------------------------------------------------------------
# Internal model — not exported
# ---------------------------------------------------------------------------
class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            # in_channel, out_channel, kernel_size=3, padding=1, bias=False
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _DecoderBlock(nn.Module):
    """Single CUP (Cascaded UPsampler) block from the TransUNet decoder. """
    """Source: https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling.py"""

    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0) -> None:
        super().__init__()
        self._skip_channels = skip_channels
        self._conv = nn.Sequential(
            _ConvBnRelu(in_channels + skip_channels, out_channels),
            _ConvBnRelu(out_channels, out_channels),
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        # bilinear upsample before concat.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self._conv(x)


# Maps the user-facing variant name to timm model name and architecture constants.
# skip_channels mirrors the official config: [512, 256, 64, 0].
# Source: github.com/Beckschen/TransUNet/vit_seg_configs.py --> get_r50_b16_config():
#   config.skip_channels = [512, 256, 64, 16]  with  config.n_skip = 3
# The fourth entry (16) is the output width of the last decoder block, not a real
# skip input — n_skip=3 means only the first 3 are used.  We encode this as 0.
_VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    'R50-ViT-B_16': {
        'timm_name': 'vit_base_r50_s16_224',
        'hidden_dim': 768,
        'skip_channels': [512, 256, 64, 0],
        'decoder_channels': [256, 128, 64, 16],
    },
}


class _TransUNetModel(nn.Module):
    """PyTorch TransUNet. """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        variant: str = 'R50-ViT-B_16',
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm  

        cfg = _VARIANT_CONFIGS[variant]
        hidden_dim: int = cfg['hidden_dim']
        skip_ch: list[int] = cfg['skip_channels']
        dec_ch: list[int] = cfg['decoder_channels']

        # use timm's vit_*_r50_s*_224 hybrid model as the encoder
        vit = timm.create_model(cfg['timm_name'], pretrained=pretrained, in_chans=in_channels)

        backbone = vit.patch_embed.backbone  # ResNetV2 (stem + 3 stages)

        # Split the timm stem into pre-pool and pool sub-modules.
        # Source: github.com/Beckschen/TransUNet/vit_seg_modeling_resnet_skip.py --> official ResNetV2.forward():
        # timm's stem is a Sequential(conv, norm, pool).  We separate conv+norm
        # from pool so we can replicate this exact capture-before-pool pattern.
        self._stem_pre_pool = nn.Sequential(backbone.stem.conv, backbone.stem.norm)
        self._stem_pool = backbone.stem.pool
        self._stages = backbone.stages               # 3 ResNet stages
        self._patch_proj = vit.patch_embed.proj      # Conv2d(1024→hidden_dim)
        self._pos_embed = nn.Parameter(vit.pos_embed.data)
        self._pos_drop = vit.pos_drop
        self._blocks = vit.blocks                    # transformer self-attention blocks
        self._norm = vit.norm                        # final LayerNorm

        # CUP decoder: 4 blocks, first 2 fuse CNN skip features.
        decoder_in_ch = [hidden_dim] + dec_ch[:-1]
        self._decoder = nn.ModuleList([
            _DecoderBlock(in_ch, out_ch, s_ch)
            for in_ch, out_ch, s_ch in zip(decoder_in_ch, dec_ch, skip_ch)
        ])

        self._head = nn.Conv2d(dec_ch[-1], num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        # ---- CNN encoder — 3 skip features ----
        # Step through the backbone manually to capture skips at H/2, H/4, H/8.
        # Source: official ResNetV2.forward() builds features = [root, block1, block2]
        # then reverses → [block2@H/8, block1@H/4, root@H/2].
        feat_root = self._stem_pre_pool(x)           # (B, 64,   H/2,  W/2)  ← skip3
        feat_pooled = self._stem_pool(feat_root)     # (B, 64,   H/4,  W/4)
        feat0 = self._stages[0](feat_pooled)         # (B, 256,  H/4,  W/4)  ← skip2
        feat1 = self._stages[1](feat0)               # (B, 512,  H/8,  W/8)  ← skip1
        feat2 = self._stages[2](feat1)               # (B, 1024, H/16, W/16)

        # ---- Patch projection → token sequence ----
        tokens = self._patch_proj(feat2)             # (B, D,    H/16, W/16)
        _, D, Ph, Pw = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)   # (B, N, D)  where N = Ph*Pw

        # skip the cls-token slot when adding positional embeddings.
        tokens = tokens + self._pos_embed[:, 1:]
        tokens = self._pos_drop(tokens)

        # ---- Transformer ----
        for block in self._blocks:
            tokens = block(tokens)
        tokens = self._norm(tokens)

        # Reshape from sequence back to 2-D spatial map.
        spatial = tokens.transpose(1, 2).reshape(B, D, Ph, Pw)  # (B, D, H/16, W/16)

        # ---- CUP decoder ----
        # skip_features mirrors official features[::-1] = [block2, block1, root, None]
        # Source: vit_seg_modeling_resnet_skip.py returns features[::-1]
        skip_features: list[Tensor | None] = [feat1, feat0, feat_root, None]
        out = spatial
        for dec_block, skip in zip(self._decoder, skip_features):
            out = dec_block(out, skip)

        return self._head(out)


# ---------------------------------------------------------------------------
# Lightning module — exported
# ---------------------------------------------------------------------------

class TransUNetModule(SegmentationModule):
    """Segmentation module using the TransUNet architecture.

    Uses a timm ``vit_*_r50_s*_224`` hybrid backbone (ResNet-50 CNN stages +
    ViT transformer) combined with a four-block Cascaded UPsampler decoder.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of segmentation classes excluding background.
        loss_fn: Loss module.
        metrics_factories: ``{name: callable}`` where each callable returns a fresh metric.
        class_names: Human-readable label for each class.
        image_size: ``(height, width)`` used during inference (must be 224×224).
        lr: Learning rate for AdamW.
        variant: TransUNet variant name.  One of ``'R50-ViT-B_16'`` (default)
            or ``'R50-ViT-L_16'``.
        pretrained: Load ImageNet-21k pre-trained ViT weights via timm.
        transform: Albumentations transform applied during inference.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        loss_fn: nn.Module | None = None,
        metrics_factories: dict[str, Callable[[], Any]] = {},
        class_names: list[str] | None = None,
        image_size: tuple[int, int] | None = None,
        lr: float = 1e-4,
        variant: str = 'R50-ViT-B_16',
        pretrained: bool = True,
        transform: A.BasicTransform | A.BaseCompose | None = None,
    ) -> None:
        if variant not in _VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant {variant!r}. Choose from {list(_VARIANT_CONFIGS)}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.variant = variant
        self.pretrained = pretrained

        super().__init__(
            transform=transform,
            loss_fn=loss_fn,
            metrics_factories=metrics_factories,
            class_names=class_names,
            lr=lr,
        )

        self.model = _TransUNetModel(
            in_channels=in_channels,
            num_classes=num_classes,
            variant=variant,
            pretrained=pretrained,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
