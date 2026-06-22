"""UNETR++ 3D segmentation module.

Architecture: hierarchical transformer encoder with Efficient Paired Attention (EPA)
and a CNN decoder with skip connections.

Reference: Shaker et al., "UNETR++: Delving into Efficient and Accurate 3D Medical
           Image Segmentation", IEEE TMI 2024.
Adapted from: https://github.com/Amshaker/unetr_plus_plus.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

import albumentations as A

from ..segmentation_module import SegmentationModule


# ---------------------------------------------------------------------------
# Internal building blocks — not exported
# ---------------------------------------------------------------------------

class _ResBlock3D(nn.Module):
    """3-D residual conv block (two Conv3d + BN + ReLU with skip).
    Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/model_components.py
    Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/transformerblock.py
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.body(x) + self.skip(x))


class _EPA(nn.Module):
    """Efficient Paired Attention.

    Two parallel attention branches (channel + spatial) that share the same
    Q and K projections, keeping compute sub-quadratic in sequence length.

    Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/transformerblock.py

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int = 4,
        channel_attn_drop: float = 0.1,
        spatial_attn_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        
        # Per-head temperature parameters (one for channel attn, one for spatial)
        self.temp_ca = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temp_sa = nn.Parameter(torch.ones(num_heads, 1, 1))

        # One linear that produces 4 projections: q_shared, k_shared, v_CA, v_SA
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=False)

        # Tied E = F: projects sequence dim N -> proj_size for spatial attention.
        # Using the same object means both get identical gradients → shared weights.
        self.E = self.F = nn.Linear(input_size, proj_size)

        self.drop_ca = nn.Dropout(channel_attn_drop)
        self.drop_sa = nn.Dropout(spatial_attn_drop)

        # Each branch projects C → C//2; concat restores C
        self.out_sa = nn.Linear(hidden_size, hidden_size // 2)
        self.out_ca = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        head_dim = C // self.num_heads

        # (B, N, 4C) → (4, B, heads, N, head_dim)
        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, head_dim)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)          # (4, B, heads, N, head_dim)
        q_sh, k_sh, v_ca, v_sa = qkvv.unbind(0)

        # Transpose to (B, heads, head_dim, N) for matrix multiply
        q_sh = q_sh.transpose(-2, -1)
        k_sh = k_sh.transpose(-2, -1)
        v_ca = v_ca.transpose(-2, -1)
        v_sa = v_sa.transpose(-2, -1)

        # Reduce sequence dim: (B, heads, head_dim, N) -> (B, heads, head_dim, proj_size)
        k_proj = self.E(k_sh)
        v_proj = self.F(v_sa)

        # L2-normalise shared Q/K for channel attention
        q_sh = F.normalize(q_sh, dim=-1)
        k_sh = F.normalize(k_sh, dim=-1)

        # ---- Channel Attention (CA) ----
        # Attend between channel features: (B, heads, head_dim, head_dim)
        attn_ca = self.drop_ca((q_sh @ k_sh.transpose(-2, -1) * self.temp_ca).softmax(dim=-1))
        # (B, heads, head_dim, N) → permute → (B, N, heads, head_dim) → (B, N, C)
        x_ca = (attn_ca @ v_ca).permute(0, 3, 1, 2).reshape(B, N, C)

        # ---- Spatial Attention (SA) ----
        # Attend across N tokens projected to proj_size: (B, heads, N, proj_size)
        attn_sa = self.drop_sa((q_sh.permute(0, 1, 3, 2) @ k_proj * self.temp_sa).softmax(dim=-1))
        # (B, heads, N, proj_size) @ (B, heads, proj_size, head_dim) → (B, N, C)
        x_sa = (attn_sa @ v_proj.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Project each branch C → C//2, then concat -> C
        return torch.cat([self.out_sa(x_sa), self.out_ca(x_ca)], dim=-1)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {'temp_ca', 'temp_sa'}


class _TransformerBlock(nn.Module):
    """Single UNETR++ transformer block.

    Sequence: LayerNorm → EPA → gamma-scaled residual → _ResBlock3D refinement.
    Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/transformerblock.py

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed: bool = False,
    ) -> None:
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")

        self.norm = nn.LayerNorm(hidden_size)
        
        # gamma: tiny learnable scale, starts near zero so residual branch is
        # almost a no-op at initialisation — aids stable early training.
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa = _EPA(input_size, hidden_size, proj_size, num_heads,
                        channel_attn_drop=dropout_rate, spatial_attn_drop=dropout_rate)
        # 3-D residual conv for spatial refinement after attention
        # Source: transformerblock.py L32: self.conv51 = UnetResBlock(3, ...)
        self.conv_refine = _ResBlock3D(hidden_size, hidden_size)
        # Final 1×1×1 projection with spatial dropout
        # Source: transformerblock.py L33: nn.Sequential(Dropout3d(0.1), Conv3d(..., 1))
        self.conv_proj = nn.Sequential(
            nn.Dropout3d(0.1, inplace=False),
            nn.Conv3d(hidden_size, hidden_size, 1),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, input_size, hidden_size)) if pos_embed else None

    def forward(self, x: Tensor) -> Tensor:
        B, C, D, H, W = x.shape

        # Flatten spatial dims -> sequence: (B, C, D, H, W) -> (B, D*H*W, C)
        # Equivalent to official: x.reshape(B, C, H*W*D).permute(0,2,1) for (B,C,H,W,D) layout
        x_seq = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

        if self.pos_emb is not None:
            x_seq = x_seq + self.pos_emb

        # EPA with learnable gamma residual
        x_seq = x_seq + self.gamma * self.epa(self.norm(x_seq))

        # Reshape back to spatial: (B, D*H*W, C) -> (B, C, D, H, W)
        attn_skip = x_seq.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # Conv refinement with skip connection
        return attn_skip + self.conv_proj(self.conv_refine(attn_skip))


class _UNETRPPEncoder(nn.Module):
    """Hierarchical UNETR++ encoder: stem + 3 downsample stages, each with EPA blocks.

    Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/model_components.py
    """

    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: int,
        dims: list[int],
        proj_sizes: list[int],
        depths: list[int],
        num_heads: int,
        transformer_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        D, H, W = img_size

        # Sequence lengths (N = D'*H'*W') at each stage, used for E/F projection
        # and positional embeddings inside EPA.
        self._input_sizes = [
            (D // 2) * (H // 4) * (W // 4),        # after stem   (÷2, ÷4, ÷4)
            (D // 4) * (H // 8) * (W // 8),          # after stage1 (÷2, ÷2, ÷2)
            (D // 8) * (H // 16) * (W // 16),        # after stage2
            (D // 16) * (H // 32) * (W // 32),       # after stage3
        ]

        # Downsampling convolutions
        # Stem: anisotropic (2,4,4) — source: model_components.py L14
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4), bias=False),
                nn.BatchNorm3d(dims[0]),
            )
        ])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=False),
                nn.BatchNorm3d(dims[i + 1]),
            ))

        # Transformer stages (one nn.Sequential of _TransformerBlock per stage)
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                _TransformerBlock(
                    self._input_sizes[i], dims[i], proj_sizes[i],
                    num_heads, dropout_rate=transformer_dropout, pos_embed=True,
                )
                for _ in range(depths[i])
            ])
            for i in range(4)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        from timm.models.layers import trunc_normal_
        
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Returns (bottleneck_seq, hidden_states).

        hidden_states[0..2] are spatial tensors (B, C, D', H', W').
        hidden_states[3] is a flattened sequence (B, N, C) ready for proj_feat.
        
        Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/model_components.py
        """
        hidden: list[Tensor] = []

        x = self.stages[0](self.downsample_layers[0](x))
        hidden.append(x)                              # enc1: (B, dims[0], D/2,  H/4,  W/4)

        for i in range(1, 4):
            x = self.stages[i](self.downsample_layers[i](x))
            if i == 3:
                # Flatten last stage to sequence for the decoder.
                # Official uses einops: "b c h w d -> b (h w d) c"
                # We replace with equivalent torch ops (no einops dependency).
                B, C, d, h, w = x.shape
                x = x.permute(0, 2, 3, 4, 1).reshape(B, d * h * w, C)  # (B, N, C)
            hidden.append(x)

        return x, hidden


class _UNETRPPDecoder(nn.Module):
    """Single UNETR++ decoder block.
    Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/model_components.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_kernel: tuple[int, ...] | int,
        out_size: int,
        num_heads: int = 4,
        proj_size: int = 64,
        depth: int = 3,
        conv_decoder: bool = False,
        dropout_rate: float = 0.15,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=upsample_kernel, stride=upsample_kernel,
            bias=False,
        )
        if conv_decoder:
            # At full resolution, self-attention is O(N²): use a ResBlock instead.
            self.refine: nn.Module = _ResBlock3D(out_channels, out_channels)
        else:
            self.refine = nn.Sequential(*[
                _TransformerBlock(out_size, out_channels, proj_size, num_heads,
                                  dropout_rate=dropout_rate, pos_embed=True)
                for _ in range(depth)
            ])

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.refine(self.up(x) + skip)


class _UNETRPPModel(nn.Module):
    """Full UNETR++ model (nn.Module, not Lightning).

    Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/unetr_pp_synapse.py

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: tuple[int, int, int],
        feature_size: int = 16,
        num_heads: int = 4,
        depths: list[int] | None = None,
        proj_sizes: list[int] | None = None,
        transformer_dropout: float = 0.15,
    ) -> None:
        
        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        if proj_sizes is None:
            proj_sizes = [64, 64, 64, 32]

        D, H, W = img_size

        # Channel counts per encoder stage (doubling each stage).
        dims = [feature_size * 2, feature_size * 4, feature_size * 8, feature_size * 16]

        # Spatial size at bottleneck (after all 4 downsampling steps).
        self.feat_size = (D // 2 // 8, H // 4 // 8, W // 4 // 8)
        self.hidden_size = dims[-1]
        self.num_classes = num_classes

        # encoder1: stride-1 ResBlock on raw input - adds a high-res skip at full resolution.
        self.encoder1 = _ResBlock3D(in_channels, feature_size)

        self.encoder = _UNETRPPEncoder(
            img_size=img_size,
            in_channels=in_channels,
            dims=dims,
            proj_sizes=proj_sizes,
            depths=depths,
            num_heads=num_heads,
            transformer_dropout=transformer_dropout,
        )

        # Sequence lengths at each decoder output (= spatial count after upsample)
        # Used to size TransformerBlock positional embeddings inside each decoder block
        dec_out_sizes = [
            (D // 8) * (H // 16) * (W // 16),   # decoder5 output = enc3 spatial
            (D // 4) * (H // 8) * (W // 8),      # decoder4 output = enc2 spatial
            (D // 2) * (H // 4) * (W // 4),      # decoder3 output = enc1 spatial
        ]

        # Source: unetr_pp_synapse.py L69-89
        self.decoder5 = _UNETRPPDecoder(dims[3], dims[2], (2, 2, 2), dec_out_sizes[0], num_heads, 64, 3)
        self.decoder4 = _UNETRPPDecoder(dims[2], dims[1], (2, 2, 2), dec_out_sizes[1], num_heads, 64, 3)
        self.decoder3 = _UNETRPPDecoder(dims[1], dims[0], (2, 2, 2), dec_out_sizes[2], num_heads, 64, 3)
        # decoder2: anisotropic upsample (2,4,4) mirrors the stem stride, conv_decoder=True.
        self.decoder2 = _UNETRPPDecoder(dims[0], feature_size, (2, 4, 4), 0, conv_decoder=True)

        self.out = nn.Conv3d(feature_size, num_classes, kernel_size=1)

    def _proj_feat(self, x: Tensor) -> Tensor:
        """Reshape bottleneck sequence (B, N, C) → spatial (B, C, D, H, W).
        Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/synapse/unetr_pp_synapse.py
        
        """
        B, N, C = x.shape
        D, H, W = self.feat_size
        return x.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        conv_block = self.encoder1(x)                   # (B, F,   D,    H,    W)

        _, hidden = self.encoder(x)
        enc1, enc2, enc3, enc4 = hidden                 # enc4 is (B, N, 8F) sequence

        dec4 = self._proj_feat(enc4)                    # (B, 8F,  D/16, H/32, W/32)

        dec3 = self.decoder5(dec4, enc3)                # (B, 4F,  D/8,  H/16, W/16)
        dec2 = self.decoder4(dec3, enc2)                # (B, 2F,  D/4,  H/8,  W/8)
        dec1 = self.decoder3(dec2, enc1)                # (B, F,   D/2,  H/4,  W/4)
        out  = self.decoder2(dec1, conv_block)          # (B, F,   D,    H,    W)

        return self.out(out)                            # (B, num_classes, D, H, W)


# ---------------------------------------------------------------------------
# Lightning module — exported
# ---------------------------------------------------------------------------

class UNETRPPModule(SegmentationModule):
    """Segmentation module wrapping UNETR++ for 3-D volumetric inputs.

    Differences from 2-D modules:
    - ``training_step``: random 3-D patch crop before forward pass.
    - ``validation_step`` / ``test_step``: sliding-window inference over
      full volumes with Gaussian importance weighting.

    Args:
        in_channels: Number of input image channels (e.g. 1 for CT, 4 for MRI).
        num_classes: Number of segmentation classes excluding background.
        img_size: Spatial size ``(D, H, W)`` of each training patch.
            Must equal ``patch_crop_size``. Fixed due to positional embeddings.
        feature_size: Base channel width.  Channel dims at each encoder stage
            are ``[2F, 4F, 8F, 16F]`` where ``F = feature_size``.
            Default 16 matches the Synapse/ACDC configs from the paper.
        num_heads: Number of attention heads in EPA.
        depths: Number of transformer blocks per encoder stage (list of 4).
        patch_crop_size: ``(D, H, W)`` patch randomly cropped during training.
            Must match ``img_size``.
        sw_overlap: Overlap ratio for sliding-window inference (0–1).
        loss_fn: Loss module.
        metrics_factories: ``{name: callable}`` where each callable returns a
            fresh :class:`torchmetrics.Metric`.
        class_names: Human-readable label for each class.
        lr: Learning rate for AdamW.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: tuple[int, int, int],
        loss_fn: nn.Module | None = None,
        metrics_factories: dict[str, Callable[[], Any]] = {},
        class_names: list[str] | None = None,
        lr: float = 1e-4,
        feature_size: int = 16,
        num_heads: int = 4,
        depths: list[int] | None = None,
        patch_crop_size: tuple[int, int, int] = (128, 128, 128),
        sw_overlap: float = 0.5,
        transform: A.BasicTransform | A.BaseCompose | None = None,
    ) -> None:
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.depths = depths or [3, 3, 3, 3]
        self.patch_crop_size = patch_crop_size
        self.sw_overlap = sw_overlap

        super().__init__(
            loss_fn=loss_fn,
            metrics_factories=metrics_factories,
            class_names=class_names,
            lr=lr,
            transform=transform,
        )

        self.model = _UNETRPPModel(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            feature_size=feature_size,
            num_heads=num_heads,
            depths=self.depths,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    # ---- Training: random crop before forward --------------------------------

    def _random_crop_3d(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Crop a random patch of ``patch_crop_size`` from image and mask.

        If any spatial dimension is smaller than the requested patch, the
        volume is zero-padded first so the output shape is always exactly
        ``patch_crop_size``.  Applied identically to both tensors.
        """
        pd, ph, pw = self.patch_crop_size
        D, H, W = image.shape[-3:]

        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        if pad_d or pad_h or pad_w:
            # F.pad takes (W_before, W_after, H_before, H_after, D_before, D_after)
            image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
            mask  = F.pad(mask,  (0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = image.shape[-3:]

        d0 = random.randint(0, D - pd)
        h0 = random.randint(0, H - ph)
        w0 = random.randint(0, W - pw)
        return (
            image[..., d0:d0 + pd, h0:h0 + ph, w0:w0 + pw],
            mask[..., d0:d0 + pd, h0:h0 + ph, w0:w0 + pw],
        )

    @staticmethod
    def _to_float_tensor(x: 'Tensor | np.ndarray') -> Tensor:
        """Convert numpy arrays to float tensors.

        ToTensorV2 in albumentations only converts the 'image' key, not the
        'volume' key used by MultiFrameDataset for 3D data. This ensures the
        model always receives proper tensors regardless of the transform output.
        """
        import numpy as np
        if isinstance(x, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(x)).float()
        return x.float()

    @override
    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        device = next(self.parameters()).device
        images = self._to_float_tensor(batch['image']).to(device)
        masks  = self._to_float_tensor(batch['masks'])[:, 1:].to(device)

        images, masks = self._random_crop_3d(images, masks)

        logits = self(images)
        loss = self.criterion(logits, masks) if self.criterion else None

        self.train_metrics.update((logits > 0).long(), masks.long())

        if loss is not None:
            self.log('train/loss', loss, on_step=True, on_epoch=True,
                     prog_bar=True, batch_size=images.shape[0])
        return loss

    # ---- Eval: sliding-window inference -------------------------------------

    @override
    def validation_step(self, batch: dict, batch_idx: int) -> Tensor | None:
        return self._eval_step(batch, 'val')

    @override
    def test_step(self, batch: dict, batch_idx: int) -> Tensor | None:
        return self._eval_step(batch, 'test')

    def _eval_step(self, batch: dict, stage: str) -> Tensor | None:
        device = next(self.parameters()).device
        images = self._to_float_tensor(batch['image']).to(device)
        masks  = self._to_float_tensor(batch['masks'])[:, 1:].to(device)

        logits = self._sliding_window_inference(images)
        loss = self.criterion(logits, masks) if self.criterion else None
        getattr(self, f'{stage}_metrics').update((logits > 0).long(), masks.long())

        if loss is not None:
            self.log(f'{stage}/loss', loss, on_epoch=True, prog_bar=True,
                     batch_size=images.shape[0])
        return loss

    def _sliding_window_inference(self, volume: Tensor) -> Tensor:
        """Tile-and-accumulate inference with Gaussian importance weighting.

        Gaussian weights assign higher confidence to the centre of each patch
        and downweight predictions near the borders, reducing stitching artifacts.
        
        Source: https://github.com/Amshaker/unetr_plus_plus/blob/main/unetr_pp/network_architecture/neural_network.py
        """
        device = next(self.parameters()).device
        volume = volume.float().to(device)
        B, C, D, H, W = volume.shape
        pd, ph, pw = self.patch_crop_size

        sd = max(1, int(pd * (1 - self.sw_overlap)))
        sh = max(1, int(ph * (1 - self.sw_overlap)))
        sw = max(1, int(pw * (1 - self.sw_overlap)))

        accum   = torch.zeros(B, self.num_classes, D, H, W, device=device)
        weights = torch.zeros(B, 1, D, H, W, device=device)
        gauss   = self._gaussian_kernel(self.patch_crop_size, device=device)

        def _tile_starts(total: int, patch: int, step: int) -> list[int]:
            starts = list(range(0, max(1, total - patch + 1), step))
            if not starts or starts[-1] + patch < total:
                starts.append(max(0, total - patch))
            return starts

        for d0 in _tile_starts(D, pd, sd):
            for h0 in _tile_starts(H, ph, sh):
                for w0 in _tile_starts(W, pw, sw):
                    patch = volume[:, :, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
                    pred = self(patch)
                    accum[:, :, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]   += pred * gauss
                    weights[:, :, d0:d0 + pd, h0:h0 + ph, w0:w0 + pw] += gauss

        return accum / weights.clamp(min=1e-6)

    @staticmethod
    def _gaussian_kernel(
        patch_size: tuple[int, int, int],
        sigma_factor: float = 0.125,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """3-D Gaussian kernel normalised to [0, 1] for patch importance weighting."""
        def _g1d(n: int) -> Tensor:
            x = torch.arange(n, device=device).float() - (n - 1) / 2.0
            sigma = n * sigma_factor
            return torch.exp(-x ** 2 / (2.0 * sigma ** 2))

        pd, ph, pw = patch_size
        kernel = _g1d(pd)[:, None, None] * _g1d(ph)[None, :, None] * _g1d(pw)[None, None, :]
        return (kernel / kernel.max()).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
