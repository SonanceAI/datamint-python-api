"""EfficientNetV2 image classification trainer."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
from typing_extensions import override

from ..classification_trainer import ImageClassificationTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose


class EfficientNetV2Trainer(ImageClassificationTrainer):
    """Trainer pre-configured for EfficientNetV2.

    Default model: **EfficientNetV2-S** pretrained on ImageNet at 384×384.
    
    Args:
        model_name: ``timm`` model name. Defaults to ``'efficientnetv2_s'``.
            Other valid choices: ``'efficientnetv2_m'``, ``'efficientnetv2_l'``,
            ``'efficientnetv2_xl'``, ``'efficientnetv2_rw_t'``.
        image_size: Target image size ``(H, W)`` or a single int for square
            images. Defaults to ``384``, the resolution EfficientNetV2-S was
            trained at.

    Example::

        trainer = EfficientNetV2Trainer(project='ChestXray')
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        model_name: str = 'efficientnetv2_s',
        image_size: int | tuple[int, int] = 384,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, image_size=image_size, **kwargs)

    @override
    def _train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            self._build_resize_transform(),
            A.ToRGB(),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
