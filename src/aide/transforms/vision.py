from __future__ import annotations

import torch
from torchvision.transforms import functional as TF

from aide.registry.registries import TransformRegistry
from aide.utils.transforms import _SplitAwareTransform


@TransformRegistry.register("AIDE_normalize")
class NormalizeToUnitRange(_SplitAwareTransform):
    """Convert uint8 image tensors to float tensors in [0, 1]."""

    def __init__(self, *, apply_to: list[str] | None = None) -> None:
        super().__init__(apply_to=apply_to or ["all"])

    def _transform(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image tensor, got: {type(image).__name__}")

        if image.is_floating_point():
            return image

        return image.float().div(255.0)


@TransformRegistry.register("AIDE_random_crop")
class RandomCrop(_SplitAwareTransform):
    """Random crop with optional padding for training augmentation."""

    def __init__(
        self,
        *,
        size: int = 32,
        padding: int = 4,
        apply_to: list[str] | None = None,
    ) -> None:
        super().__init__(apply_to=apply_to or ["train"])
        self.size = size
        self.padding = padding

    def _transform(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image tensor, got: {type(image).__name__}")

        if self.padding > 0:
            image = TF.pad(image, [self.padding] * 4, fill=0)

        _, height, width = image.shape
        if height < self.size or width < self.size:
            raise ValueError(f"Crop size {self.size} is larger than image size {height}x{width}")

        top = int(torch.randint(0, height - self.size + 1, (1,)).item())
        left = int(torch.randint(0, width - self.size + 1, (1,)).item())
        return TF.crop(image, top, left, self.size, self.size)
