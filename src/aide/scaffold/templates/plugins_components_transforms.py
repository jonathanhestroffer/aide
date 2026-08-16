from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from aide.core.components import TransformComponent
from aide.registry.registries import TransformRegistry

_VALID_SPLITS = {"train", "val", "test", "all"}


class _MappedDataset(Dataset):
    """Dataset wrapper that transforms only the input tensor."""

    def __init__(self, dataset: Dataset, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[no-any-return]

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if isinstance(sample, tuple):
            image, *rest = sample
            return (self.transform(image), *rest)
        if isinstance(sample, list):
            image, *rest = sample
            return [self.transform(image), *rest]
        return self.transform(sample)


class _SplitAwareTransform(TransformComponent):
    """Apply tensor transforms only to selected splits."""

    def __init__(self, *, apply_to: list[str] | None = None) -> None:
        super().__init__()
        requested = [split.lower() for split in (apply_to or ["all"])]
        invalid = sorted(set(requested) - _VALID_SPLITS)
        if invalid:
            valid = ", ".join(sorted(_VALID_SPLITS))
            invalid_text = ", ".join(invalid)
            raise ValueError(f"Invalid apply_to split(s): {invalid_text}. Valid values: {valid}")
        self.apply_to = set(requested)

    def _should_apply(self, split_name: str) -> bool:
        return "all" in self.apply_to or split_name in self.apply_to

    def _transform_image(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, dataset: Dataset | list[Dataset]) -> Dataset | list[Dataset]:
        if isinstance(dataset, list):
            split_names = ("train", "val", "test")
            transformed: list[Dataset] = []
            for index, ds in enumerate(dataset):
                split_name = split_names[index] if index < len(split_names) else "all"
                if self._should_apply(split_name):
                    transformed.append(_MappedDataset(ds, self._transform_image))
                else:
                    transformed.append(ds)
            return transformed

        if self._should_apply("all"):
            return _MappedDataset(dataset, self._transform_image)
        return dataset


@TransformRegistry.register("scaffold_normalize")
class NormalizeToUnitRange(_SplitAwareTransform):
    """Convert uint8 image tensors to float tensors in [0, 1]."""

    def __init__(self, *, apply_to: list[str] | None = None) -> None:
        super().__init__(apply_to=apply_to or ["all"])

    def _transform_image(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image tensor, got: {type(image).__name__}")

        if image.is_floating_point():
            return image

        return image.float().div(255.0)


@TransformRegistry.register("scaffold_random_crop")
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

    def _transform_image(self, image: torch.Tensor) -> torch.Tensor:
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
