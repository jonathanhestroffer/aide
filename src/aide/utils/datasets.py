from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset


class _MappedDataset(Dataset):
    """Dataset wrapper that transforms only the input tensor.

    Args:
        dataset (Dataset): The original dataset to wrap.
        transform (Callable[[torch.Tensor], torch.Tensor]): A function to apply to the input tensor.
    """

    def __init__(self, dataset: Dataset, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[no-any-return]

    def __getitem__(self, index: int) -> Any:
        """Retrieve an item from the dataset and apply the transform to the input tensor.

        Supports samples in the form of tuples, lists, or dictionaries.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Any: The transformed item from the dataset.
        """
        sample = self.dataset[index]
        if isinstance(sample, tuple):
            image, *rest = sample
            return (self.transform(image), *rest)
        if isinstance(sample, list):
            image, *rest = sample
            return [self.transform(image), *rest]
        if isinstance(sample, dict):
            sample = sample.copy()
            sample["inputs"] = self.transform(sample["inputs"])
            return sample
        return self.transform(sample)
