from __future__ import annotations

import torch
from torch.utils.data import Dataset

from aide.core.components import TransformComponent
from aide.utils.datasets import _MappedDataset

_VALID_SPLITS = {"train", "val", "test", "all"}


class _SplitAwareTransform(TransformComponent):
    """Apply tensor transforms only to selected splits.

    Args:
        apply_to (list[str] | None): List of dataset splits to apply the transform to.
            Valid values are "train", "val", "test", and "all". Defaults to ["all"].
    """

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

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, dataset: Dataset | list[Dataset]) -> Dataset | list[Dataset]:
        """Apply the transform to the given dataset or list of datasets.

        This method will apply the transform only to the splits specified in `apply_to`.
        If a list of datasets is provided, it will match the datasets to the splits
        in the order: "train", "val", "test".

        If the dataset is not a list, it will be treated as a single dataset
        corresponding to the "all" split.

        Args:
            dataset (Dataset | list[Dataset]): The dataset or list of datasets to transform.

        Returns:
            Dataset | list[Dataset]: The transformed dataset or list of datasets.
        """
        if isinstance(dataset, list):
            split_names = ("train", "val", "test")
            transformed: list[Dataset] = []
            for index, ds in enumerate(dataset):
                split_name = split_names[index] if index < len(split_names) else "all"
                if self._should_apply(split_name):
                    transformed.append(_MappedDataset(ds, self._transform))
                else:
                    transformed.append(ds)
            return transformed

        if self._should_apply("all"):
            return _MappedDataset(dataset, self._transform)
        return dataset
