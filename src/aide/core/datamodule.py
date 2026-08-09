from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import lightning as L
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from aide.components.factory import build_transform_component
from aide.core.config import DataModuleConfig
from aide.core.config.component import ComponentConfig


def _resolve_artifact_uri(artifact_uri: str) -> Path:

    parsed = urlparse(artifact_uri)

    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path or artifact_uri).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Artifact location does not exist: {artifact_uri}")
        return path

    raise ValueError(
        f"Unsupported artifact URI scheme: '{parsed.scheme}'. "
        "Use local path, file://, http://, or https://"
    )


def _to_dataset(obj: Any, *, source: Path) -> Dataset:
    if isinstance(obj, Dataset):
        return obj

    raise TypeError(
        "Artifact is not a torch Dataset. "
        f"Source: {source}. Expected Dataset, got: {type(obj).__name__}"
    )


def _load_dataset_from_path(path: Path) -> Dataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _to_dataset(payload, source=path)


def _apply_transforms(
    *,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset | None,
    transforms: list[ComponentConfig],
) -> tuple[Dataset, Dataset, Dataset | None]:
    if not transforms:
        return train_dataset, val_dataset, test_dataset

    split_datasets: list[Dataset] = [train_dataset, val_dataset]
    has_test = test_dataset is not None
    if has_test:
        split_datasets.append(test_dataset)

    for transform_config in transforms:
        transform = build_transform_component(transform_config)
        transformed = transform(split_datasets)

        if not isinstance(transformed, list):
            raise TypeError(
                "Transform components must return a list of datasets when given split datasets"
            )

        if len(transformed) != len(split_datasets):
            raise ValueError(
                "Transform component changed split count. "
                f"Expected {len(split_datasets)}, got {len(transformed)}"
            )

        for index, dataset in enumerate(transformed):
            if not isinstance(dataset, Dataset):
                raise TypeError(
                    f"Transform output at index {index} is not a Dataset: {type(dataset).__name__}"
                )

        split_datasets = transformed

    updated_train = split_datasets[0]
    updated_val = split_datasets[1]
    updated_test = split_datasets[2] if has_test else None
    return updated_train, updated_val, updated_test


class ArtifactDataModule(L.LightningDataModule):
    """Build a LightningDataModule from train/val/test dataset artifacts."""

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset | None = None,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    @classmethod
    def from_artifact_uri(
        cls,
        artifact_uri: str,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> "ArtifactDataModule":
        """Load datasets from a JSON manifest artifact."""
        resolved = _resolve_artifact_uri(artifact_uri)

        if resolved.suffix.lower() != ".json":
            raise ValueError(
                f"File artifact must be a JSON manifest containing split paths. Got: {resolved}"
            )

        manifest = json.loads(resolved.read_text(encoding="utf-8"))
        for key in ("train", "val"):
            if key not in manifest:
                raise KeyError(f"Manifest missing required split key: '{key}'")

        base_dir = resolved.parent
        train_dataset = _load_dataset_from_path(base_dir / manifest["train"])
        val_dataset = _load_dataset_from_path(base_dir / manifest["val"])
        test_dataset = None
        if "test" in manifest and manifest["test"]:
            test_dataset = _load_dataset_from_path(base_dir / manifest["test"])

        return cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is None:
            return []
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


def build_datamodule(config: DataModuleConfig) -> ArtifactDataModule:
    """Create an artifact-backed datamodule from typed experiment config."""
    num_workers = config.num_workers
    if num_workers is None:
        num_workers = os.cpu_count() or 0

    datamodule = ArtifactDataModule.from_artifact_uri(
        config.artifact_uri,
        batch_size=config.batch_size,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )

    train_dataset, val_dataset, test_dataset = _apply_transforms(
        train_dataset=datamodule.train_dataset,
        val_dataset=datamodule.val_dataset,
        test_dataset=datamodule.test_dataset,
        transforms=config.transforms or [],
    )
    datamodule.train_dataset = train_dataset
    datamodule.val_dataset = val_dataset
    datamodule.test_dataset = test_dataset

    return datamodule
