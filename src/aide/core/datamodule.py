from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from aide.components.factory import build_dataset, build_transform
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
        transform = build_transform(transform_config)
        if transform is None:
            continue
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

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()
        self.train_dataset = build_dataset(config.train_dataset)
        self.val_dataset = build_dataset(config.val_dataset)
        self.test_dataset = None

        if config.test_dataset:
            self.test_dataset = build_dataset(config.test_dataset)

        global_dataloader = config.global_dataloader

        self.train_dataloader_params = global_dataloader.override_with(config.train_dataloader)
        self.val_dataloader_params = global_dataloader.override_with(config.val_dataloader)
        self.test_dataloader_params = global_dataloader.override_with(config.test_dataloader)

        print(self.train_dataloader_params)
        print(self.val_dataloader_params)
        print(self.test_dataloader_params)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, **self.train_dataloader_params.model_dump())

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.val_dataset is None:
            return []
        return DataLoader(self.val_dataset, **self.val_dataloader_params.model_dump())

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is None:
            return []
        return DataLoader(self.test_dataset, **self.test_dataloader_params.model_dump())


def build_datamodule(config: DataModuleConfig) -> ArtifactDataModule:
    """Create an artifact-backed datamodule from typed experiment config."""

    datamodule = ArtifactDataModule(config=config)

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
