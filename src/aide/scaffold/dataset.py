"""CIFAR-10 artifact creation used by the experiment scaffold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10


def _tensor_dataset_from_cifar(dataset: CIFAR10, indices: list[int]) -> TensorDataset:
    images: list[torch.Tensor] = []
    labels: list[int] = []

    for index in indices:
        image, label = dataset[index]
        np_image = np.asarray(image)
        if np_image.ndim != 3 or np_image.shape[2] != 3:
            raise ValueError(f"Expected CIFAR image with 3 channels, got shape {np_image.shape}")

        tensor_image = torch.from_numpy(np_image.copy()).permute(2, 0, 1).to(torch.uint8)
        images.append(tensor_image)
        labels.append(int(label))

    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.int64))


def create_cifar10_artifacts(artifact_dir: str | Path, *, seed: int = 42) -> Path:
    """Create or reuse CIFAR-10 artifacts under ``<artifact_dir>/cifar10``."""
    output_dir = Path(artifact_dir).expanduser().resolve() / "cifar10"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        return manifest_path

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"

    train_source = CIFAR10(raw_dir, train=True, download=True)
    test_source = CIFAR10(raw_dir, train=False, download=True)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    train_dataset = _tensor_dataset_from_cifar(train_source, indices[:45000])
    val_dataset = _tensor_dataset_from_cifar(train_source, indices[45000:50000])
    test_dataset = _tensor_dataset_from_cifar(test_source, list(range(10000)))

    paths = {
        "train": output_dir / "train.pt",
        "val": output_dir / "val.pt",
        "test": output_dir / "test.pt",
    }
    torch.save(train_dataset, paths["train"])
    torch.save(val_dataset, paths["val"])
    torch.save(test_dataset, paths["test"])

    manifest_path.write_text(
        json.dumps(
            {
                "train": paths["train"].name,
                "val": paths["val"].name,
                "test": paths["test"].name,
                "meta": {
                    "dataset": "cifar10",
                    "num_train": 45000,
                    "num_val": 5000,
                    "num_test": 10000,
                    "num_classes": 10,
                    "seed": seed,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path
