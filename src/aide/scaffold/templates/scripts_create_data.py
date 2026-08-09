from __future__ import annotations

import argparse
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

        np_image = np_image.copy()
        tensor_image = torch.from_numpy(np_image).permute(2, 0, 1).to(torch.uint8)
        images.append(tensor_image)
        labels.append(int(label))

    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.int64))


def _split_cifar_train(
    dataset: CIFAR10, train_size: int, val_size: int, seed: int
) -> tuple[TensorDataset, TensorDataset]:
    if train_size + val_size > len(dataset):
        raise ValueError(
            "Requested train_size + val_size exceeds available CIFAR-10 training samples (50_000)."
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]

    return (
        _tensor_dataset_from_cifar(dataset, train_indices),
        _tensor_dataset_from_cifar(dataset, val_indices),
    )


def _tensor_dataset_from_cifar_test(dataset: CIFAR10, test_size: int) -> TensorDataset:
    if test_size > len(dataset):
        raise ValueError("Requested test_size exceeds available CIFAR-10 test samples (10_000).")

    indices = list(range(test_size))
    return _tensor_dataset_from_cifar(dataset, indices)


def create_dataset_artifacts(
    output_dir: Path,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    num_classes: int,
    seed: int,
) -> Path:
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_source = CIFAR10(raw_dir, train=True, download=True)
    test_source = CIFAR10(raw_dir, train=False, download=True)

    train_dataset, val_dataset = _split_cifar_train(train_source, train_size, val_size, seed)
    test_dataset = _tensor_dataset_from_cifar_test(test_source, test_size)

    train_path = output_dir / "train.pt"
    val_path = output_dir / "val.pt"
    test_path = output_dir / "test.pt"

    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)

    manifest = {
        "train": train_path.name,
        "val": val_path.name,
        "test": test_path.name,
        "meta": {
            "dataset": "cifar10",
            "num_train": train_size,
            "num_val": val_size,
            "num_test": test_size,
            "num_classes": 10,
            "seed": seed,
        },
    }

    manifest_path = output_dir / "datasets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CIFAR-10 dataset artifacts with train/val/test splits"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--train-size",
        type=int,
        default=45000,
        help="Number of training examples to keep from the CIFAR-10 training split",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="Number of validation examples carved from the CIFAR-10 training split",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10000,
        help="Number of test examples to use from the CIFAR-10 test split",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = create_dataset_artifacts(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    print(f"Wrote dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
