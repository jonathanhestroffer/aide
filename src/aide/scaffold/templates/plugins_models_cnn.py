from __future__ import annotations

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import (
    OptimizerLRScheduler,
)

from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


class ConvBlock(nn.Module):
    """A simple convolutional block with Conv2d, BatchNorm2d, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Classifier(nn.Module):
    """A simple classifier with a linear layer."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@ModelRegistry.register("scaffold_cnn")
class ScaffoldCNN(TrainableModel):
    """Small CNN used by the generated scaffold example."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, lr: float = 1e-3) -> None:
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            Classifier(128 * 8 * 8, num_classes),
        )

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "train_acc",
            (preds.argmax(dim=1) == targets).float().mean(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, targets = batch
        preds = self(images)
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_acc",
            (preds.argmax(dim=1) == targets).float().mean(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
