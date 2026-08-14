from abc import ABC, abstractmethod
from typing import Any

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler


class TrainableModel(L.LightningModule, ABC):
    """Base class for all trainable models in the ML platform."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the model. This method should be overridden by subclasses."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    @abstractmethod
    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        """Training step of the model. This method should be overridden by subclasses."""
        raise NotImplementedError("Training step must be implemented by subclasses.")

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs) -> OptimizerLRScheduler:
        """Configure the optimizers for the model.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Configure optimizers method must be implemented by subclasses.")

    def validation_step(self, *args, **kwargs) -> Any:
        """Validation step of the model. This method should be overridden by subclasses."""
        pass
