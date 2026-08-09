from abc import ABC, abstractmethod

import lightning as L

from aide.core.components import Component, Identity


class TrainableModel(L.LightningModule, ABC):
    """
    Base class for all trainable models in the ML platform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor: Component = Identity()
        self.postprocessor: Component = Identity()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the model. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    @abstractmethod
    def training_step(self, *args, **kwargs):
        """
        Training step of the model. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Training step must be implemented by subclasses.")

    def validation_step(self, *args, **kwargs):
        """
        Validation step of the model. This method should be overridden by subclasses.
        """
        pass

    def predict_step(self, *args, **kwargs):
        """
        Prediction step of the model. This method should be overridden by subclasses.
        """
        pass

    @abstractmethod
    def configure_optimizers(self, *args, **kwargs):
        """
        Configure the optimizers for the model. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Configure optimizers method must be implemented by subclasses.")
