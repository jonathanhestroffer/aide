from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from torch.utils.data import Dataset


class Component(nn.Module, ABC):
    """
    Base class for all components in the ML platform.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the component.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class TransformComponent(Component):
    """
    A specialized component that transforms data before configuring DataLoaders.
    Must return a Dataset or a list of Datasets in the forward pass.
    """

    def forward(self, dataset: Dataset | list[Dataset]) -> Dataset | list[Dataset]:
        """
        Transform the dataset.
        """
        ...


class Identity(Component):
    """
    A component that performs the identity operation.
    """

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """
        Forward pass of the component.
        """
        return x
