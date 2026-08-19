from torch.utils.data import Dataset

from aide.core.components import TransformComponent
from aide.core.config.component import ComponentConfig
from aide.core.trainable import TrainableModel
from aide.registry.registries import DatasetRegistry, ModelRegistry, TransformRegistry


def build_dataset(config: ComponentConfig) -> Dataset:
    """
    Build a dataset component instance from the given dataset class and arguments.

    Args:
        config (ComponentConfig): The configuration for the dataset component.
    Returns:
        Dataset: An instance of the specified dataset component class.
    """
    cls = DatasetRegistry.get(config.class_name)
    return cls(**config.params)


def build_transform(config: ComponentConfig) -> TransformComponent:
    """
    Build a transform component instance from the given transform class and arguments.

    Args:
        config (ComponentConfig): The configuration for the transform component.
    Returns:
        TransformComponent: An instance of the specified transform component class.
    """
    cls = TransformRegistry.get(config.class_name)
    return cls(**config.params)


def build_trainable(config: ComponentConfig) -> TrainableModel:
    """
    Build a trainable model instance from the given model class and arguments.

    Args:
        config (ComponentConfig): The configuration for the trainable model.
    Returns:
        TrainableModel: An instance of the specified trainable model class.
    """
    cls = ModelRegistry.get(config.class_name)
    return cls(**config.params)
