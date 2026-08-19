from aide.core.config.component import ComponentConfig
from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


def build_trainable_model(config: ComponentConfig) -> TrainableModel:
    """
    Build a trainable model instance from the given model class and arguments.

    Args:
        config (ComponentConfig): The configuration for the trainable model.
    Returns:
        TrainableModel: An instance of the specified trainable model class.
    """
    cls = ModelRegistry.get(config.class_name)
    return cls(**config.params)


def load_trainable_model(model: TrainableModel, checkpoint_path: str) -> TrainableModel:
    """
    Load a trainable model from a checkpoint.

    Args:
        model (TrainableModel): The model instance to load the checkpoint into.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        TrainableModel: The model instance with loaded weights.
    """
    return model.load_from_checkpoint(checkpoint_path)
