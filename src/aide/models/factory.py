from aide.components.factory import build_component
from aide.core.config.trainable import TrainableConfig
from aide.core.trainable import TrainableModel
from aide.registry.registries import ModelRegistry


def build_trainable_model(config: TrainableConfig) -> TrainableModel:
    """
    Build a trainable model instance from the given model class and arguments.

    Args:
        config (TrainableConfig): The configuration for the trainable model.
    Returns:
        TrainableModel: An instance of the specified trainable model class.
    """
    model_cfg = config.model
    preprocessor_cfg = config.preprocessor
    postprocessor_cfg = config.postprocessor

    cls = ModelRegistry.get(model_cfg.class_name)
    model = cls(**model_cfg.params)

    if preprocessor_cfg is not None:
        model.preprocessor = build_component(preprocessor_cfg)

    if postprocessor_cfg is not None:
        model.postprocessor = build_component(postprocessor_cfg)

    return model
