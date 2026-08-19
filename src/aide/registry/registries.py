from __future__ import annotations

from torch.utils.data import Dataset

from aide.core.components import TransformComponent
from aide.core.registry import Registry
from aide.core.trainable import TrainableModel

ModelRegistry = Registry[type[TrainableModel]](
    "models", expected_type=TrainableModel, allow_override=True
)

TransformRegistry = Registry[type[TransformComponent]](
    "transforms", expected_type=TransformComponent, allow_override=True
)

DatasetRegistry = Registry[type[Dataset]]("datasets", expected_type=Dataset, allow_override=True)
