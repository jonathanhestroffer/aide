"""Scaffold layout definitions for `ml-platform init`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaffoldFile:
    """Single file to generate in an experiment scaffold."""

    relative_path: str
    template_name: str


DEFAULT_LAYOUT: tuple[ScaffoldFile, ...] = (
    ScaffoldFile(relative_path="configs/config.yaml", template_name="config.yaml"),
    ScaffoldFile(
        relative_path="configs/experiment/default.yaml",
        template_name="config_experiment_default.yaml",
    ),
    ScaffoldFile(relative_path="configs/model/cnn.yaml", template_name="config_model_cnn.yaml"),
    ScaffoldFile(
        relative_path="configs/datamodule/artifact.yaml",
        template_name="config_datamodule_artifact.yaml",
    ),
    ScaffoldFile(
        relative_path="configs/trainer/default.yaml",
        template_name="config_trainer_default.yaml",
    ),
    ScaffoldFile(
        relative_path="configs/infrastructure/local.yaml",
        template_name="config_infrastructure_local.yaml",
    ),
    ScaffoldFile(
        relative_path="configs/checkpoint/default.yaml",
        template_name="config_checkpoint_default.yaml",
    ),
    ScaffoldFile(relative_path=".env", template_name="env"),
    ScaffoldFile(relative_path="train.py", template_name="train_py"),
    ScaffoldFile(relative_path="plugins/__init__.py", template_name="plugins__init__.py"),
    ScaffoldFile(
        relative_path="plugins/models/__init__.py",
        template_name="plugins_models__init__.py",
    ),
    ScaffoldFile(relative_path="plugins/models/cnn.py", template_name="plugins_models_cnn.py"),
    ScaffoldFile(
        relative_path="plugins/components/__init__.py",
        template_name="plugins_components__init__.py",
    ),
    ScaffoldFile(
        relative_path="plugins/components/transforms.py",
        template_name="plugins_components_transforms.py",
    ),
)
