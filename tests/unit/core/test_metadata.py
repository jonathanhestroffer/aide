from __future__ import annotations

import os
from pathlib import Path

from aide.core.config import (
    CheckpointConfig,
    ComponentConfig,
    DataModuleConfig,
    ExperimentConfig,
    ExperimentMetadata,
    InfrastructureConfig,
    TrainerConfig,
)
from aide.core.metadata import build_experiment_snapshot


def test_build_experiment_snapshot_includes_required_metadata(tmp_path: Path):
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text('{"dataset": "test"}\n', encoding="utf-8")

    os.environ["AIDE_DATASET_MANIFEST"] = str(manifest_path)
    os.environ["PYTHONHASHSEED"] = "0"

    config = ExperimentConfig(
        metadata=ExperimentMetadata(name="test", description="metadata test"),
        model=ComponentConfig(class_name="test_model"),
        datamodule=DataModuleConfig(
            train_dataset=ComponentConfig(class_name="procedural_shapes"),
            val_dataset=ComponentConfig(class_name="procedural_shapes"),
        ),
        trainer=TrainerConfig(),
        checkpoint=CheckpointConfig(),
        infrastructure=InfrastructureConfig(),
    )

    resolved = config.model_dump(exclude_none=True)
    snapshot = build_experiment_snapshot(config, resolved)

    assert "git" in snapshot
    assert "environment" in snapshot
    assert "config_version" in snapshot

    assert isinstance(snapshot["config_version"], str)
