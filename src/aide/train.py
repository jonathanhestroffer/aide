from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from aide.core.config import ExperimentConfig
from aide.core.datamodule import build_datamodule
from aide.core.trainer import Trainer
from aide.models.factory import build_trainable_model
from aide.utils.plugins import load_plugins


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run a training job from Hydra config.

    Defaults to packaged config (aide/configs/train.yaml), but supports
    overrides via --config-path/--config-name and standard Hydra CLI options.
    """
    config_data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_data, dict):
        raise TypeError("Hydra config must resolve to a mapping for ExperimentConfig")

    experiment_config = ExperimentConfig.model_validate(config_data)

    load_plugins(experiment_config.infrastructure.plugins)

    model = build_trainable_model(experiment_config.trainable)
    datamodule = build_datamodule(experiment_config.datamodule)

    trainer = Trainer(experiment_config)
    trainer.fit(
        model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
