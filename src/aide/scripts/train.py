from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from aide.components.factory import build_trainable
from aide.core.config import ExperimentConfig
from aide.core.datamodule import build_datamodule
from aide.core.trainer import Trainer
from aide.utils.plugins import load_plugins


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    config_data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_data, dict):
        raise TypeError("Hydra config must resolve to a mapping for ExperimentConfig")

    experiment_config = ExperimentConfig.model_validate(config_data)

    load_plugins()

    model = build_trainable(experiment_config.model)
    datamodule = build_datamodule(experiment_config.datamodule)

    trainer = Trainer(experiment_config)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
