import lightning as L

from aide.callbacks.factory import build_callbacks
from aide.core.config import ExperimentConfig, InfrastructureConfig, TrainerConfig
from aide.core.logger import MLFlowLoggerAdapter
from aide.core.trainable import TrainableModel


class Trainer:
    """Wrapper around Lightning Trainer and MLflow logger.

    This class owns both the configured `MLFlowLoggerAdapter` and the
    `lightning.Trainer` instance used for training.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:

        infra_config: InfrastructureConfig = cfg.infrastructure
        trainer_config: TrainerConfig = cfg.trainer

        self.logger = MLFlowLoggerAdapter(
            experiment_name="ml-platform",
            tracking_uri=infra_config.tracking_uri,
            artifact_location=infra_config.artifact_location,
        )

        trainer_kwargs = trainer_config.model_dump(exclude_none=True)

        self.trainer = L.Trainer(
            logger=self.logger,
            callbacks=build_callbacks(cfg),
            default_root_dir=infra_config.save_dir,
            **trainer_kwargs,
        )

        self.experiment_config = cfg.model_dump(exclude_none=True)

    def fit(
        self,
        module: TrainableModel,
        datamodule: L.LightningDataModule,
    ) -> None:

        self.logger.log_dict(
            self.experiment_config,
            artifact_file="experiment_config.json",
        )

        self.trainer.fit(
            module,
            datamodule=datamodule,
        )
