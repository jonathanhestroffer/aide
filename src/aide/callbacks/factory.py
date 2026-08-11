from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

from aide.core.config.experiment import ExperimentConfig


class ArtifactModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that uploads saved checkpoints to artifact storage after training."""

    def __init__(self, *args, artifact_path: str | None = "checkpoints", **kwargs):
        # self._artifact_path = artifact_path
        super().__init__(*args, **kwargs)

    def on_train_end(self, trainer, pl_module) -> None:
        super().on_train_end(trainer, pl_module)

        ckpt_dir = Path(self.dirpath or ".")
        # if not ckpt_dir.exists() or self._artifact_path is None:
        # return

        logger = getattr(trainer, "logger", None)
        if logger is None or not hasattr(logger, "log_artifacts"):
            return

        try:
            logger.log_artifacts(
                local_dir=str(ckpt_dir),
                # artifact_path=self._artifact_path,
                artifact_path="checkpoints",
            )
        except Exception:
            # Avoid interrupting training cleanup if artifact logging fails.
            pass


def build_callbacks(cfg: ExperimentConfig) -> list:
    """Builds a list of callbacks based on the provided experiment configuration.

    Currently, this function only supports the ModelCheckpoint callback, but it can be extended
    to include other callbacks in the future.
    """

    callbacks = []

    # Add ModelCheckpoint callback
    ckpt = cfg.checkpoint

    if ckpt.dirpath is None:
        save_dir = cfg.infrastructure.save_dir or "."
        ckpt.dirpath = str(
            (Path(save_dir).expanduser().resolve() / cfg.metadata.name / "checkpoints")
        )

    if ckpt.enabled:
        callbacks.append(
            ArtifactModelCheckpoint(
                dirpath=ckpt.dirpath,
                filename=ckpt.filename,
                monitor=ckpt.monitor,
                mode=ckpt.mode,
                save_last=ckpt.save_last,
                save_top_k=ckpt.save_top_k,
                every_n_epochs=ckpt.every_n_epochs,
                every_n_train_steps=ckpt.every_n_train_steps,
                artifact_path="checkpoints",
            )
        )

    return callbacks
