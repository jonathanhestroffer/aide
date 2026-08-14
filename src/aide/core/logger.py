from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only


class MLFlowLoggerAdapter(MLFlowLogger):
    """Expose MLflow client methods directly on the Lightning logger."""

    def _resolve_run_id(self, run_id: str | None = None) -> str:
        active_run_id = run_id or self.run_id
        if not active_run_id:
            raise RuntimeError("MLflow run_id is not available yet. Start a training run first.")
        return active_run_id

    @rank_zero_only
    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log a single artifact to MLflow."""

        self.experiment.log_artifact(
            run_id=self._resolve_run_id(run_id),
            local_path=local_path,
            artifact_path=artifact_path,
        )

    @rank_zero_only
    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Log multiple artifacts from a directory to MLflow."""

        self.experiment.log_artifacts(
            run_id=self._resolve_run_id(run_id),
            local_dir=local_dir,
            artifact_path=artifact_path,
        )

    @rank_zero_only
    def log_dict(
        self,
        dictionary: dict[str, Any],
        artifact_file: str,
        run_id: str | None = None,
    ) -> None:
        """Log a dictionary as a JSON artifact to MLflow."""

        self.experiment.log_dict(
            run_id=self._resolve_run_id(run_id),
            dictionary=dictionary,
            artifact_file=artifact_file,
        )

    @rank_zero_only
    def log_tensor(
        self,
        tensor: torch.Tensor,
        artifact_file: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Save a tensor as a PyTorch artifact and log it to MLflow."""

        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / artifact_file

            torch.save(
                tensor.detach().cpu(),
                file_path,
            )

            self.log_artifact(
                local_path=str(file_path),
                artifact_path=artifact_path,
                run_id=run_id,
            )

    @rank_zero_only
    def log_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        artifact_file: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Save multiple tensors as a single PyTorch artifact and log it to MLflow."""

        tensors_cpu = {name: tensor.detach().cpu() for name, tensor in tensors.items()}

        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / artifact_file

            torch.save(
                tensors_cpu,
                file_path,
            )

            self.log_artifact(
                local_path=str(file_path),
                artifact_path=artifact_path,
                run_id=run_id,
            )
