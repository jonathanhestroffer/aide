from __future__ import annotations

from typing import Any

from lightning.pytorch.loggers import MLFlowLogger


class MLFlowLoggerAdapter(MLFlowLogger):
    """Expose MLflow client methods directly on the Lightning logger."""

    def _resolve_run_id(self, run_id: str | None = None) -> str:
        active_run_id = run_id or self.run_id
        if not active_run_id:
            raise RuntimeError("MLflow run_id is not available yet. Start a training run first.")
        return active_run_id

    def log_param(self, key: str, value: Any, run_id: str | None = None) -> None:
        self.experiment.log_param(
            run_id=self._resolve_run_id(run_id),
            key=key,
            value=value,
        )

    def log_params(self, params: dict[str, Any], run_id: str | None = None) -> None:
        self.experiment.log_params(
            run_id=self._resolve_run_id(run_id),
            params=params,
        )

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.experiment.log_metric(
            run_id=self._resolve_run_id(run_id),
            key=key,
            value=value,
            step=step,
        )

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.experiment.log_artifact(
            run_id=self._resolve_run_id(run_id),
            local_path=local_path,
            artifact_path=artifact_path,
        )

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.experiment.log_artifacts(
            run_id=self._resolve_run_id(run_id),
            local_dir=local_dir,
            artifact_path=artifact_path,
        )

    def log_dict(
        self,
        dictionary: dict[str, Any],
        artifact_file: str,
        run_id: str | None = None,
    ) -> None:
        self.experiment.log_dict(
            run_id=self._resolve_run_id(run_id),
            dictionary=dictionary,
            artifact_file=artifact_file,
        )

    def set_tag(self, key: str, value: Any, run_id: str | None = None) -> None:
        self.experiment.set_tag(
            run_id=self._resolve_run_id(run_id),
            key=key,
            value=value,
        )

    def set_tags(self, tags: dict[str, Any], run_id: str | None = None) -> None:
        self.experiment.set_tags(
            run_id=self._resolve_run_id(run_id),
            tags=tags,
        )
