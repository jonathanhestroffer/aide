import subprocess
from pathlib import Path

import mlflow


def test_init_and_train_workflow(tmp_path: Path):
    # define target directory in a temporary location
    project_dir = tmp_path / "demo_project"

    # aide init <target-dir>
    init_res = subprocess.run(["aide", "init", str(project_dir)], capture_output=True, text=True)

    assert init_res.returncode == 0, f"aide init failed with error:\n{init_res.stderr}"

    # execute aide train
    # passing cwd=project_dir
    train_res = subprocess.run(
        [
            "aide",
            "train",
            "--experiment",
            "default",
        ],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )

    assert train_res.returncode == 0, f"aide train failed with error:\n{train_res.stderr}"

    # verify that checkpoints were created in the expected location
    checkpoints_dir = project_dir / "workspace" / "lightning_logs" / "default" / "checkpoints"
    assert checkpoints_dir.exists(), "Checkpoints directory was not created"
    assert any(checkpoints_dir.iterdir()), "No checkpoint files were created"

    # verify that hydra config files were created
    hydra_logs_dir = project_dir / "workspace" / "hydra_logs"
    assert hydra_logs_dir.exists(), "Hydra logs directory was not created"

    # verify that the MLflow tracking database was created
    mlflow_tracking_uri = project_dir / "workspace" / "metadata" / "mlflow.db"
    assert mlflow_tracking_uri.exists(), "MLflow tracking database was not created"

    # verify that the MLflow run was logged
    # and model achieved
    runs = mlflow.search_runs(
        experiment_names=["default"],
    )
