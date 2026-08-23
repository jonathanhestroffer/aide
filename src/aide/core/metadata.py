from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from aide.core.config.experiment import ExperimentConfig
from aide.core.environment import get_project_path


def _hash_bytes(value: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(value)
    return digest.hexdigest()


def _hash_file(path: Path | str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _hash_dict(value: object) -> str:
    normalized = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return _hash_bytes(normalized.encode("utf-8"))


def get_git_metadata(root: Path | None = None) -> dict[str, str | None]:
    """Retrieve the current Git commit hash and branch name for a given directory."""
    root = root or get_project_path()

    def _run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    # Let Git verify if the directory is inside a git repository
    if _run_git("rev-parse", "--is-inside-work-tree") != "true":
        return {
            "commit": None,
            "branch": None,
        }

    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def get_python_environment(root: Path | None = None) -> dict[str, object]:
    root = root or get_project_path()
    return {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "lockfile_hash": _hash_file(root / "uv.lock"),
        "aide_environment": {
            "AIDE_PROJECT_ROOT": os.environ.get("AIDE_PROJECT_ROOT"),
            "AIDE_CONFIGS_PATH": os.environ.get("AIDE_CONFIGS_PATH"),
        },
    }


def build_experiment_snapshot(
    config: ExperimentConfig, resolved_config: dict[str, object]
) -> dict[str, dict]:
    snapshot = {
        "git": get_git_metadata(),
        "environment": get_python_environment(),
        "config_version": _hash_dict(resolved_config),
    }
    return snapshot
