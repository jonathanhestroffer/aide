"""Simple `aide init` command wrapper around scaffold generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from aide.scaffold.dataset import create_cifar10_artifacts
from aide.scaffold.generator import ScaffoldResult, scaffold_experiment, write_dataset_manifest_uri


def _print_scaffold_result(result: ScaffoldResult, manifest_path: str) -> None:
    """Prints the result of the scaffold generation to the console."""

    for rel_path in result.created:
        print(f"created: {rel_path}")
    for rel_path in result.overwritten:
        print(f"overwritten: {rel_path}")
    for rel_path in result.skipped:
        print(f"skip (exists): {rel_path}")

    print(
        "\nDone. Next steps:"
        "\n  cd "
        f"{result.base_path}"
        f"\n  CIFAR-10 manifest: {manifest_path}"
        "\n  aide train --experiment default"
    )


def aide_init_command(
    experiment_name: str,
    target_dir: str = ".",
    overwrite: bool = False,
    artifact_dir: str | None = None,
) -> int:
    """Create an experiment scaffold with sibling configs and plugins directories.

    Args:
        experiment_name (str): Name of the experiment directory to create.
        target_dir (str): Parent directory where the experiment directory is created.
        overwrite (bool): Whether to overwrite existing scaffold files.
        artifact_dir (str | None): Directory that will contain cifar10/manifest.json
            and the split artifacts. Defaults to the current working directory.

    Returns:
        int: Exit code (0 for success).
    """
    result = scaffold_experiment(
        experiment_name=experiment_name,
        target_dir=target_dir,
        overwrite=overwrite,
    )
    resolved_artifact_dir = artifact_dir or str(Path.cwd())
    manifest_path = create_cifar10_artifacts(resolved_artifact_dir)
    write_dataset_manifest_uri(result.base_path, str(manifest_path))
    _print_scaffold_result(result, str(manifest_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize an AIDE experiment scaffold")

    parser.add_argument(
        "experiment_name",
        help="Experiment directory name to create (contains configs/ and plugins/)",
    )

    parser.add_argument(
        "--target-dir",
        default=".",
        help="Parent directory where the experiment directory is created",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated scaffold files that already exist",
    )

    parser.add_argument(
        "--artifact-dir",
        help=(
            "Directory that will contain cifar10/manifest.json and the split artifacts "
            "(defaults to the current working directory)"
        ),
    )

    args = parser.parse_args(argv)

    return aide_init_command(
        args.experiment_name,
        target_dir=args.target_dir,
        overwrite=args.overwrite,
        artifact_dir=args.artifact_dir,
    )


def run_aide_init() -> None:
    """Console entrypoint for the standalone mlp-init command."""
    parser = argparse.ArgumentParser(description="Create scaffold files in a target directory")

    parser.add_argument(
        "target_dir",
        nargs="?",
        default=".",
        help="Directory where scaffold files should be created",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated scaffold files that already exist",
    )

    parser.add_argument(
        "--artifact-dir",
        help=(
            "Directory that will contain cifar10/manifest.json and the split artifacts "
            "(defaults to the current working directory)"
        ),
    )

    args = parser.parse_args()

    aide_init_command(
        experiment_name="default",
        target_dir=args.target_dir,
        overwrite=args.overwrite,
        artifact_dir=args.artifact_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
