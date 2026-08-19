"""Simple `aide init` command wrapper around scaffold generation."""

from __future__ import annotations

import argparse

from aide.scaffold.generator import ScaffoldResult, scaffold_experiment


def _print_scaffold_result(result: ScaffoldResult) -> None:
    """Prints the result of the scaffold generation to the console."""

    for rel_path in result.created:
        print(f"created: {rel_path}")
    for rel_path in result.overwritten:
        print(f"overwritten: {rel_path}")
    for rel_path in result.skipped:
        print(f"skip (exists): {rel_path}")

    print(f"\nDone. Next steps:\n  cd {result.base_path}\n  aide train --experiment default")


def aide_init_command(
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
        target_dir=target_dir,
        overwrite=overwrite,
    )
    _print_scaffold_result(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize an AIDE experiment scaffold")

    parser.add_argument(
        "target_dir",
        metavar="<target_dir>",
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
        target_dir=args.target_dir,
        overwrite=args.overwrite,
        artifact_dir=args.artifact_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
