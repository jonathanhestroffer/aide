from __future__ import annotations

import argparse
import subprocess
import sys

from aide.core.environment import get_config_path


def _normalize_argv(argv: list[str]) -> list[str]:
    """Treat direct Hydra args as the `train` subcommand for convenience."""

    if len(argv) <= 1:
        return argv

    first = argv[1]
    known = {
        "init",
        "list",
        "train",
        "--help",
        "-h",
    }
    if first in known:
        return argv

    return [argv[0], "train", *argv[1:]]


def _run_train(experiment: str, remaining_args: list[str]) -> int:
    """Run the `train` subcommand with the given experiment and remaining args."""

    config_path = get_config_path(".env")

    command = [
        sys.executable,
        "-m",
        "aide.scripts.train",
        f"--config-path={config_path}",
        f"--config-name=experiment/{experiment}",
        *remaining_args,
    ]

    print(f"Running: {' '.join(command)}")

    return subprocess.run(command).returncode


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    normalized = _normalize_argv(raw_argv)

    parser = argparse.ArgumentParser(description="CLI for running AIDE workflows")
    subparsers = parser.add_subparsers(dest="command")

    #
    # Subcommand: train
    #
    train_parser = subparsers.add_parser(
        "train",
        help="Run an AIDE experiment",
        description=(
            "Run an AIDE experiment by selecting a scaffolded Hydra experiment configuration."
        ),
    )

    train_parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name to run.",
    )

    train_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional Hydra arguments to pass through to the training runtime.",
    )

    #
    # Subcommand: init
    #
    init_parser = subparsers.add_parser(
        "init",
        help="Create a new experiment scaffold",
        description="Create a new AIDE experiment scaffold in the target directory.",
    )

    init_parser.add_argument(
        "args",
        metavar="<target-dir>",
        nargs=argparse.REMAINDER,
        help="Target directory for the new experiment scaffold.",
    )

    #
    # Subcommand: list
    #
    list_parser = subparsers.add_parser(
        "list",
        help="List registered registry entries",
        description="List registered models, components, or transforms for the current project.",
    )

    list_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the list command.",
    )

    parsed = parser.parse_args(normalized[1:])

    if parsed.command == "train":
        return _run_train(parsed.experiment, parsed.args)

    if parsed.command == "init":
        from aide.scripts.init import main as init_main

        return int(init_main(parsed.args))

    if parsed.command == "list":
        from aide.scripts.list import main as list_main

        return int(list_main(parsed.args))

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
