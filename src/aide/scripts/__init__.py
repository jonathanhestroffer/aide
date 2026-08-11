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
        "train",
        "init",
        "list",
        "--help",
        "-h",
    }
    if first in known:
        return argv

    return [argv[0], "train", *argv[1:]]


def _run_train(args: list[str]) -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)

    parsed, remaining = parser.parse_known_args(args)

    config_path = get_config_path(".env")

    command = [
        sys.executable,
        "-m",
        "aide.scripts.train",
        f"--config-path={config_path}",
        f"--config-name=experiment/{parsed.experiment}",
        *remaining,
    ]

    print(f"Running: {' '.join(command)}")

    return subprocess.run(command).returncode


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    normalized = _normalize_argv(raw_argv)

    parser = argparse.ArgumentParser(description="CLI for running AIDE workflows")
    parser.add_argument("command", nargs="?", choices=["train", "init", "list"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(normalized[1:])

    if parsed.command == "train":
        return _run_train(parsed.args)

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
