from __future__ import annotations

import argparse
from pathlib import Path

from aide.registry.registries import DatasetRegistry, ModelRegistry, TransformRegistry
from aide.utils.plugins import load_plugins


def _print_registry(name: str, keys: list[str]) -> None:
    print(f"{name} ({len(keys)})")
    for key in sorted(keys):
        print(f"  - {key}")
    print()


def list_registry(project_path: str, kind: str = "all") -> int:
    """List currently registered model/component/transform keys."""
    project_root = Path(project_path).resolve()
    configs_dir = project_root / "configs"
    plugins_dir = project_root / "plugins"
    if not project_root.is_dir() or not configs_dir.is_dir() or not plugins_dir.is_dir():
        raise ValueError(
            "Project must be initialized and include sibling 'configs/' and 'plugins/' directories."
        )

    load_plugins(None, project_root=project_root)

    normalized = kind.strip().lower()
    registries: dict[str, list[str]] = {
        "models": ModelRegistry.keys(),
        "datasets": DatasetRegistry.keys(),
        "transforms": TransformRegistry.keys(),
    }

    if normalized == "all":
        for registry_name in ("models", "datasets", "transforms"):
            _print_registry(registry_name, registries[registry_name])
        return 0

    if normalized not in registries:
        valid = ", ".join(["all", *registries.keys()])
        raise ValueError(f"Unknown --kind '{kind}'. Expected one of: {valid}")

    _print_registry(normalized, registries[normalized])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List registered model/component/transform keys")
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to an initialized experiment directory (must contain configs/ and plugins/)",
    )
    parser.add_argument(
        "--kind",
        "-k",
        default="all",
        help="Registry to list: all, models, datasets, transforms",
    )
    args = parser.parse_args(argv)
    return list_registry(args.project_path, kind=args.kind)


if __name__ == "__main__":
    raise SystemExit(main())
