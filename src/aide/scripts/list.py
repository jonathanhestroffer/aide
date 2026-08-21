from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from aide.registry.registries import DatasetRegistry, ModelRegistry, TransformRegistry
from aide.utils.plugins import load_plugins


def _format_registry_source(obj: object) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"

    name = getattr(obj, "__name__", None)
    if name:
        return name

    if inspect.ismodule(obj):
        return getattr(obj, "__file__", repr(obj))

    return repr(obj)


def _print_registry(name: str, entries: list[tuple[str, object]]) -> None:
    print(f"{name} ({len(entries)})")
    for key, obj in sorted(entries, key=lambda item: item[0]):
        print(f"  - {key} -> {_format_registry_source(obj)}")
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
    registries: dict[str, list[tuple[str, object]]] = {
        "models": [(key, ModelRegistry.get(key)) for key in ModelRegistry.keys()],
        "datasets": [(key, DatasetRegistry.get(key)) for key in DatasetRegistry.keys()],
        "transforms": [(key, TransformRegistry.get(key)) for key in TransformRegistry.keys()],
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
    parser = argparse.ArgumentParser(description="List registered model/dataset/transform keys")
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
