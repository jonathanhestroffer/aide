from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

# Only the framework's own internal registries — always safe to import.
FRAMEWORK_PACKAGES = [
    "aide.models",
    "aide.components",
]


def discover_and_import(
    user_packages: list[str] | None = None,
    user_files: list[str] | None = None,
    user_plugin_dirs: list[str] | None = None,
) -> dict[str, ModuleType]:
    """Import framework packages plus user-specified plugin sources.

    Decorator-based registration happens at import time, so importing every
    module under each package (or each file directly) populates registries.
    """
    imported_modules: dict[str, ModuleType] = {}

    # Always pick up the framework's own built-in components.
    _walk_packages(FRAMEWORK_PACKAGES, imported_modules)

    # User-owned packages — can be anywhere on sys.path, no relation to AIDE.
    if user_packages:
        _walk_packages(user_packages, imported_modules)

    # User-owned single files — no packaging required at all.
    if user_files:
        for file_path in user_files:
            module = _import_from_path(file_path)
            imported_modules[module.__name__] = module

    # User-owned local plugin directories (recursively import *.py).
    if user_plugin_dirs:
        for plugin_dir in user_plugin_dirs:
            _walk_python_files(plugin_dir, imported_modules)

    return imported_modules


def _walk_packages(package_names: list[str], imported_modules: dict[str, ModuleType]) -> None:
    for package_name in package_names:
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError as exc:
            if exc.name == package_name.split(".", maxsplit=1)[0]:
                continue
            raise

        imported_modules[package_name] = package

        package_path = getattr(package, "__path__", None)
        if package_path is None:
            continue

        for module_info in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
            module = importlib.import_module(module_info.name)
            imported_modules[module_info.name] = module


def _walk_python_files(plugin_dir: str, imported_modules: dict[str, ModuleType]) -> None:
    root = Path(plugin_dir).resolve()
    if not root.is_dir():
        return

    for file_path in _iter_python_files(root):
        module_name = _module_name_from_root(root, file_path)
        if module_name in imported_modules:
            continue

        existing_module = sys.modules.get(module_name)
        if existing_module is not None:
            imported_modules[module_name] = existing_module
            continue

        module = _import_from_path(str(file_path), module_name=module_name)
        imported_modules[module.__name__] = module


def _iter_python_files(root: Path) -> Iterable[Path]:
    # Import root-level __init__.py first so `plugins.*` names resolve predictably.
    root_init = root / "__init__.py"
    if root_init.is_file():
        yield root_init

    for file_path in sorted(root.rglob("*.py")):
        if file_path.name == "__init__.py":
            if file_path == root_init:
                continue
        if "__pycache__" in file_path.parts:
            continue
        yield file_path


def _module_name_from_root(root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(root)
    module_parts = list(relative.parts)
    module_parts[-1] = file_path.stem

    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]

    if not module_parts:
        return root.name

    return ".".join([root.name, *module_parts])


def _import_from_path(path: str, module_name: str | None = None) -> ModuleType:
    file_path = Path(path).resolve()
    import_name = module_name or file_path.stem
    spec = importlib.util.spec_from_file_location(import_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load plugin file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[import_name] = module
    spec.loader.exec_module(module)
    return module
