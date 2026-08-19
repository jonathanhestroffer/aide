from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

FRAMEWORK_PACKAGES = ["aide"]


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

    # Framework packages — always import these first.
    _walk_packages(FRAMEWORK_PACKAGES, imported_modules)

    # User-owned packages — can be anywhere on sys.path, no relation to AIDE.
    # these override any framework packages if there are name conflicts.
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
    """Import all modules under the given package names, recursively.

    This is useful for plugin discovery, where we want to ensure that all modules
    under a package are imported so that any decorators or registration logic is executed.

    Args:
        package_names (list[str]): List of package names to import.
        imported_modules (dict[str, ModuleType]): Dictionary to store imported modules.
    """
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

        for path_entry in package_path:
            for file_path in _iter_package_python_files(Path(path_entry)):
                module_name = _module_name_from_package(package_name, Path(path_entry), file_path)
                if module_name in imported_modules:
                    continue

                existing_module = sys.modules.get(module_name)
                if existing_module is not None:
                    imported_modules[module_name] = existing_module
                    continue

                module = importlib.import_module(module_name)
                imported_modules[module_name] = module


def _module_name_from_package(package_name: str, package_root: Path, file_path: Path) -> str:
    """Build module name for a file under a package root."""
    relative = file_path.relative_to(package_root)
    module_parts = list(relative.with_suffix("").parts)

    if module_parts and module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]

    if not module_parts:
        return package_name

    return ".".join([package_name, *module_parts])


def _iter_package_python_files(root: Path) -> Iterable[Path]:
    """Recursively yield Python files under a package path entry."""
    if not root.is_dir():
        return

    for file_path in sorted(root.rglob("*.py")):
        if "__pycache__" in file_path.parts:
            continue
        yield file_path


def _walk_python_files(plugin_dir: str, imported_modules: dict[str, ModuleType]) -> None:
    """Recursively import all Python files under the given directory.

    Args:
        plugin_dir (str): The directory containing Python files to import.
        imported_modules (dict[str, ModuleType]): Dictionary to store imported modules.
    """
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
    """Recursively yield all Python files under the given root directory.

    Args:
        root (Path): The root directory to search for Python files.

    Yields:
        Path: Paths to Python files under the root directory.
    """
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
    """Construct a module name from the root directory and file path.

    Args:
        root (Path): The root directory.
        file_path (Path): The file path to convert into a module name.

    Returns:
        str: The module name corresponding to the file path.
    """
    relative = file_path.relative_to(root)
    module_parts = list(relative.parts)
    module_parts[-1] = file_path.stem

    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]

    if not module_parts:
        return root.name

    return ".".join([root.name, *module_parts])


def _import_from_path(path: str, module_name: str | None = None) -> ModuleType:
    """Import a module from a given file path.

    Args:
        path (str): The file path to the Python module.
        module_name (str | None): The name to assign to the imported module.
            If None, the file stem is used.

    Returns:
        ModuleType: The imported module.

    Raises:
        ImportError: If the module cannot be imported from the given path.
    """
    file_path = Path(path).resolve()
    import_name = module_name or file_path.stem
    spec = importlib.util.spec_from_file_location(import_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load plugin file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[import_name] = module
    spec.loader.exec_module(module)
    return module
