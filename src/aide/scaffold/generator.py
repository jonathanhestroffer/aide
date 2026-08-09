"""Scaffold file generator for new experiment projects."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from aide.scaffold.layout import DEFAULT_LAYOUT, ScaffoldFile


@dataclass(frozen=True)
class ScaffoldResult:
    """Outcome of scaffold generation."""

    base_path: Path
    created: tuple[str, ...]
    overwritten: tuple[str, ...]
    skipped: tuple[str, ...]


def _read_template(template_name: str) -> str:
    template_path = resources.files("aide.scaffold.templates") / template_name
    return template_path.read_text(encoding="utf-8")


def _render_template(template: str, *, experiment_name: str) -> str:
    # Use explicit token replacement to avoid clobbering braces used by Hydra and Python.
    return template.replace("{{experiment_name}}", experiment_name)


def _write_scaffold_file(
    base_path: Path,
    file_spec: ScaffoldFile,
    *,
    experiment_name: str,
    overwrite: bool = False,
) -> tuple[bool, bool]:
    full_path = base_path / file_spec.relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    existed = full_path.exists()
    if existed and not overwrite:
        return False, True

    template = _read_template(file_spec.template_name)
    rendered = _render_template(template, experiment_name=experiment_name)
    full_path.write_text(rendered, encoding="utf-8")
    return True, existed


def scaffold_experiment(
    experiment_name: str,
    target_dir: str = ".",
    overwrite: bool = False,
) -> ScaffoldResult:
    """Create an experiment scaffold, optionally overwriting existing generated files."""
    root = Path(target_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    base_path = root / experiment_name
    base_path.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    overwritten: list[str] = []
    skipped: list[str] = []

    for file_spec in DEFAULT_LAYOUT:
        did_write, existed = _write_scaffold_file(
            base_path,
            file_spec,
            experiment_name=experiment_name,
            overwrite=overwrite,
        )
        if did_write:
            if existed:
                overwritten.append(file_spec.relative_path)
            else:
                created.append(file_spec.relative_path)
        else:
            skipped.append(file_spec.relative_path)

    return ScaffoldResult(
        base_path=base_path,
        created=tuple(created),
        overwritten=tuple(overwritten),
        skipped=tuple(skipped),
    )
