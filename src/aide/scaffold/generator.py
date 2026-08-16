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


def _write_scaffold_file(
    base_path: Path,
    file_spec: ScaffoldFile,
    *,
    overwrite: bool = False,
) -> tuple[bool, bool]:
    full_path = base_path / file_spec.relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    existed = full_path.exists()
    if existed and not overwrite:
        return False, True

    template = _read_template(file_spec.template_name)
    full_path.write_text(template, encoding="utf-8")
    return True, existed


def write_dataset_manifest_uri(base_path: Path, artifact_uri: str) -> None:
    """Set the generated project's dataset manifest location in its environment file."""
    env_path = base_path / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines()
    updated_lines = [
        f"AIDE_DATASET_MANIFEST={artifact_uri}"
        if line.startswith("AIDE_DATASET_MANIFEST=")
        else line
        for line in lines
    ]
    env_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def scaffold_experiment(
    target_dir: str,
    overwrite: bool = False,
) -> ScaffoldResult:
    """Create an experiment scaffold, optionally overwriting existing generated files."""
    root = Path(target_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    created: list[str] = []
    overwritten: list[str] = []
    skipped: list[str] = []

    for file_spec in DEFAULT_LAYOUT:
        did_write, existed = _write_scaffold_file(
            root,
            file_spec,
            overwrite=overwrite,
        )
        if did_write:
            if existed:
                overwritten.append(str(file_spec.relative_path))
            else:
                created.append(str(file_spec.relative_path))
        else:
            skipped.append(str(file_spec.relative_path))

    return ScaffoldResult(
        base_path=root,
        created=tuple(created),
        overwritten=tuple(overwritten),
        skipped=tuple(skipped),
    )
