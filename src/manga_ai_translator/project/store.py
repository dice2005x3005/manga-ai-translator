"""Loading and saving projects on disk."""

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from manga_ai_translator import __version__
from manga_ai_translator._validation import format_validation_error
from manga_ai_translator.project.schema import SCHEMA_VERSION, Project

PROJECT_FILENAME = "project.json"
PROJECT_DIR_SUFFIX = ".mangat"

Migration = Callable[[dict[str, Any]], dict[str, Any]]
MIGRATIONS: dict[int, Migration] = {}
"""Upgrade functions keyed by the schema version they upgrade *from*."""

_REPLACE_ATTEMPTS = 5


class ProjectError(Exception):
    """Raised when a project cannot be read or written."""


def save_project(project: Project, project_dir: Path) -> Path:
    project.updated_at = datetime.now(UTC)
    project.app_version = __version__
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / PROJECT_FILENAME
    write_text_atomic(path, project.model_dump_json(indent=2) + "\n")
    return path


def load_project(project_dir: Path) -> Project:
    path = project_dir / PROJECT_FILENAME
    if not path.is_file():
        raise ProjectError(f"No project found in {project_dir} (missing {PROJECT_FILENAME})")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProjectError(f"{path} does not contain a project")

    data = migrate(data)
    try:
        return Project.model_validate(data)
    except ValidationError as exc:
        raise ProjectError(f"Invalid project file {path}:\n{format_validation_error(exc)}") from exc


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade raw project data from older schema versions to the current one."""
    version = data.get("schema_version")
    if not isinstance(version, int):
        raise ProjectError("Project file has no valid schema_version")
    if version > SCHEMA_VERSION:
        raise ProjectError(
            f"Project uses schema version {version}, but this mangat only understands up to "
            f"{SCHEMA_VERSION}. Update mangat to open it."
        )
    while version < SCHEMA_VERSION:
        upgrade = MIGRATIONS.get(version)
        if upgrade is None:
            raise ProjectError(f"No migration available from schema version {version}")
        data = upgrade(data)
        version += 1
        data["schema_version"] = version
    return data


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` so that ``path`` holds either the old or the new content, never a mix.

    The content goes to a temporary file in the same folder that then replaces the target.
    Replacing is retried briefly because antivirus or sync tools (e.g. OneDrive) can hold
    the target open for a moment on Windows.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def to_project_path(path: Path, project_dir: Path) -> str:
    """Express ``path`` relative to the project folder when possible, so projects can move."""
    absolute = path.resolve()
    try:
        return Path(os.path.relpath(absolute, project_dir.resolve())).as_posix()
    except ValueError:  # on another drive (Windows): keep it absolute
        return absolute.as_posix()


def resolve_project_path(stored: str, project_dir: Path) -> Path:
    path = Path(stored)
    return path if path.is_absolute() else (project_dir / path).resolve()
