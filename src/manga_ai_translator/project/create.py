"""Creating a project from a folder of page images and checking those images later."""

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from manga_ai_translator import __version__
from manga_ai_translator.fileutils import sha256_file
from manga_ai_translator.imaging import list_images, read_image_info
from manga_ai_translator.project.schema import Page, Project, SourceImage
from manga_ai_translator.project.store import (
    PROJECT_DIR_SUFFIX,
    PROJECT_FILENAME,
    ProjectError,
    resolve_project_path,
    save_project,
    to_project_path,
)

ProgressCallback = Callable[[int, int, Path], None]
"""Called as ``(done, total, path)`` after each page is read."""


class SourceStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    MODIFIED = "modified"


def default_project_dir(images_dir: Path) -> Path:
    """``chapter01/`` -> ``chapter01.mangat/`` next to it, so the source folder is untouched."""
    images_dir = images_dir.resolve()
    return images_dir.with_name(images_dir.name + PROJECT_DIR_SUFFIX)


def create_project(
    images_dir: Path,
    project_dir: Path | None = None,
    name: str | None = None,
    *,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> tuple[Project, Path]:
    """Build and save a project with one page per image in ``images_dir``.

    Returns the project and the folder it was saved in.
    """
    images_dir = images_dir.resolve()
    project_dir = (project_dir or default_project_dir(images_dir)).resolve()
    if (project_dir / PROJECT_FILENAME).exists() and not overwrite:
        raise ProjectError(f"A project already exists in {project_dir}. Use --force to replace it.")

    images = list_images(images_dir)
    if not images:
        raise ProjectError(f"No supported images (JPG, PNG, WEBP) found in {images_dir}")

    pages = []
    for index, path in enumerate(images):
        info = read_image_info(path)
        source = SourceImage(
            path=to_project_path(path, project_dir),
            sha256=sha256_file(path),
            width=info.width,
            height=info.height,
            mode=info.mode,
            format=info.format,
        )
        pages.append(Page(index=index, source=source))
        if progress is not None:
            progress(index + 1, len(images), path)

    now = datetime.now(UTC)
    project = Project(
        name=name or images_dir.name,
        app_version=__version__,
        created_at=now,
        updated_at=now,
        source_dir=to_project_path(images_dir, project_dir),
        pages=pages,
    )
    save_project(project, project_dir)
    return project, project_dir


def check_sources(project: Project, project_dir: Path, *, verify_hash: bool) -> list[SourceStatus]:
    """Status of each page's original image. Hashing every file is optional because it is slower."""
    statuses = []
    for page in project.pages:
        path = resolve_project_path(page.source.path, project_dir)
        if not path.is_file():
            statuses.append(SourceStatus.MISSING)
        elif verify_hash and sha256_file(path) != page.source.sha256:
            statuses.append(SourceStatus.MODIFIED)
        else:
            statuses.append(SourceStatus.OK)
    return statuses
