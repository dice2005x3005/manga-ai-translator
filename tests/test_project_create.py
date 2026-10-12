from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from manga_ai_translator.cli import app
from manga_ai_translator.fileutils import sha256_file
from manga_ai_translator.imaging import ImageError
from manga_ai_translator.project import ProjectError, load_project
from manga_ai_translator.project.create import (
    SourceStatus,
    check_sources,
    create_project,
    default_project_dir,
)

runner = CliRunner()


@pytest.fixture
def chapter(tmp_path: Path) -> Path:
    """A chapter folder with three synthetic pages whose names need natural sorting."""
    folder = tmp_path / "chapter01"
    folder.mkdir()
    Image.new("L", (600, 900), 255).save(folder / "page10.png")
    Image.new("RGB", (600, 900), (250, 250, 250)).save(folder / "page2.jpg")
    Image.new("L", (900, 600), 255).save(folder / "page1.webp", lossless=True)
    (folder / "credits.txt").write_text("not an image", encoding="utf-8")
    return folder


def test_create_project_from_folder(chapter: Path) -> None:
    calls: list[tuple[int, int]] = []
    project, project_dir = create_project(
        chapter, progress=lambda done, total, _: calls.append((done, total))
    )

    assert project_dir == chapter.resolve().with_name("chapter01.mangat")
    assert project.name == "chapter01"
    assert project.source_dir == "../chapter01"
    assert [p.source.path for p in project.pages] == [
        "../chapter01/page1.webp",
        "../chapter01/page2.jpg",
        "../chapter01/page10.png",
    ]
    first = project.pages[0].source
    # WebP has no grayscale mode, so Pillow stores the page as RGB.
    assert (first.width, first.height, first.mode, first.format) == (900, 600, "RGB", "WEBP")
    assert first.sha256 == sha256_file(chapter / "page1.webp")
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert load_project(project_dir) == project


def test_default_project_dir_is_a_sibling(tmp_path: Path) -> None:
    assert default_project_dir(tmp_path / "ch5") == (tmp_path / "ch5.mangat").resolve()


def test_custom_dir_and_name(chapter: Path, tmp_path: Path) -> None:
    project, project_dir = create_project(chapter, tmp_path / "work" / "x", name="Vol 1 Ch 1")
    assert project_dir == (tmp_path / "work" / "x").resolve()
    assert project.name == "Vol 1 Ch 1"
    assert project.pages[0].source.path == "../../chapter01/page1.webp"


def test_existing_project_needs_overwrite(chapter: Path) -> None:
    create_project(chapter)
    with pytest.raises(ProjectError, match="already exists"):
        create_project(chapter)
    create_project(chapter, overwrite=True)


def test_folder_without_images(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="No supported images"):
        create_project(tmp_path)


def test_corrupt_image_is_reported(chapter: Path) -> None:
    (chapter / "page3.png").write_bytes(b"garbage")
    with pytest.raises(ImageError, match=r"page3\.png"):
        create_project(chapter)


def test_check_sources_detects_missing_and_modified(chapter: Path) -> None:
    project, project_dir = create_project(chapter)
    (chapter / "page1.webp").unlink()
    Image.new("L", (600, 900), 0).save(chapter / "page10.png")

    assert check_sources(project, project_dir, verify_hash=False) == [
        SourceStatus.MISSING,
        SourceStatus.OK,
        SourceStatus.OK,
    ]
    assert check_sources(project, project_dir, verify_hash=True) == [
        SourceStatus.MISSING,
        SourceStatus.OK,
        SourceStatus.MODIFIED,
    ]


def test_cli_new_and_info(chapter: Path) -> None:
    result = runner.invoke(app, ["project", "new", str(chapter)])
    assert result.exit_code == 0, result.output
    assert "3 pages" in result.output

    project_dir = chapter.with_name("chapter01.mangat")
    result = runner.invoke(app, ["project", "info", str(project_dir), "--verify"])
    assert result.exit_code == 0, result.output
    assert "page10.png" in result.output
    assert "match the project" in result.output


def test_cli_new_fails_cleanly_on_existing_project(chapter: Path) -> None:
    runner.invoke(app, ["project", "new", str(chapter)])
    result = runner.invoke(app, ["project", "new", str(chapter)])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_cli_info_reports_missing_pages(chapter: Path) -> None:
    runner.invoke(app, ["project", "new", str(chapter)])
    (chapter / "page2.jpg").unlink()
    result = runner.invoke(app, ["project", "info", str(chapter.with_name("chapter01.mangat"))])
    assert result.exit_code == 0
    assert "missing" in result.output


def test_cli_info_without_project(tmp_path: Path) -> None:
    result = runner.invoke(app, ["project", "info", str(tmp_path)])
    assert result.exit_code == 1
    assert "No project found" in result.output
