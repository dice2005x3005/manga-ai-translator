import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from manga_ai_translator.project import (
    PROJECT_FILENAME,
    Page,
    Project,
    ProjectError,
    SourceImage,
    Stage,
    StageStatus,
    TextBlock,
    load_project,
    resolve_project_path,
    save_project,
    store,
    to_project_path,
)


def make_page(index: int = 0, blocks: list[TextBlock] | None = None) -> Page:
    source = SourceImage(
        path=f"../raw/{index + 1:03}.png",
        sha256="0" * 64,
        width=800,
        height=1200,
        mode="L",
        format="PNG",
    )
    return Page(index=index, source=source, blocks=blocks or [])


def make_block(block_id: str = "p000-b00", **fields: object) -> TextBlock:
    return TextBlock(id=block_id, order=0, bbox=(10, 20, 110, 220), **fields)


def make_project(pages: list[Page] | None = None) -> Project:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Project(
        name="chapter-01",
        app_version="0.0.0",
        created_at=now,
        updated_at=now,
        source_dir="../raw",
        pages=pages if pages is not None else [make_page()],
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    block = make_block(source_text="こんにちは", translation="Hello")
    project = make_project([make_page(0, [block]), make_page(1)])
    path = save_project(project, tmp_path / "proj")

    assert path.name == PROJECT_FILENAME
    assert "こんにちは" in path.read_text(encoding="utf-8")  # readable, not \u-escaped
    assert load_project(tmp_path / "proj") == project


def test_save_updates_timestamp_and_version(tmp_path: Path) -> None:
    project = make_project()
    before = project.updated_at
    save_project(project, tmp_path)
    assert project.updated_at > before
    assert project.app_version != "0.0.0"


def test_failed_save_keeps_previous_file_and_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "proj"
    project = make_project()
    path = save_project(project, project_dir)
    original = path.read_text(encoding="utf-8")

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store.os, "replace", broken_replace)
    project.name = "renamed"
    with pytest.raises(OSError, match="disk full"):
        save_project(project, project_dir)

    assert path.read_text(encoding="utf-8") == original
    assert [p.name for p in project_dir.iterdir()] == [PROJECT_FILENAME]


def test_replace_is_retried_when_file_is_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_replace = store.os.replace
    calls = []

    def flaky_replace(src: str, dst: str) -> None:
        calls.append(dst)
        if len(calls) == 1:
            raise PermissionError("locked by another process")
        real_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky_replace)
    store.write_text_atomic(tmp_path / "f.txt", "content")
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "content"
    assert len(calls) == 2


def test_load_missing_project(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="No project found"):
        load_project(tmp_path)


def test_load_corrupt_json(tmp_path: Path) -> None:
    (tmp_path / PROJECT_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectError, match="Cannot read"):
        load_project(tmp_path)


def test_load_invalid_content_reports_location(tmp_path: Path) -> None:
    data = json.loads(make_project().model_dump_json())
    data["pages"][0]["source"]["width"] = -5
    (tmp_path / PROJECT_FILENAME).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProjectError, match=r"pages\.0\.source\.width"):
        load_project(tmp_path)


def test_newer_schema_is_rejected() -> None:
    with pytest.raises(ProjectError, match="Update mangat"):
        store.migrate({"schema_version": 99})


def test_migrations_run_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(
        store,
        "MIGRATIONS",
        {1: lambda d: d | {"steps": ["1->2"]}, 2: lambda d: d | {"steps": [*d["steps"], "2->3"]}},
    )
    assert store.migrate({"schema_version": 1}) == {"schema_version": 3, "steps": ["1->2", "2->3"]}


def test_missing_migration_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "SCHEMA_VERSION", 2)
    with pytest.raises(ProjectError, match="No migration"):
        store.migrate({"schema_version": 1})


def test_missing_stages_are_filled_as_pending() -> None:
    data = make_page().model_dump()
    data["stages"] = {"detection": "done"}
    page = Page.model_validate(data)
    assert page.stages[Stage.DETECTION] is StageStatus.DONE
    assert page.stages[Stage.TYPESETTING] is StageStatus.PENDING
    assert set(page.stages) == set(Stage)


@pytest.mark.parametrize("bbox", [(10, 10, 5, 20), (10, 10, 20, 10), (-1, 0, 5, 5)])
def test_invalid_bbox(bbox: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValidationError, match="bbox"):
        TextBlock(id="b", order=0, bbox=bbox)


def test_duplicate_block_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate block ids: b1"):
        make_page(blocks=[make_block("b1"), make_block("b1")])


def test_page_indices_must_be_sequential() -> None:
    with pytest.raises(ValidationError, match="page indices"):
        make_project([make_page(0), make_page(2)])


def test_block_lookup() -> None:
    page = make_page(blocks=[make_block("a"), make_block("b")])
    assert page.block("b").id == "b"
    with pytest.raises(KeyError):
        page.block("zzz")


def test_pipeline_update_keeps_manual_edits() -> None:
    block = make_block(translation="machine")
    block.apply_user_edit(translation="Fixed by hand")
    assert block.user_edited.translation

    kept = block.apply_pipeline_update(
        translation="new machine output", source_text="こんにちは", bbox=(0, 0, 50, 50)
    )
    assert kept == ["translation"]
    assert block.translation == "Fixed by hand"
    assert block.source_text == "こんにちは"
    assert block.bbox == (0, 0, 50, 50)


def test_geometry_edit_protects_all_geometry_fields() -> None:
    block = make_block()
    block.apply_user_edit(bbox=(1, 1, 30, 30))
    assert block.apply_pipeline_update(polygon=[(0.0, 0.0)], bubble=None) == ["polygon", "bubble"]


def test_edits_are_validated() -> None:
    block = make_block()
    with pytest.raises(ValidationError):
        block.apply_pipeline_update(bbox=(5, 5, 1, 1))
    with pytest.raises(ValueError, match="not an editable"):
        block.apply_user_edit(id="other")


def test_project_paths_are_relative_and_resolvable(tmp_path: Path) -> None:
    project_dir = tmp_path / "chapter.mangat"
    image = tmp_path / "chapter" / "001.png"
    stored = to_project_path(image, project_dir)
    assert stored == "../chapter/001.png"
    assert resolve_project_path(stored, project_dir) == image.resolve()


def test_absolute_project_paths_are_kept(tmp_path: Path) -> None:
    absolute = (tmp_path / "x.png").resolve()
    assert resolve_project_path(absolute.as_posix(), tmp_path / "proj") == absolute
