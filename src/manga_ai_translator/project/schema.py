"""Data model of a translation project: one chapter, its pages and their text blocks.

A project is saved as ``project.json`` inside its own folder. Original images are never
modified: pages reference them by path (relative to the project folder when possible) and
record their SHA-256 hash so changes on disk can be detected.
"""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
"""Bump when the JSON layout changes incompatibly, and add a migration in ``store``."""


class Stage(StrEnum):
    DETECTION = "detection"
    OCR = "ocr"
    TRANSLATION = "translation"
    INPAINTING = "inpainting"
    TYPESETTING = "typesetting"


class StageStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class TextKind(StrEnum):
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    SFX = "sfx"
    SIGN = "sign"
    OTHER = "other"


class Orientation(StrEnum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class Alignment(StrEnum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _check_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("bbox must be [x0, y0, x1, y1] with 0 <= x0 < x1 and 0 <= y0 < y1")
    return bbox


BBox = Annotated[tuple[int, int, int, int], AfterValidator(_check_bbox)]
Point = tuple[float, float]


class TextStyle(_Model):
    """Typesetting overrides for one block. ``None`` means automatic / project default."""

    font: str | None = None
    size: float | None = Field(default=None, gt=0)
    uppercase: bool | None = None
    alignment: Alignment | None = None


class UserEdits(_Model):
    """Which parts of a block were changed by hand. The pipeline never overwrites them."""

    geometry: bool = False
    order: bool = False
    kind: bool = False
    skip: bool = False
    source_text: bool = False
    translation: bool = False
    style: bool = False


# Block field -> UserEdits flag that protects it.
_EDIT_FLAGS = {
    "bbox": "geometry",
    "polygon": "geometry",
    "bubble": "geometry",
    "orientation": "geometry",
    "order": "order",
    "kind": "kind",
    "skip": "skip",
    "source_text": "source_text",
    "translation": "translation",
    "style": "style",
}


class TextBlock(_Model):
    id: str = Field(min_length=1)
    order: int = Field(ge=0, description="Position in reading order within the page.")
    kind: TextKind = TextKind.DIALOGUE
    bbox: BBox
    polygon: list[Point] | None = Field(default=None, description="Outline of the text.")
    bubble: list[Point] | None = Field(default=None, description="Outline of its bubble.")
    orientation: Orientation | None = None
    source_text: str = ""
    translation: str = ""
    style: TextStyle = Field(default_factory=TextStyle)
    skip: bool = Field(default=False, description="Leave the original text untouched.")
    user_edited: UserEdits = Field(default_factory=UserEdits)

    def apply_pipeline_update(self, **changes: Any) -> list[str]:
        """Set fields computed by the pipeline, keeping any the user edited by hand.

        Returns the names of the fields that were kept because of a manual edit.
        """
        kept = []
        for name, value in changes.items():
            if getattr(self.user_edited, _edit_flag(name)):
                kept.append(name)
            else:
                setattr(self, name, value)
        return kept

    def apply_user_edit(self, **changes: Any) -> None:
        """Set fields edited by hand and protect them from later pipeline runs."""
        for name, value in changes.items():
            flag = _edit_flag(name)
            setattr(self, name, value)
            setattr(self.user_edited, flag, True)


def _edit_flag(field_name: str) -> str:
    try:
        return _EDIT_FLAGS[field_name]
    except KeyError:
        raise ValueError(f"{field_name!r} is not an editable text block field") from None


class SourceImage(_Model):
    path: str = Field(description="POSIX path, relative to the project folder when possible.")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mode: str = Field(description='Pillow mode of the original file, e.g. "L", "RGB", "CMYK".')
    format: str = Field(description='Pillow format name, e.g. "JPEG", "PNG", "WEBP".')


def _all_pending() -> dict[Stage, StageStatus]:
    return dict.fromkeys(Stage, StageStatus.PENDING)


class Page(_Model):
    index: int = Field(ge=0)
    source: SourceImage
    stages: dict[Stage, StageStatus] = Field(default_factory=_all_pending)
    blocks: list[TextBlock] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Derived files (masks, cleaned page, render) by name, relative to the project.",
    )

    @field_validator("stages")
    @classmethod
    def _fill_missing_stages(cls, stages: dict[Stage, StageStatus]) -> dict[Stage, StageStatus]:
        return _all_pending() | stages

    @model_validator(mode="after")
    def _check_unique_block_ids(self) -> Self:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for block in self.blocks:
            if block.id in seen:
                duplicates.add(block.id)
            seen.add(block.id)
        if duplicates:
            raise ValueError(f"duplicate block ids: {', '.join(sorted(duplicates))}")
        return self

    def block(self, block_id: str) -> TextBlock:
        for block in self.blocks:
            if block.id == block_id:
                return block
        raise KeyError(block_id)


class Project(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    name: str = Field(min_length=1)
    app_version: str = Field(description="Version of mangat that last saved the project.")
    created_at: datetime
    updated_at: datetime
    source_dir: str = Field(description="Folder the pages were imported from.")
    pages: list[Page] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_page_indices(self) -> Self:
        indices = [page.index for page in self.pages]
        if indices != list(range(len(self.pages))):
            raise ValueError("page indices must be 0, 1, 2, ... in order")
        return self
