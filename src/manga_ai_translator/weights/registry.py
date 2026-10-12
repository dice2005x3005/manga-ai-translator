"""Catalog of the model weights mangat can download.

Weights are never committed to the repository. They are downloaded on demand from their
original publisher, pinned to an exact revision and verified with SHA-256.
"""

from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath
from typing import Annotated, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator

from manga_ai_translator.project.schema import Stage


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HuggingFaceSource(_Frozen):
    kind: Literal["huggingface"] = "huggingface"
    repo_id: str
    revision: str = Field(
        pattern=r"^[0-9a-f]{40}$",
        description="Full commit hash, never a branch, so the file cannot change under us.",
    )
    path: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{quote(self.path)}"


class UrlSource(_Frozen):
    kind: Literal["url"] = "url"
    url: str = Field(pattern=r"^https://")


Source = Annotated[HuggingFaceSource | UrlSource, Field(discriminator="kind")]


class ModelFile(_Frozen):
    target: str = Field(description="Path inside the model's folder, e.g. 'model.onnx'.")
    source: Source
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(gt=0)

    @field_validator("target")
    @classmethod
    def _safe_relative_path(cls, target: str) -> str:
        path = PurePosixPath(target)
        if path.is_absolute() or ".." in path.parts or "\\" in target or ":" in target:
            raise ValueError("target must be a relative path inside the model folder")
        return target


class ModelSpec(_Frozen):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    description: str
    stage: Stage | None = None
    license: str
    license_url: str
    files: tuple[ModelFile, ...] = Field(min_length=1)

    @property
    def total_size(self) -> int:
        return sum(file.size for file in self.files)


class UnknownModelError(KeyError):
    def __str__(self) -> str:
        return f"Unknown model {self.args[0]!r}. Run `mangat models list` to see the catalog."


class Registry:
    def __init__(self, specs: Iterable[ModelSpec]) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for spec in specs:
            if spec.id in self._specs:
                raise ValueError(f"duplicate model id {spec.id!r}")
            self._specs[spec.id] = spec

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise UnknownModelError(model_id) from None

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


BUILTIN_MODELS: tuple[ModelSpec, ...] = ()
"""Filled in as each pipeline stage is implemented."""


def default_registry() -> Registry:
    return Registry(BUILTIN_MODELS)
