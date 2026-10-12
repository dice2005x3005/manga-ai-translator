"""Model weights: catalog, download and verification."""

from manga_ai_translator.weights.manager import (
    ChecksumError,
    DownloadError,
    ModelManager,
    ModelStatus,
    create_http_client,
)
from manga_ai_translator.weights.registry import (
    HuggingFaceSource,
    ModelFile,
    ModelSpec,
    Registry,
    UnknownModelError,
    UrlSource,
    default_registry,
)

__all__ = [
    "ChecksumError",
    "DownloadError",
    "HuggingFaceSource",
    "ModelFile",
    "ModelManager",
    "ModelSpec",
    "ModelStatus",
    "Registry",
    "UnknownModelError",
    "UrlSource",
    "create_http_client",
    "default_registry",
]
