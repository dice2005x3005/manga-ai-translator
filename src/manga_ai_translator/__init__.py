"""Manga AI Translator: translate manga from Japanese to English with local AI models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("manga-ai-translator")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw source tree
    __version__ = "0.0.0"

APP_NAME = "MangaAITranslator"
