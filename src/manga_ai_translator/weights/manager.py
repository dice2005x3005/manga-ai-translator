"""Downloading, checking and locating model weights on disk.

Each model lives in ``<models_dir>/<model id>/``. Downloads go to a ``.part`` file that is
resumed after interruptions and only renamed into place once its SHA-256 matches.
"""

import logging
import os
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import httpx

from manga_ai_translator import __version__
from manga_ai_translator.fileutils import sha256_file
from manga_ai_translator.weights.registry import HuggingFaceSource, ModelFile, ModelSpec

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
"""Called as ``(bytes_done, bytes_total)`` for the whole model."""

_MAX_ATTEMPTS = 4
_CHUNK_BYTES = 1 << 20
_TIMEOUT = httpx.Timeout(60.0, connect=15.0)


class ModelStatus(StrEnum):
    INSTALLED = "installed"
    PARTIAL = "partial"
    MISSING = "missing"


class DownloadError(Exception):
    """Raised when a model file cannot be downloaded."""


class ChecksumError(DownloadError):
    """Raised when a downloaded file does not match its expected SHA-256."""


class _RetryableError(Exception):
    pass


def create_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": f"mangat/{__version__}"},
    )


class ModelManager:
    def __init__(self, models_dir: Path, client: httpx.Client | None = None) -> None:
        self.models_dir = models_dir
        self._client = client or create_http_client()

    def model_dir(self, spec: ModelSpec) -> Path:
        return self.models_dir / spec.id

    def file_path(self, spec: ModelSpec, file: ModelFile) -> Path:
        return self.model_dir(spec) / file.target

    def status(self, spec: ModelSpec) -> ModelStatus:
        """Quick check based on file sizes; use :meth:`verify` for a full checksum check."""
        present = [self._has_expected_size(self.file_path(spec, file), file) for file in spec.files]
        if all(present):
            return ModelStatus.INSTALLED
        return ModelStatus.PARTIAL if any(present) else ModelStatus.MISSING

    def verify(self, spec: ModelSpec) -> list[str]:
        """Hash every file of the model and describe each problem found."""
        problems = []
        for file in spec.files:
            path = self.file_path(spec, file)
            if not path.is_file():
                problems.append(f"{file.target}: missing")
            elif path.stat().st_size != file.size:
                problems.append(f"{file.target}: wrong size")
            elif sha256_file(path) != file.sha256:
                problems.append(f"{file.target}: checksum mismatch")
        return problems

    def download(
        self, spec: ModelSpec, progress: ProgressCallback | None = None, *, force: bool = False
    ) -> None:
        total = spec.total_size
        done = 0
        for file in spec.files:
            path = self.file_path(spec, file)
            if force:
                path.unlink(missing_ok=True)
            if not self._has_expected_size(path, file):
                base = done

                def on_file_bytes(file_done: int, base: int = base) -> None:
                    if progress is not None:
                        progress(base + file_done, total)

                logger.info("Downloading %s/%s", spec.id, file.target)
                self._download_file(file, path, on_file_bytes)
            done += file.size
            if progress is not None:
                progress(done, total)

    @staticmethod
    def _has_expected_size(path: Path, file: ModelFile) -> bool:
        return path.is_file() and path.stat().st_size == file.size

    def _download_file(self, file: ModelFile, path: Path, on_bytes: Callable[[int], None]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        url = file.source.url

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            offset = part.stat().st_size if part.exists() else 0
            if offset > file.size:
                part.unlink()
                offset = 0
            try:
                if offset < file.size:
                    self._fetch(file, part, offset, on_bytes)
                if part.stat().st_size != file.size:
                    raise _RetryableError(
                        f"got {part.stat().st_size} of {file.size} bytes from {url}"
                    )
                break
            except (httpx.TransportError, _RetryableError) as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise DownloadError(
                        f"Download of {url} failed after {_MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                logger.warning("Download interrupted (%s), retrying in %ss", exc, attempt)
                time.sleep(attempt)

        digest = sha256_file(part)
        if digest != file.sha256:
            part.unlink()
            raise ChecksumError(
                f"{file.target} from {url} has SHA-256 {digest}, expected {file.sha256}. "
                "The download was discarded."
            )
        os.replace(part, path)

    def _fetch(
        self, file: ModelFile, part: Path, offset: int, on_bytes: Callable[[int], None]
    ) -> None:
        headers = {}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        if isinstance(file.source, HuggingFaceSource) and (token := os.environ.get("HF_TOKEN")):
            headers["Authorization"] = f"Bearer {token}"

        with self._client.stream("GET", file.source.url, headers=headers) as response:
            if response.status_code == 416:  # nothing left to send: the part is complete
                return
            if response.status_code >= 500:
                raise _RetryableError(f"server error HTTP {response.status_code}")
            if response.status_code not in (200, 206):
                raise DownloadError(f"{file.source.url} returned HTTP {response.status_code}")
            if response.status_code == 200:
                offset = 0  # the server ignored the Range header: start over

            written = offset
            on_bytes(written)
            with part.open("ab" if offset else "wb") as out:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    out.write(chunk)
                    written += len(chunk)
                    on_bytes(written)
