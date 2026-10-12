import hashlib
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from manga_ai_translator.cli import app, models_cmd
from manga_ai_translator.weights import (
    ChecksumError,
    DownloadError,
    HuggingFaceSource,
    ModelFile,
    ModelManager,
    ModelSpec,
    ModelStatus,
    Registry,
    UnknownModelError,
    UrlSource,
)
from manga_ai_translator.weights import manager as manager_module

WEIGHTS = bytes(range(256)) * 400
CONFIG = b'{"layers": 12}'
REVISION = "0123456789abcdef0123456789abcdef01234567"


def model_file(name: str, data: bytes, *, sha_of: bytes | None = None) -> ModelFile:
    return ModelFile(
        target=name,
        source=UrlSource(url=f"https://example.com/{name}"),
        sha256=hashlib.sha256(sha_of if sha_of is not None else data).hexdigest(),
        size=len(data),
    )


def make_spec(*files: ModelFile, model_id: str = "test-model") -> ModelSpec:
    return ModelSpec(
        id=model_id,
        description="Test model",
        license="MIT",
        license_url="https://example.com/license",
        files=files,
    )


class FakeServer:
    """In-memory HTTP server with optional Range support and injectable failures."""

    def __init__(self, files: dict[str, bytes], *, honor_range: bool = True) -> None:
        self.files = {f"/{name}": data for name, data in files.items()}
        self.honor_range = honor_range
        self.requests: list[httpx.Request] = []
        self.connection_errors = 0
        self.statuses: list[int] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.connection_errors:
            self.connection_errors -= 1
            raise httpx.ConnectError("connection reset", request=request)
        if self.statuses:
            return httpx.Response(self.statuses.pop(0))
        data = self.files.get(request.url.path)
        if data is None:
            return httpx.Response(404)
        if (range_header := request.headers.get("Range")) and self.honor_range:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            if start >= len(data):
                return httpx.Response(416)
            return httpx.Response(206, content=data[start:])
        return httpx.Response(200, content=data)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module.time, "sleep", lambda _seconds: None)


@pytest.fixture
def server() -> FakeServer:
    return FakeServer({"weights.bin": WEIGHTS, "config.json": CONFIG})


@pytest.fixture
def spec() -> ModelSpec:
    return make_spec(model_file("weights.bin", WEIGHTS), model_file("config.json", CONFIG))


def manager_for(server: FakeServer, tmp_path: Path) -> ModelManager:
    return ModelManager(tmp_path / "models", server.client())


def test_download_installs_and_verifies(
    server: FakeServer, spec: ModelSpec, tmp_path: Path
) -> None:
    manager = manager_for(server, tmp_path)
    updates: list[tuple[int, int]] = []
    manager.download(spec, lambda done, total: updates.append((done, total)))

    assert (tmp_path / "models" / "test-model" / "weights.bin").read_bytes() == WEIGHTS
    assert manager.status(spec) is ModelStatus.INSTALLED
    assert manager.verify(spec) == []
    assert updates[-1] == (spec.total_size, spec.total_size)
    assert all(done <= total for done, total in updates)


def test_installed_model_is_not_downloaded_again(
    server: FakeServer, spec: ModelSpec, tmp_path: Path
) -> None:
    manager = manager_for(server, tmp_path)
    manager.download(spec)
    count = len(server.requests)
    manager.download(spec)
    assert len(server.requests) == count

    manager.download(spec, force=True)
    assert len(server.requests) == count + 2


def test_interrupted_download_is_resumed(
    server: FakeServer, spec: ModelSpec, tmp_path: Path
) -> None:
    manager = manager_for(server, tmp_path)
    part = manager.file_path(spec, spec.files[0]).with_name("weights.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(WEIGHTS[:1000])

    manager.download(spec)
    assert server.requests[0].headers["Range"] == "bytes=1000-"
    assert manager.verify(spec) == []


def test_server_ignoring_range_restarts_cleanly(spec: ModelSpec, tmp_path: Path) -> None:
    server = FakeServer({"weights.bin": WEIGHTS, "config.json": CONFIG}, honor_range=False)
    manager = manager_for(server, tmp_path)
    part = manager.file_path(spec, spec.files[0]).with_name("weights.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(WEIGHTS[:1000])

    manager.download(spec)
    assert manager.verify(spec) == []


def test_oversized_partial_file_is_discarded(
    server: FakeServer, spec: ModelSpec, tmp_path: Path
) -> None:
    manager = manager_for(server, tmp_path)
    part = manager.file_path(spec, spec.files[0]).with_name("weights.bin.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"x" * (len(WEIGHTS) + 10))

    manager.download(spec)
    assert manager.verify(spec) == []


def test_checksum_mismatch_discards_the_file(server: FakeServer, tmp_path: Path) -> None:
    bad = make_spec(model_file("weights.bin", WEIGHTS, sha_of=b"something else"))
    manager = manager_for(server, tmp_path)
    with pytest.raises(ChecksumError, match="expected"):
        manager.download(bad)
    assert list((tmp_path / "models" / "test-model").iterdir()) == []


def test_missing_file_on_server_fails_without_retrying(server: FakeServer, tmp_path: Path) -> None:
    missing = make_spec(model_file("nope.bin", b"data"))
    with pytest.raises(DownloadError, match="HTTP 404"):
        manager_for(server, tmp_path).download(missing)
    assert len(server.requests) == 1


def test_transient_errors_are_retried(server: FakeServer, spec: ModelSpec, tmp_path: Path) -> None:
    server.connection_errors = 2
    server.statuses = [503]
    manager = manager_for(server, tmp_path)
    manager.download(spec)
    assert manager.verify(spec) == []


def test_persistent_errors_give_up(server: FakeServer, spec: ModelSpec, tmp_path: Path) -> None:
    server.connection_errors = 100
    with pytest.raises(DownloadError, match="after 4 attempts"):
        manager_for(server, tmp_path).download(spec)


def test_status_and_verify_report_problems(
    server: FakeServer, spec: ModelSpec, tmp_path: Path
) -> None:
    manager = manager_for(server, tmp_path)
    assert manager.status(spec) is ModelStatus.MISSING

    manager.download(make_spec(spec.files[0]))
    assert manager.status(spec) is ModelStatus.PARTIAL
    assert manager.verify(spec) == ["config.json: missing"]

    manager.download(spec)
    manager.file_path(spec, spec.files[0]).write_bytes(b"\0" * len(WEIGHTS))
    assert manager.status(spec) is ModelStatus.INSTALLED  # sizes match
    assert manager.verify(spec) == ["weights.bin: checksum mismatch"]


def test_hugging_face_source_url_and_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = HuggingFaceSource(repo_id="org/repo", revision=REVISION, path="onnx/model v2.onnx")
    assert source.url == f"https://huggingface.co/org/repo/resolve/{REVISION}/onnx/model%20v2.onnx"

    server = FakeServer({f"org/repo/resolve/{REVISION}/onnx/model v2.onnx": WEIGHTS})
    file = ModelFile(
        target="model.onnx",
        source=source,
        sha256=hashlib.sha256(WEIGHTS).hexdigest(),
        size=len(WEIGHTS),
    )
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    manager_for(server, tmp_path).download(make_spec(file))
    assert server.requests[0].headers["Authorization"] == "Bearer hf_test"


@pytest.mark.parametrize("target", ["../escape.bin", "/abs.bin", "C:/x.bin", "a\\b.bin"])
def test_unsafe_targets_are_rejected(target: str) -> None:
    with pytest.raises(ValidationError, match="relative path"):
        model_file(target, b"x")


def test_sources_are_pinned_and_secure() -> None:
    with pytest.raises(ValidationError):
        HuggingFaceSource(repo_id="org/repo", revision="main", path="m.onnx")
    with pytest.raises(ValidationError):
        UrlSource(url="http://insecure.example.com/m.onnx")


def test_registry_lookup(spec: ModelSpec) -> None:
    registry = Registry([spec])
    assert registry.get("test-model") is spec
    assert len(registry) == 1
    with pytest.raises(UnknownModelError, match="models list"):
        registry.get("other")
    with pytest.raises(ValueError, match="duplicate"):
        Registry([spec, spec])


# --- CLI ----------------------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture
def cli_catalog(
    monkeypatch: pytest.MonkeyPatch, server: FakeServer, spec: ModelSpec
) -> Iterator[FakeServer]:
    monkeypatch.setattr(models_cmd, "default_registry", lambda: Registry([spec]))
    monkeypatch.setattr(models_cmd, "create_http_client", server.client)
    yield server


def test_cli_with_empty_catalog() -> None:
    for command in (["list"], ["download", "--all"], ["verify"]):
        result = runner.invoke(app, ["models", *command])
        assert result.exit_code == 0, result.output
        assert "No models are registered yet" in result.output


@pytest.mark.usefixtures("cli_catalog")
def test_cli_download_list_and_verify() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert "test-model" in result.output
    assert "missing" in result.output

    result = runner.invoke(app, ["models", "download", "test-model"])
    assert result.exit_code == 0, result.output
    assert "1 model(s) ready" in result.output

    assert "installed" in runner.invoke(app, ["models", "list"]).output
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 0, result.output
    assert "test-model: ok" in result.output


@pytest.mark.usefixtures("cli_catalog")
def test_cli_verify_reports_damage(app_dir: Path) -> None:
    runner.invoke(app, ["models", "download", "--all"])
    (app_dir / "models" / "test-model" / "config.json").write_bytes(b"X" * len(CONFIG))
    result = runner.invoke(app, ["models", "verify", "test-model"])
    assert result.exit_code == 1
    assert "checksum mismatch" in result.output
    assert "--force" in result.output


@pytest.mark.usefixtures("cli_catalog")
def test_cli_download_argument_errors() -> None:
    result = runner.invoke(app, ["models", "download"])
    assert result.exit_code == 1
    assert "--all" in result.output

    result = runner.invoke(app, ["models", "download", "nope"])
    assert result.exit_code == 1
    assert "Unknown model 'nope'" in result.output
