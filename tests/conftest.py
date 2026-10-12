from collections.abc import Iterator
from pathlib import Path

import pytest

from manga_ai_translator import config as config_module
from manga_ai_translator.logging_setup import reset_logging


@pytest.fixture(autouse=True)
def app_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Keep every test away from the real user config, model and log folders."""
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    config_file = app_dir / "config.toml"
    config_file.write_text('[paths]\nmodels_dir = "models"\nlogs_dir = "logs"\n', encoding="utf-8")
    monkeypatch.setenv(config_module.CONFIG_ENV_VAR, str(config_file))
    monkeypatch.setattr(config_module, "default_config_path", lambda: app_dir / "user.toml")
    yield app_dir
    reset_logging()
