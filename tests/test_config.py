import re
import tomllib
from pathlib import Path

import pytest

from manga_ai_translator.config import (
    CONFIG_ENV_VAR,
    AppConfig,
    ConfigError,
    Device,
    LogLevel,
    Precision,
    QualityProfile,
    example_config_text,
    load_config,
    to_toml,
)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_no_config_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR)
    loaded = load_config()
    assert loaded.source is None
    assert loaded.config.runtime.device is Device.AUTO
    assert loaded.config.runtime.precision is Precision.FP32
    assert loaded.config.runtime.profile is QualityProfile.QUALITY
    assert loaded.config.paths.models_dir.name == "models"


def test_env_var_file_with_relative_paths_anchored_to_the_file(app_dir: Path) -> None:
    loaded = load_config()
    assert loaded.source == app_dir / "config.toml"
    assert loaded.config.paths.models_dir == app_dir / "models"
    assert loaded.config.paths.logs_dir == app_dir / "logs"


def test_explicit_path_wins_over_env_var(tmp_path: Path) -> None:
    path = write(tmp_path / "other.toml", '[runtime]\ndevice = "cpu"\n')
    loaded = load_config(path)
    assert loaded.source == path
    assert loaded.config.runtime.device is Device.CPU


def test_missing_explicit_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(write(tmp_path / "bad.toml", "[runtime\n"))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"runtime\.devise"):
        load_config(write(tmp_path / "typo.toml", '[runtime]\ndevise = "cpu"\n'))


def test_invalid_value_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"runtime\.precision"):
        load_config(write(tmp_path / "bad.toml", '[runtime]\nprecision = "int3"\n'))


def test_environment_variables_are_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANGAT_TEST_DIR", str(tmp_path / "somewhere"))
    path = write(tmp_path / "c.toml", '[paths]\nmodels_dir = "$MANGAT_TEST_DIR/models"\n')
    assert load_config(path).config.paths.models_dir == tmp_path / "somewhere" / "models"


def test_log_level_is_case_insensitive(tmp_path: Path) -> None:
    path = write(tmp_path / "c.toml", '[logging]\nlevel = "debug"\n')
    assert load_config(path).config.logging.level is LogLevel.DEBUG


def test_example_config_matches_defaults_and_its_keys_are_valid(tmp_path: Path) -> None:
    text = example_config_text()
    commented = load_config(write(tmp_path / "commented.toml", text)).config
    assert commented == AppConfig()

    uncommented = re.sub(r"^# (\w+ = .*)$", r"\1", text, flags=re.MULTILINE)
    assert uncommented != text
    load_config(write(tmp_path / "uncommented.toml", uncommented))


def test_to_toml_round_trip() -> None:
    config = AppConfig()
    config.runtime.device = Device.CPU
    config.paths.models_dir = Path(r"C:\Users\someone\models")
    assert AppConfig.model_validate(tomllib.loads(to_toml(config))) == config
