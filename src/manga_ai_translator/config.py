"""Application configuration.

Values are resolved in layers: built-in defaults < TOML config file < command line overrides.
The config file is looked up in this order: ``--config``, ``$MANGAT_CONFIG``, then the
per-user config file (``%LOCALAPPDATA%\\MangaAITranslator\\config.toml`` on Windows).
"""

import os
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir, user_log_dir
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from manga_ai_translator import APP_NAME

CONFIG_ENV_VAR = "MANGAT_CONFIG"
CONFIG_FILENAME = "config.toml"


class ConfigError(Exception):
    """Raised when a configuration file cannot be read or is invalid."""


class Device(StrEnum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"


class QualityProfile(StrEnum):
    QUALITY = "quality"
    BALANCED = "balanced"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def default_config_path() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False)) / CONFIG_FILENAME


def default_models_dir() -> Path:
    return Path(user_data_dir(APP_NAME, appauthor=False)) / "models"


def default_logs_dir() -> Path:
    return Path(user_log_dir(APP_NAME, appauthor=False))


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(_Section):
    models_dir: Path = Field(default_factory=default_models_dir)
    logs_dir: Path = Field(default_factory=default_logs_dir)

    @field_validator("models_dir", "logs_dir")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return Path(os.path.expandvars(value)).expanduser()


class RuntimeConfig(_Section):
    device: Device = Device.AUTO
    precision: Precision = Precision.FP32
    profile: QualityProfile = QualityProfile.QUALITY


class LoggingConfig(_Section):
    level: LogLevel = LogLevel.INFO

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


class AppConfig(_Section):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


@dataclass(frozen=True)
class LoadedConfig:
    config: AppConfig
    source: Path | None
    """The file the config was read from, or None when only built-in defaults are used."""


def resolve_config_path(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit
    if env_path := os.environ.get(CONFIG_ENV_VAR):
        return Path(env_path)
    default = default_config_path()
    return default if default.is_file() else None


def load_config(explicit: Path | None = None) -> LoadedConfig:
    path = resolve_config_path(explicit)
    if path is None:
        return LoadedConfig(AppConfig(), None)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc

    try:
        config = AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config file {path}:\n{format_validation_error(exc)}") from exc

    # Relative paths in a config file are relative to that file, not to the working directory.
    for name in ("models_dir", "logs_dir"):
        value: Path = getattr(config.paths, name)
        if not value.is_absolute():
            setattr(config.paths, name, path.parent / value)
    return LoadedConfig(config, path)


def format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)


def example_config_text() -> str:
    return resources.files(__package__).joinpath("config.example.toml").read_text(encoding="utf-8")


def to_toml(config: AppConfig) -> str:
    """Serialize the effective configuration as TOML (all values are strings)."""
    sections = []
    for section_name, section in config:
        lines = [f"[{section_name}]"]
        for key, value in section:
            text = str(value.value if isinstance(value, StrEnum) else value)
            lines.append(f"{key} = {_toml_string(text)}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def _toml_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
