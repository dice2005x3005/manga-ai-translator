from pathlib import Path

from typer.testing import CliRunner

from manga_ai_translator import __version__
from manga_ai_translator.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_config_show_reports_source(app_dir: Path) -> None:
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert str(app_dir / "config.toml") in result.output
    assert "[runtime]" in result.output


def test_config_init_writes_example_and_refuses_overwrite(app_dir: Path) -> None:
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0, result.output
    assert (app_dir / "user.toml").read_text(encoding="utf-8").startswith("# Manga AI Translator")

    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 1
    assert "--force" in result.output


def test_invalid_config_exits_with_code_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[runtime]\ndevice = 'tpu'\n", encoding="utf-8")
    result = runner.invoke(app, ["--config", str(bad), "config", "show"])
    assert result.exit_code == 2
    assert "runtime.device" in result.output


def test_log_level_option_overrides_config() -> None:
    result = runner.invoke(app, ["--log-level", "debug", "config", "show"])
    assert result.exit_code == 0, result.output
    assert 'level = "DEBUG"' in result.output
