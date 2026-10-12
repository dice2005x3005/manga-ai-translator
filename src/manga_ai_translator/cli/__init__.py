"""Command line interface (`mangat`)."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from manga_ai_translator import __version__
from manga_ai_translator.cli._context import AppContext, console, fail
from manga_ai_translator.cli.config_cmd import config_app
from manga_ai_translator.cli.project_cmd import project_app
from manga_ai_translator.config import ConfigError, LogLevel, load_config
from manga_ai_translator.logging_setup import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="mangat",
    help="Translate manga from Japanese to English with local AI models.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
app.add_typer(project_app, name="project")
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mangat {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    config_file: Annotated[
        Path | None,
        typer.Option("--config", dir_okay=False, help="Path to a config.toml file."),
    ] = None,
    log_level: Annotated[
        LogLevel | None,
        typer.Option("--log-level", case_sensitive=False, help="Console log level."),
    ] = None,
) -> None:
    """Translate manga from Japanese to English with local AI models."""
    try:
        loaded = load_config(config_file)
    except ConfigError as exc:
        fail(str(exc), code=2)
    config = loaded.config
    if log_level is not None:
        config.logging.level = log_level

    log_file = setup_logging(config.logging.level, config.paths.logs_dir)
    logger.debug("mangat %s, config: %s", __version__, loaded.source or "built-in defaults")
    ctx.obj = AppContext(config=config, config_source=loaded.source, log_file=log_file)


def main() -> None:
    app()
