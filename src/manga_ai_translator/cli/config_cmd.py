"""`mangat config`: inspect or create the configuration file."""

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from manga_ai_translator import config as cfg
from manga_ai_translator.cli._context import console, fail, get_context

config_app = typer.Typer(help="Show or create the configuration file.", no_args_is_help=True)


@config_app.command("show")
def show(ctx: typer.Context) -> None:
    """Print the effective configuration and where it was loaded from."""
    app_ctx = get_context(ctx)
    if app_ctx.config_source is None:
        console.print(
            "# No config file found, using built-in defaults. "
            "Create one with `mangat config init`.",
            markup=False,
            soft_wrap=True,
        )
    else:
        console.print(f"# Loaded from {app_ctx.config_source}", markup=False, soft_wrap=True)
    console.print(cfg.to_toml(app_ctx.config), markup=False, highlight=False, soft_wrap=True)


@config_app.command("init")
def init(
    path: Annotated[
        Path | None,
        typer.Option(help="Where to write the file. Defaults to the per-user config file."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a commented example config file to edit."""
    target = path or cfg.default_config_path()
    if target.exists() and not force:
        fail(f"{target} already exists. Use --force to overwrite it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(cfg.example_config_text(), encoding="utf-8")
    console.print(f"Created [bold]{escape(str(target))}[/]")
