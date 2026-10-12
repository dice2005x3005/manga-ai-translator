"""State and helpers shared by every CLI command."""

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.markup import escape

from manga_ai_translator.config import AppConfig

console = Console()
err_console = Console(stderr=True)


@dataclass
class AppContext:
    config: AppConfig
    config_source: Path | None
    log_file: Path | None


def get_context(ctx: typer.Context) -> AppContext:
    obj = ctx.find_root().obj
    assert isinstance(obj, AppContext), "the root callback must run before any command"
    return obj


def fail(message: str, code: int = 1) -> NoReturn:
    err_console.print(f"[bold red]Error:[/] {escape(message)}")
    raise typer.Exit(code)
