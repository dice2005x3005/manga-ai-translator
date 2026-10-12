"""Command line interface (`mangat`)."""

from typing import Annotated

import typer
from rich.console import Console

from manga_ai_translator import __version__

app = typer.Typer(
    name="mangat",
    help="Translate manga from Japanese to English with local AI models.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mangat {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Translate manga from Japanese to English with local AI models."""


def main() -> None:
    app()
