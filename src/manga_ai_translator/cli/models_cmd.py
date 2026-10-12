"""`mangat models`: list, download and verify model weights."""

from typing import Annotated

import typer
from rich.filesize import decimal
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from manga_ai_translator.cli._context import console, fail, get_context
from manga_ai_translator.weights import (
    DownloadError,
    ModelManager,
    ModelSpec,
    ModelStatus,
    Registry,
    UnknownModelError,
    create_http_client,
    default_registry,
)

models_app = typer.Typer(help="List, download and verify AI model weights.", no_args_is_help=True)

_STATUS_STYLE = {
    ModelStatus.INSTALLED: "green",
    ModelStatus.PARTIAL: "yellow",
    ModelStatus.MISSING: "dim",
}
_NO_MODELS = "No models are registered yet. They are added as each pipeline stage is built."

ModelIds = Annotated[
    list[str] | None, typer.Argument(help="Model ids, as shown by `mangat models list`.")
]


def _manager(ctx: typer.Context) -> ModelManager:
    return ModelManager(get_context(ctx).config.paths.models_dir, create_http_client())


def _select(registry: Registry, model_ids: list[str] | None) -> list[ModelSpec]:
    try:
        return [registry.get(model_id) for model_id in model_ids or []]
    except UnknownModelError as exc:
        fail(str(exc))


@models_app.command("list")
def list_models(ctx: typer.Context) -> None:
    """Show every model in the catalog and whether it is downloaded."""
    registry = default_registry()
    manager = _manager(ctx)
    console.print(f"Models folder: {escape(str(manager.models_dir))}", soft_wrap=True)
    if not len(registry):
        console.print(_NO_MODELS)
        return

    table = Table("Model", "Stage", "Size", "Status", "License")
    for spec in registry:
        status = manager.status(spec)
        table.add_row(
            spec.id,
            spec.stage.value if spec.stage else "-",
            decimal(spec.total_size),
            f"[{_STATUS_STYLE[status]}]{status.value}[/]",
            escape(spec.license),
        )
    console.print(table)


@models_app.command("download")
def download(
    ctx: typer.Context,
    model_ids: ModelIds = None,
    all_models: Annotated[
        bool, typer.Option("--all", help="Download every model in the catalog.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Download again even if already present.")
    ] = False,
) -> None:
    """Download model weights, resuming interrupted downloads and checking their SHA-256."""
    registry = default_registry()
    if all_models:
        specs = list(registry)
    elif model_ids:
        specs = _select(registry, model_ids)
    else:
        fail("Name at least one model id or use --all.")
    if not specs:
        console.print(_NO_MODELS)
        return

    manager = _manager(ctx)
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for spec in specs:
            task = progress.add_task(spec.id, total=spec.total_size)
            try:
                manager.download(
                    spec,
                    lambda done, _total, task=task: progress.update(task, completed=done),
                    force=force,
                )
            except DownloadError as exc:
                progress.stop()
                fail(str(exc))
    console.print(
        f"{len(specs)} model(s) ready in {escape(str(manager.models_dir))}", soft_wrap=True
    )


@models_app.command("verify")
def verify(ctx: typer.Context, model_ids: ModelIds = None) -> None:
    """Check the SHA-256 of downloaded models (all of them by default)."""
    registry = default_registry()
    specs = _select(registry, model_ids) if model_ids else list(registry)
    if not specs:
        console.print(_NO_MODELS)
        return

    manager = _manager(ctx)
    broken = []
    for spec in specs:
        if manager.status(spec) is ModelStatus.MISSING:
            console.print(f"{spec.id}: [dim]not downloaded[/]")
            continue
        with console.status(f"Hashing {spec.id}..."):
            problems = manager.verify(spec)
        if problems:
            broken.append(spec.id)
            console.print(f"{spec.id}: [bold red]damaged[/]")
            for problem in problems:
                console.print(f"  - {escape(problem)}")
        else:
            console.print(f"{spec.id}: [green]ok[/]")

    if broken:
        fail(f"Download again with `mangat models download {' '.join(broken)} --force`.")
