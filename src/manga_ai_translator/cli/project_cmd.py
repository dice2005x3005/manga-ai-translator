"""`mangat project`: create and inspect translation projects."""

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from manga_ai_translator.cli._context import console, fail
from manga_ai_translator.imaging import ImageError
from manga_ai_translator.project import ProjectError, StageStatus, load_project
from manga_ai_translator.project.create import SourceStatus, check_sources, create_project

project_app = typer.Typer(help="Create and inspect translation projects.", no_args_is_help=True)


@project_app.command("new")
def new(
    images_dir: Annotated[
        Path,
        typer.Argument(help="Folder with the chapter's page images.", file_okay=False, exists=True),
    ],
    project_dir: Annotated[
        Path | None,
        typer.Option(
            "--project-dir",
            "-o",
            file_okay=False,
            help="Where to create the project. Default: '<images folder>.mangat' next to it.",
        ),
    ] = None,
    name: Annotated[
        str | None, typer.Option(help="Project name. Default: the images folder name.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing project.")] = False,
) -> None:
    """Create a project with one page per image (JPG, PNG, WEBP), in natural order."""
    with Progress(
        TextColumn("Reading pages"), BarColumn(), MofNCompleteColumn(), console=console
    ) as progress:
        task = progress.add_task("pages", total=None)

        def on_page(done: int, total: int, _path: Path) -> None:
            progress.update(task, completed=done, total=total)

        try:
            project, created_dir = create_project(
                images_dir, project_dir, name, overwrite=force, progress=on_page
            )
        except (ProjectError, ImageError) as exc:
            progress.stop()
            fail(str(exc))

    console.print(
        f"Created project [bold]{escape(project.name)}[/] with {len(project.pages)} pages "
        f"in {escape(str(created_dir))}",
        soft_wrap=True,
    )


@project_app.command("info")
def info(
    project_dir: Annotated[
        Path, typer.Argument(help="Project folder (the one containing project.json).")
    ],
    verify: Annotated[
        bool, typer.Option("--verify", help="Hash every image to detect modified pages.")
    ] = False,
) -> None:
    """Show a project's pages and the state of each processing stage."""
    try:
        project = load_project(project_dir)
    except ProjectError as exc:
        fail(str(exc))

    console.print(f"[bold]{escape(project.name)}[/]  ({len(project.pages)} pages)")
    console.print(f"Folder:  {escape(str(project_dir.resolve()))}", soft_wrap=True)
    console.print(f"Images:  {escape(project.source_dir)}", soft_wrap=True)
    console.print(f"Created: {project.created_at:%Y-%m-%d %H:%M} UTC")
    console.print(
        f"Updated: {project.updated_at:%Y-%m-%d %H:%M} UTC (mangat {project.app_version})"
    )

    statuses = check_sources(project, project_dir, verify_hash=verify)
    table = Table("#", "Image", "Size", "Mode", "Blocks", "Stages done", "Source")
    for page, status in zip(project.pages, statuses, strict=True):
        done = [stage.value for stage, state in page.stages.items() if state is StageStatus.DONE]
        table.add_row(
            str(page.index + 1),
            escape(Path(page.source.path).name),
            f"{page.source.width}x{page.source.height}",
            page.source.mode,
            str(len(page.blocks)),
            ", ".join(done) or "-",
            status.value if status is SourceStatus.OK else f"[bold red]{status.value}[/]",
        )
    console.print(table)

    problems = sum(status is not SourceStatus.OK for status in statuses)
    if problems:
        console.print(f"[bold red]{problems} page image(s) are missing or modified.[/]")
    elif verify:
        console.print("All page images match the project.")
    else:
        console.print("All page images found. Use --verify to also check their content.")
