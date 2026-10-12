"""`mangat doctor`: report the hardware and how mangat will use it."""

import json
import logging
from contextlib import nullcontext
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from manga_ai_translator import __version__
from manga_ai_translator.cli._context import console, get_context
from manga_ai_translator.config import RuntimeConfig
from manga_ai_translator.hardware import (
    GIB,
    GpuTestResult,
    HardwareReport,
    Recommendation,
    detect_hardware,
    interpret_gpu_test,
    recommend,
    run_gpu_test,
)

logger = logging.getLogger(__name__)


def doctor(
    ctx: typer.Context,
    test_gpu: Annotated[
        bool,
        typer.Option("--test-gpu", help="Run a short numeric test on the GPU in fp32 and fp16."),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the report as JSON (handy to share).")
    ] = False,
) -> None:
    """Check the CPU, RAM, GPU and PyTorch setup and recommend how to run the models."""
    runtime = get_context(ctx).config.runtime
    models_dir = get_context(ctx).config.paths.models_dir

    with nullcontext() if as_json else console.status("Inspecting hardware..."):
        report = detect_hardware(models_dir)
    recommendation = recommend(report, runtime)
    logger.debug("Hardware report: %s", report.model_dump_json())

    gpu_test: GpuTestResult | None = None
    gpu_test_error: str | None = None
    if test_gpu:
        if not report.torch.cuda_available:
            gpu_test_error = "CUDA is not available, there is no GPU to test."
        else:
            with nullcontext() if as_json else console.status("Testing the GPU..."):
                try:
                    gpu_test = run_gpu_test()
                except Exception as exc:
                    logger.debug("GPU test crashed", exc_info=True)
                    gpu_test_error = f"{type(exc).__name__}: {exc}"
            if gpu_test is not None:
                recommendation.notes.extend(interpret_gpu_test(gpu_test))

    if as_json:
        payload = {
            "mangat_version": __version__,
            "report": report.model_dump(mode="json"),
            "recommendation": recommendation.model_dump(mode="json"),
            "gpu_test": gpu_test.model_dump(mode="json") if gpu_test else None,
            "gpu_test_error": gpu_test_error,
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    _print_report(report)
    if gpu_test is not None:
        _print_gpu_test(gpu_test)
    elif gpu_test_error is not None:
        console.print(f"\n[bold red]GPU test:[/] {escape(gpu_test_error)}")
    _print_recommendation(recommendation, runtime)


def _gib(size: int) -> str:
    return f"{size / GIB:.1f} GiB"


def _yes_no(value: bool | None) -> str:
    return "unknown" if value is None else ("yes" if value else "no")


def _print_report(report: HardwareReport) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    cpu = report.cpu
    table.add_row("OS", escape(report.os))
    table.add_row("Python", report.python)
    table.add_row(
        "CPU",
        f"{escape(cpu.name)} ({cpu.physical_cores} cores / {cpu.logical_cores} threads, "
        f"AVX2: {_yes_no(cpu.avx2)})",
    )
    table.add_row(
        "RAM",
        f"{_gib(report.memory.total_bytes)} total, {_gib(report.memory.available_bytes)} available",
    )

    if report.nvidia.gpus:
        for gpu in report.nvidia.gpus:
            free = f", {_gib(gpu.free_vram_bytes)} free" if gpu.free_vram_bytes is not None else ""
            table.add_row(
                f"GPU {gpu.index}", f"{escape(gpu.name)}, {_gib(gpu.total_vram_bytes)} VRAM{free}"
            )
        table.add_row("NVIDIA driver", report.nvidia.driver_version or "unknown")
    else:
        table.add_row("GPU", "no NVIDIA GPU detected")

    torch_info = report.torch
    if not torch_info.installed:
        torch_text = "[red]not installed[/]"
        if torch_info.error:
            torch_text += f" ({escape(torch_info.error)})"
    elif torch_info.cuda_build is None:
        torch_text = f"{torch_info.version} (CPU-only build)"
    elif torch_info.cuda_available:
        capability = ", ".join(torch_info.compute_capabilities)
        torch_text = (
            f"{torch_info.version} (CUDA {torch_info.cuda_build}, GPU usable, "
            f"compute capability {capability})"
        )
    else:
        torch_text = f"{torch_info.version} (CUDA {torch_info.cuda_build}, [red]GPU not usable[/])"
    table.add_row("PyTorch", torch_text)

    free = _gib(report.disk.free_bytes) if report.disk.free_bytes is not None else "unknown"
    table.add_row("Models folder", f"{escape(report.disk.path)} ({free} free)")

    console.print("[bold underline]System[/]")
    console.print(table)


def _print_gpu_test(result: GpuTestResult) -> None:
    table = Table("Precision", "Result", "Max relative error", "Matmul TFLOPS")
    for check in result.checks:
        table.add_row(
            check.dtype,
            "[green]ok[/]" if check.ok else "[bold red]FAILED[/]",
            f"{check.max_relative_error:.2e}" if check.max_relative_error is not None else "-",
            f"{check.matmul_tflops:.2f}" if check.matmul_tflops is not None else "-",
        )
        if check.error:
            table.add_row("", f"[red]{escape(check.error)}[/]", "", "")
    console.print(f"\n[bold underline]GPU test[/] ({escape(result.device)})")
    console.print(table)


def _print_recommendation(recommendation: Recommendation, runtime: RuntimeConfig) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Device", f"{recommendation.device} (configured: {runtime.device.value})")
    table.add_row(
        "Precision", f"{recommendation.precision.value} (configured: {runtime.precision.value})"
    )
    table.add_row(
        "Profile", f"{recommendation.profile.value} (configured: {runtime.profile.value})"
    )
    console.print("\n[bold underline]Recommendation[/]")
    console.print(table)
    for note in recommendation.notes:
        console.print(f"[yellow]- {escape(note)}[/]")
