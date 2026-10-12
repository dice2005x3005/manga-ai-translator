import json
import platform
from pathlib import Path

import pytest
from typer.testing import CliRunner

from manga_ai_translator.cli import app
from manga_ai_translator.config import Device, Precision, QualityProfile, RuntimeConfig
from manga_ai_translator.hardware import (
    GIB,
    CpuInfo,
    DiskInfo,
    DtypeCheck,
    GpuInfo,
    GpuTestResult,
    HardwareReport,
    MemoryInfo,
    NvidiaInfo,
    TorchInfo,
    detect_disk,
    detect_hardware,
    interpret_gpu_test,
    is_gtx16,
    recommend,
    run_gpu_test,
)

runner = CliRunner()

GTX_1660 = GpuInfo(index=0, name="NVIDIA GeForce GTX 1660", total_vram_bytes=6 * GIB,
                   free_vram_bytes=5 * GIB)  # fmt: skip
CUDA_TORCH = TorchInfo(
    installed=True,
    version="2.14.1+cu126",
    cuda_build="12.6",
    cuda_available=True,
    compute_capabilities=["7.5"],
    arch_list=["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"],
)
CPU_TORCH = TorchInfo(installed=True, version="2.14.1+cpu")


def make_report(
    *,
    gpus: list[GpuInfo] | None = None,
    torch: TorchInfo = CUDA_TORCH,
    ram_gib: float = 15.7,
    avx2: bool | None = True,
    free_disk_gib: float = 200,
) -> HardwareReport:
    """The target desktop (Ryzen 5 1600, 16 GB, GTX 1660) unless overridden."""
    return HardwareReport(
        os="Windows 11",
        python="3.12.15",
        cpu=CpuInfo(name="AMD Ryzen 5 1600", physical_cores=6, logical_cores=12, avx2=avx2),
        memory=MemoryInfo(total_bytes=int(ram_gib * GIB), available_bytes=8 * GIB),
        nvidia=NvidiaInfo(driver_version="580.00", gpus=[GTX_1660] if gpus is None else gpus),
        torch=torch,
        disk=DiskInfo(path="C:/models", free_bytes=int(free_disk_gib * GIB)),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NVIDIA GeForce GTX 1660", True),
        ("NVIDIA GeForce GTX 1660 SUPER", True),
        ("GeForce GTX 1650 Ti", True),
        ("NVIDIA GeForce RTX 2060", False),
        ("NVIDIA GeForce GTX 1060 6GB", False),
    ],
)
def test_is_gtx16(name: str, expected: bool) -> None:
    assert is_gtx16(name) is expected


def test_target_desktop_gets_gpu_and_quality_profile() -> None:
    rec = recommend(make_report(), RuntimeConfig())
    assert (rec.device, rec.precision, rec.profile) == (
        "cuda",
        Precision.FP32,
        QualityProfile.QUALITY,
    )
    assert rec.notes == []


def test_laptop_without_nvidia_uses_cpu_and_balanced_profile() -> None:
    rec = recommend(make_report(gpus=[], torch=CPU_TORCH), RuntimeConfig())
    assert (rec.device, rec.profile) == ("cpu", QualityProfile.BALANCED)
    assert any("very slow" in note for note in rec.notes)


def test_gpu_with_cpu_only_torch_suggests_cuda_extra() -> None:
    rec = recommend(make_report(torch=CPU_TORCH), RuntimeConfig())
    assert rec.device == "cpu"
    assert any("uv sync --extra cuda" in note for note in rec.notes)


def test_missing_torch_is_reported() -> None:
    rec = recommend(make_report(torch=TorchInfo(installed=False)), RuntimeConfig())
    assert rec.device == "cpu"
    assert any("not installed" in note for note in rec.notes)


def test_cuda_build_that_cannot_use_gpu_suggests_driver_update() -> None:
    broken = CUDA_TORCH.model_copy(update={"cuda_available": False})
    rec = recommend(make_report(torch=broken), RuntimeConfig())
    assert rec.device == "cpu"
    assert any("driver" in note for note in rec.notes)


def test_gpu_architecture_missing_from_build() -> None:
    no_turing = CUDA_TORCH.model_copy(update={"arch_list": ["sm_80", "sm_90"]})
    rec = recommend(make_report(torch=no_turing), RuntimeConfig())
    assert rec.device == "cpu"
    assert any("sm_75" in note for note in rec.notes)


def test_fp16_on_gtx16_is_warned_about() -> None:
    rec = recommend(make_report(), RuntimeConfig(precision=Precision.FP16))
    assert rec.precision is Precision.FP16
    assert any("GTX 16xx" in note for note in rec.notes)


def test_fp16_on_cpu_falls_back_to_fp32() -> None:
    report = make_report(gpus=[], torch=CPU_TORCH)
    rec = recommend(report, RuntimeConfig(precision=Precision.FP16))
    assert rec.precision is Precision.FP32


def test_forced_cuda_without_cuda_falls_back_to_cpu() -> None:
    report = make_report(gpus=[], torch=CPU_TORCH)
    rec = recommend(report, RuntimeConfig(device=Device.CUDA))
    assert rec.device == "cpu"
    assert any("configured" in note for note in rec.notes)


def test_forced_cpu_is_respected() -> None:
    assert recommend(make_report(), RuntimeConfig(device=Device.CPU)).device == "cpu"


def test_low_ram_means_balanced_profile() -> None:
    rec = recommend(make_report(ram_gib=7.9), RuntimeConfig())
    assert rec.profile is QualityProfile.BALANCED


def test_small_gpu_means_balanced_profile() -> None:
    small = GTX_1660.model_copy(update={"total_vram_bytes": 4 * GIB})
    assert recommend(make_report(gpus=[small]), RuntimeConfig()).profile is QualityProfile.BALANCED


def test_missing_avx2_and_low_disk_are_warned_about() -> None:
    rec = recommend(make_report(avx2=False, free_disk_gib=5), RuntimeConfig())
    assert any("AVX2" in note for note in rec.notes)
    assert any("GiB free" in note for note in rec.notes)


def test_detect_hardware_on_this_machine(tmp_path: Path) -> None:
    report = detect_hardware(tmp_path / "models")
    assert report.python == platform.python_version()
    assert report.memory.total_bytes > 0
    assert report.cpu.logical_cores
    assert report.torch.installed
    assert report.disk.free_bytes


def test_detect_disk_uses_nearest_existing_folder(tmp_path: Path) -> None:
    disk = detect_disk(tmp_path / "does" / "not" / "exist")
    assert disk.path.endswith("exist")
    assert disk.free_bytes


def test_numeric_checks_run_on_cpu() -> None:
    result = run_gpu_test("cpu", size=64)
    assert [check.dtype for check in result.checks] == ["float32", "float16"]
    fp32 = result.check("float32")
    assert fp32 is not None and fp32.ok
    assert fp32.max_relative_error is not None and fp32.max_relative_error < 1e-3


def test_gpu_test_requires_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        run_gpu_test()


def test_interpret_gpu_test() -> None:
    good32 = DtypeCheck(dtype="float32", ok=True, matmul_tflops=4.0)
    bad16 = DtypeCheck(dtype="float16", ok=False)
    good16 = DtypeCheck(dtype="float16", ok=True, matmul_tflops=8.0)

    assert (
        "keep precision" in interpret_gpu_test(GpuTestResult(device="x", checks=[good32, bad16]))[0]
    )
    assert "2.0x" in interpret_gpu_test(GpuTestResult(device="x", checks=[good32, good16]))[0]
    bad32 = DtypeCheck(dtype="float32", ok=False)
    assert (
        "even in fp32" in interpret_gpu_test(GpuTestResult(device="x", checks=[bad32, good16]))[0]
    )


def test_doctor_command() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    for heading in ("System", "PyTorch", "Recommendation"):
        assert heading in result.output


def test_doctor_json_is_parseable() -> None:
    result = runner.invoke(app, ["doctor", "--json", "--test-gpu"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {
        "mangat_version",
        "report",
        "recommendation",
        "gpu_test",
        "gpu_test_error",
    }
    assert payload["recommendation"]["device"] in {"cuda", "cpu"}
