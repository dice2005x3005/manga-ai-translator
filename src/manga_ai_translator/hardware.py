"""Hardware detection, device recommendations and a GPU sanity test (`mangat doctor`).

Every probe is defensive: a failing probe reports "unknown" instead of crashing, because the
whole point of the report is to diagnose machines where something is not set up correctly.
"""

import ctypes
import logging
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import psutil
from pydantic import BaseModel

from manga_ai_translator.config import Device, Precision, QualityProfile, RuntimeConfig

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

GIB = 1024**3

# Below these limits the "quality" profile (12-14B translation models, partly on the GPU)
# does not fit. A 16 GB machine reports roughly 15.7 GiB.
_QUALITY_MIN_RAM_GIB = 14.0
_QUALITY_MIN_VRAM_GIB = 5.5
_MIN_FREE_DISK_GIB = 25.0

_WINDOWS_PF_AVX2 = 40  # PF_AVX2_INSTRUCTIONS_AVAILABLE for IsProcessorFeaturePresent


class CpuInfo(BaseModel):
    name: str
    physical_cores: int | None
    logical_cores: int | None
    avx2: bool | None


class MemoryInfo(BaseModel):
    total_bytes: int
    available_bytes: int


class GpuInfo(BaseModel):
    index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int | None


class NvidiaInfo(BaseModel):
    driver_version: str | None = None
    gpus: list[GpuInfo] = []


class TorchInfo(BaseModel):
    installed: bool
    version: str | None = None
    cuda_build: str | None = None
    """CUDA version PyTorch was built for; None for the CPU-only build."""
    cuda_available: bool = False
    compute_capabilities: list[str] = []
    arch_list: list[str] = []
    """GPU architectures this PyTorch build has kernels for, e.g. "sm_75"."""
    error: str | None = None


class DiskInfo(BaseModel):
    path: str
    free_bytes: int | None


class HardwareReport(BaseModel):
    os: str
    python: str
    cpu: CpuInfo
    memory: MemoryInfo
    nvidia: NvidiaInfo
    torch: TorchInfo
    disk: DiskInfo


class Recommendation(BaseModel):
    device: Literal["cuda", "cpu"]
    precision: Precision
    profile: QualityProfile
    notes: list[str]


class DtypeCheck(BaseModel):
    dtype: str
    ok: bool
    max_relative_error: float | None = None
    matmul_tflops: float | None = None
    error: str | None = None


class GpuTestResult(BaseModel):
    device: str
    checks: list[DtypeCheck]

    def check(self, dtype: str) -> DtypeCheck | None:
        return next((c for c in self.checks if c.dtype == dtype), None)


def is_gtx16(gpu_name: str) -> bool:
    """GTX 16xx cards (Turing without tensor cores) are known to misbehave in fp16."""
    return re.search(r"GTX\s*16\d\d", gpu_name, re.IGNORECASE) is not None


# --- Probes -------------------------------------------------------------------------------


def detect_hardware(models_dir: Path) -> HardwareReport:
    return HardwareReport(
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        python=platform.python_version(),
        cpu=detect_cpu(),
        memory=detect_memory(),
        nvidia=detect_nvidia(),
        torch=detect_torch(),
        disk=detect_disk(models_dir),
    )


def detect_cpu() -> CpuInfo:
    name, avx2 = platform.processor() or "unknown", None
    try:
        if sys.platform == "win32":
            name, avx2 = _windows_cpu()
        elif sys.platform == "linux":
            name, avx2 = _linux_cpu()
    except Exception:
        logger.debug("CPU probe failed", exc_info=True)
    return CpuInfo(
        name=name,
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        avx2=avx2,
    )


def _windows_cpu() -> tuple[str, bool]:
    import winreg

    key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
        name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(_WINDOWS_PF_AVX2))
    return name, avx2


def _linux_cpu() -> tuple[str, bool | None]:
    name, avx2 = platform.processor() or "unknown", None
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            name = value.strip()
        elif key.strip() == "flags":
            avx2 = "avx2" in value.split()
            break
    return name, avx2


def detect_memory() -> MemoryInfo:
    memory = psutil.virtual_memory()
    return MemoryInfo(total_bytes=memory.total, available_bytes=memory.available)


def detect_nvidia() -> NvidiaInfo:
    """Query the NVIDIA driver directly, so GPUs show up even without a CUDA PyTorch build."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        logger.debug("NVML unavailable (no NVIDIA driver or GPU)", exc_info=True)
        return NvidiaInfo()
    try:
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GpuInfo(
                    index=index,
                    name=_text(pynvml.nvmlDeviceGetName(handle)),
                    total_vram_bytes=int(memory.total),
                    free_vram_bytes=int(memory.free),
                )
            )
        return NvidiaInfo(driver_version=_text(pynvml.nvmlSystemGetDriverVersion()), gpus=gpus)
    except Exception:
        logger.debug("NVML query failed", exc_info=True)
        return NvidiaInfo()
    finally:
        pynvml.nvmlShutdown()


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def detect_torch() -> TorchInfo:
    try:
        import torch
    except ImportError:
        return TorchInfo(installed=False)
    except Exception as exc:  # e.g. a broken CUDA DLL
        return TorchInfo(installed=False, error=f"{type(exc).__name__}: {exc}")

    info = TorchInfo(installed=True, version=torch.__version__, cuda_build=torch.version.cuda)
    try:
        info.cuda_available = torch.cuda.is_available()
        if info.cuda_available:
            info.compute_capabilities = [
                "{}.{}".format(*torch.cuda.get_device_capability(i))
                for i in range(torch.cuda.device_count())
            ]
            info.arch_list = list(torch.cuda.get_arch_list())
    except Exception as exc:
        info.cuda_available = False
        info.error = f"{type(exc).__name__}: {exc}"
    return info


def detect_disk(path: Path) -> DiskInfo:
    existing = path
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    try:
        free = shutil.disk_usage(existing).free
    except OSError:
        free = None
    return DiskInfo(path=str(path), free_bytes=free)


# --- Recommendation -----------------------------------------------------------------------


def recommend(report: HardwareReport, runtime: RuntimeConfig) -> Recommendation:
    notes: list[str] = []
    torch_info = report.torch
    gpu = report.nvidia.gpus[0] if report.nvidia.gpus else None

    cuda_usable = torch_info.cuda_available
    if not torch_info.installed:
        notes.append(
            "PyTorch is not installed or failed to load. Run `uv sync --extra cuda` "
            "(NVIDIA GPU) or `uv sync --extra cpu`."
        )
    elif gpu is not None and torch_info.cuda_build is None:
        notes.append(
            f"{gpu.name} was found but the CPU-only PyTorch build is installed. "
            "Run `uv sync --extra cuda` to use the GPU."
        )
    elif gpu is not None and not cuda_usable:
        notes.append(
            "PyTorch was built with CUDA but cannot use the GPU. Update the NVIDIA driver."
        )
    if cuda_usable and torch_info.compute_capabilities:
        arch = "sm_" + torch_info.compute_capabilities[0].replace(".", "")
        if torch_info.arch_list and arch not in torch_info.arch_list:
            cuda_usable = False
            notes.append(
                f"This PyTorch build has no kernels for the GPU ({arch}). "
                "Install a PyTorch build that supports it."
            )

    if runtime.device is Device.CPU:
        device: Literal["cuda", "cpu"] = "cpu"
    elif cuda_usable:
        device = "cuda"
    else:
        device = "cpu"
        if runtime.device is Device.CUDA:
            notes.append("device = 'cuda' is configured but CUDA is not usable; using the CPU.")

    precision = runtime.precision
    if precision is Precision.FP16:
        if device == "cpu":
            precision = Precision.FP32
            notes.append("fp16 brings no benefit on the CPU; using fp32.")
        elif gpu is not None and is_gtx16(gpu.name):
            notes.append(
                "fp16 is configured on a GTX 16xx card, which is known to produce broken "
                "results in fp16. Run `mangat doctor --test-gpu` or switch to fp32."
            )

    ram_gib = report.memory.total_bytes / GIB
    vram_gib = gpu.total_vram_bytes / GIB if gpu is not None and device == "cuda" else 0.0
    profile = QualityProfile.QUALITY
    if ram_gib < _QUALITY_MIN_RAM_GIB:
        profile = QualityProfile.BALANCED
        notes.append(
            f"{ram_gib:.1f} GiB of RAM is not enough for 12-14B translation models; "
            "the balanced profile uses 7-9B models."
        )
    elif device == "cpu":
        profile = QualityProfile.BALANCED
        notes.append(
            "Without a usable NVIDIA GPU, large translation models are very slow; "
            "the balanced profile is recommended."
        )
    elif vram_gib < _QUALITY_MIN_VRAM_GIB:
        profile = QualityProfile.BALANCED
        notes.append(f"{vram_gib:.1f} GiB of VRAM is little for the quality profile.")

    if report.cpu.avx2 is False:
        notes.append("This CPU has no AVX2: running language models on the CPU will be slow.")
    if report.disk.free_bytes is not None and report.disk.free_bytes / GIB < _MIN_FREE_DISK_GIB:
        notes.append(
            f"Only {report.disk.free_bytes / GIB:.1f} GiB free in {report.disk.path}. "
            "The models can take 10-25 GB."
        )
    return Recommendation(device=device, precision=precision, profile=profile, notes=notes)


# --- GPU test -----------------------------------------------------------------------------

_TOLERANCE = {"float32": 1e-3, "float16": 1e-2}
_MATMUL_SIZE = 2048
_BENCH_ITERATIONS = 10


def run_gpu_test(device: str = "cuda:0", size: int = _MATMUL_SIZE) -> GpuTestResult:
    """Check that matrix multiplication and convolution give correct, finite results.

    Each dtype is compared against a float64 CPU reference. A GPU that returns NaN or
    garbage in fp16 (a known GTX 16xx problem) shows up as a failed fp16 check.
    """
    import torch
    import torch.nn.functional as F

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, there is no GPU to test")

    generator = torch.Generator().manual_seed(0)
    a = torch.randn(size, size, generator=generator)
    b = torch.randn(size, size, generator=generator)
    side = max(size // 16, 8)
    images = torch.randn(1, 64, side, side, generator=generator)
    kernels = torch.randn(64, 64, 3, 3, generator=generator) * 0.05
    matmul_ref = a.double() @ b.double()
    conv_ref = F.conv2d(images.double(), kernels.double(), padding=1)

    checks = []
    for dtype in (torch.float32, torch.float16):
        name = str(dtype).removeprefix("torch.")
        try:
            da, db = a.to(target, dtype), b.to(target, dtype)
            matmul_out = (da @ db).double().cpu()
            conv_out = F.conv2d(images.to(target, dtype), kernels.to(target, dtype), padding=1)
            conv_out = conv_out.double().cpu()

            finite = bool(torch.isfinite(matmul_out).all() and torch.isfinite(conv_out).all())
            error = max(
                _relative_error(matmul_out, matmul_ref), _relative_error(conv_out, conv_ref)
            )
            checks.append(
                DtypeCheck(
                    dtype=name,
                    ok=finite and error < _TOLERANCE[name],
                    max_relative_error=error if finite else None,
                    matmul_tflops=_matmul_tflops(da, db),
                )
            )
        except Exception as exc:
            checks.append(DtypeCheck(dtype=name, ok=False, error=f"{type(exc).__name__}: {exc}"))

    label = torch.cuda.get_device_name(target) if target.type == "cuda" else device
    return GpuTestResult(device=label, checks=checks)


def interpret_gpu_test(result: GpuTestResult) -> list[str]:
    fp32, fp16 = result.check("float32"), result.check("float16")
    notes = []
    if fp32 is not None and not fp32.ok:
        notes.append(
            "The GPU gave wrong results even in fp32: update the driver and check for "
            "overclocking or overheating before using it."
        )
    if fp16 is not None and not fp16.ok:
        notes.append("fp16 gives wrong results on this GPU: keep precision = 'fp32'.")
    elif fp16 is not None and fp32 is not None and fp32.ok:
        speedup = (
            f" ({fp16.matmul_tflops / fp32.matmul_tflops:.1f}x the fp32 speed)"
            if fp16.matmul_tflops and fp32.matmul_tflops
            else ""
        )
        notes.append(
            f"fp16 works on this GPU{speedup}. fp32 remains the default because it is the "
            "safest choice for image quality."
        )
    return notes


def _relative_error(output: "torch.Tensor", reference: "torch.Tensor") -> float:
    return float((output - reference).abs().max() / reference.abs().max())


def _matmul_tflops(a: "torch.Tensor", b: "torch.Tensor") -> float:
    import torch

    def sync() -> None:
        if a.device.type == "cuda":
            torch.cuda.synchronize(a.device)

    a @ b  # warm-up
    sync()
    start = time.perf_counter()
    for _ in range(_BENCH_ITERATIONS):
        a @ b
    sync()
    elapsed = time.perf_counter() - start
    return 2 * a.shape[0] * a.shape[1] * b.shape[1] * _BENCH_ITERATIONS / elapsed / 1e12
