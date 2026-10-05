"""Make GPU acceleration work out of the box, and quietly survive without it.

Two separate problems, deliberately kept apart:

1. **CUDA DLL discovery.** The NVIDIA pip packages (``nvidia-cublas-cu12``,
   ``nvidia-cudnn-cu12``, ...) install ``cublas64_12.dll``/``cudnn64_9.dll``
   under ``site-packages/nvidia/<lib>/bin``. Nothing puts that on the loader
   path, so CTranslate2 and llama.cpp fail at ``init`` time with a bare
   ``cudart_error`` unless we add those directories first -- which has to
   happen before the engines are loaded, not before they are imported.

2. **Fallback.** A GPU build is never a hard requirement. Anything that is not
   explicitly disabled is attempted on the GPU first and retried on the CPU;
   the reason is logged once, at most, so a laptop without a usable NVIDIA
   driver still gets working (if slower) transcription and card generation.

Run this module directly to see what the current machine looks like::

    python utils/gpu_bootstrap.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Handles returned by os.add_dll_directory must stay alive for the process
# lifetime, otherwise Windows immediately forgets the directory again.
_dll_handles: List[object] = []
_registered = False


def _nvidia_dll_dirs() -> List[Path]:
    """Every ``nvidia/*/bin`` (win) or ``nvidia/*/lib`` (nix) in the venv."""
    dirs: List[Path] = []
    for entry in sys.path:
        nvidia = Path(entry) / "nvidia"
        if not nvidia.is_dir():
            continue
        for lib in nvidia.iterdir():
            if not lib.is_dir():
                continue
            for sub in ("bin", "lib", "lib64", "Library/bin"):
                candidate = lib / sub
                if candidate.is_dir():
                    dirs.append(candidate)
    return dirs


def enable_cuda_dlls() -> List[Path]:
    """Put the pip-installed CUDA libraries on the loader path. Idempotent."""
    global _registered
    if _registered:
        return []
    _registered = True

    dirs = _nvidia_dll_dirs()
    if not dirs:
        return []

    for path in dirs:
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                _dll_handles.append(os.add_dll_directory(str(path)))
            except OSError:
                pass
    current = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for path in dirs:
        if str(path) not in current:
            current.append(str(path))
    os.environ["PATH"] = os.pathsep.join(current)
    return dirs


def nvidia_smi_query(fields: str) -> Optional[str]:
    """Run `nvidia-smi --query-gpu=<fields>`; None when there is no nvidia-smi.

    The only reliable way to know an NVIDIA driver is present and new enough;
    on Windows the DLLs alone cannot tell us which GPU or driver we got.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.strip().splitlines()[0]


def describe_gpu() -> str:
    """One-line human description of the best available accelerator."""
    line = nvidia_smi_query("name,driver_version")
    if line:
        name, _, driver = line.partition(",")
        return f"{name.strip()} (driver {driver.strip()})"
    return "no NVIDIA GPU detected"


def cuda_device_count() -> int:
    """Number of CUDA devices CTranslate2 can see, 0 when unusable.

    Asked of CTranslate2 rather than of nvidia-smi, because CTranslate2 is what
    actually has to load the CUDA libraries.
    """
    enable_cuda_dlls()
    try:
        import ctranslate2
    except Exception:
        return 0
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception:
        return 0
    return count if isinstance(count, int) else 0


def prefer_gpu() -> bool:
    """True unless the user explicitly asked for CPU-only (``VOICE2ANKI_CPU``)."""
    return os.environ.get("VOICE2ANKI_CPU", "").strip().lower() not in {"1", "true", "yes"}


_reported: set = set()


def report_once(engine: str, message: str) -> None:
    """Log a GPU/CPU decision exactly once per engine."""
    if engine in _reported:
        return
    _reported.add(engine)
    from .logger import red

    red(message)


def gpu_then_cpu(engine: str, build_gpu, build_cpu) -> Tuple[object, bool]:
    """Build an engine on the GPU if possible, else on the CPU.

    ``build_gpu``/``build_cpu`` are zero-arg factories. Returns
    ``(instance, used_gpu)``. A GPU attempt that raises is logged and retried
    on the CPU, so a missing driver, missing DLLs or an unsupported compute
    capability all degrade to "slower but working".
    """
    enable_cuda_dlls()

    if prefer_gpu():
        try:
            instance = build_gpu()
        except Exception as err:
            report_once(
                engine,
                f"{engine}: GPU unavailable ({type(err).__name__}: {err}) - falling back to CPU.",
            )
        else:
            report_once(engine, f"{engine}: GPU acceleration active.")
            return instance, True

    try:
        return build_cpu(), False
    except Exception as err:
        report_once(engine, f"{engine}: CPU mode failed too ({type(err).__name__}: {err}).")
        raise


if __name__ == "__main__":
    dirs = enable_cuda_dlls()
    print(f"GPU: {describe_gpu()}")
    print(f"CUDA devices visible to CTranslate2: {cuda_device_count()}")
    print(f"NVIDIA DLL directories registered: {len(dirs)}")
    for path in dirs:
        print(f"  {path}")
    if not dirs:
        print("  (none: run 'python setup_env.py' to install the CUDA libraries)")