#!/usr/bin/env python3
"""One-shot, isolated dev-environment bootstrap for Voice2Anki.

Creates a project virtual environment WITHOUT depending on whatever interpreter
happens to be on your PATH. If `uv` is not already installed we fetch the
official standalone uv binary ourselves, then use it to install a pinned,
prebuilt CPython (Astral python-build-standalone) and build `.venv/` from it.

No admin rights required. Safe to re-run (idempotent).

Usage:
    python setup_env.py              # create .venv and install from uv.lock
    python setup_env.py --check      # probe the existing venv; report missing bits
    python setup_env.py --recreate   # wipe the existing venv and rebuild it
    python setup_env.py --dry-run    # print what would run, change nothing
    python setup_env.py --skip-models  # install packages but fetch no model weights
    python setup_env.py --cpu-only    # skip GPU detection and install CPU wheels
    python setup_env.py --backend cu125  # force a specific llama.cpp backend

If uv.lock is present the environment is synced to it exactly (reproducible);
otherwise requirements.txt ranges are used and a warning is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

IS_WIN = sys.platform == "win32"
REPO = Path(__file__).resolve().parent
UV_HOME = REPO / ".uv"
DEFAULT_PYTHON = "3.13"


def info(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def run(cmd, dry: bool) -> int:
    parts = [str(c) for c in cmd]
    printable = " ".join(p if len(p) <= 40 else p[:37] + "..." for p in parts)
    if dry:
        info(f"[dry-run] would run: {printable}")
        return 0
    info(f"+ {printable}")
    return subprocess.run(parts).returncode



def desired_python(cli: str | None) -> str:
    if cli:
        return cli
    pin = REPO / ".python-version"
    if pin.exists():
        lines = pin.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].strip():
            return lines[0].strip()
    return DEFAULT_PYTHON


def uv_exec_name() -> str:
    return "uv.exe" if IS_WIN else "uv"


def uv_local_path() -> Path:
    return UV_HOME / "bin" / uv_exec_name()


def find_uv(allow_bootstrap: bool, dry: bool) -> Path:
    found = shutil.which("uv")
    if found:
        return Path(found)
    local = uv_local_path()
    if local.exists():
        return local
    if dry:
        info(f"[dry-run] uv not found; would bootstrap the standalone uv into {local}")
        return local
    if not allow_bootstrap:
        die("uv not found on PATH. Re-run without --no-bootstrap to fetch it automatically.")
    return bootstrap_uv()


def _uv_asset() -> tuple[str, str]:
    machine = platform.machine().lower()
    arch = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(machine)
    if not arch:
        die(f"unsupported CPU architecture for uv bootstrap: {machine}")
    system = platform.system()
    if system == "Windows":
        return f"uv-{arch}-pc-windows-msvc.zip", "zip"
    if system == "Darwin":
        return f"uv-{arch}-apple-darwin.tar.gz", "targz"
    if system == "Linux":
        return f"uv-{arch}-unknown-linux-gnu.tar.gz", "targz"
    die(f"unsupported OS for uv bootstrap: {system}")


def bootstrap_uv() -> Path:
    asset, kind = _uv_asset()
    url = f"https://github.com/astral-sh/uv/releases/latest/download/{asset}"
    info(f"uv not found; downloading the official standalone uv from:\n  {url}")
    tmpdir = Path(tempfile.mkdtemp(prefix="v2a-uv-"))
    archive = tmpdir / asset
    request = urllib.request.Request(url, headers={"User-Agent": "Voice2Anki setup_env"})
    try:
        with urllib.request.urlopen(request, timeout=120) as resp, open(archive, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        if kind == "zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmpdir)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(tmpdir)
        exe = uv_exec_name()
        hits = [p for p in tmpdir.rglob(exe) if p.is_file()]
        if not hits:
            die(f"downloaded uv archive but could not find '{exe}' inside")
        bindir = UV_HOME / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        dest = bindir / exe
        shutil.copy2(hits[0], dest)
        if not IS_WIN:
            dest.chmod(0o755)
        return dest
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def venv_python(venv: Path) -> Path:
    return (venv / "Scripts" / "python.exe") if IS_WIN else (venv / "bin" / "python")


def venv_site_packages(venv: Path) -> Path:
    """Where to drop DLLs the pip packages do not provide (see cuda_runtime)."""
    if IS_WIN:
        return venv / "Lib" / "site-packages"
    return next((venv / "lib").glob("python3.*/site-packages"), venv / "lib" / "site-packages")


# ---------------------------------------------------------------------------
# Local AI: accelerator detection, llama.cpp backend, CUDA runtime, models.
#
# Nothing here is bundled in the repository; every byte is fetched at install
# time into .venv or the OS user-data directory.
# ---------------------------------------------------------------------------

#: llama-cpp-python publishes one wheel per accelerator on its own indexes, so
#: the wheel depends on the machine. A universal lockfile cannot express that,
#: which is why it is installed here instead of from uv.lock.
LLAMA_CPP_VERSION = "0.3.36"
LLAMA_CPP_INDEX = "https://abetlen.github.io/llama-cpp-python/whl"
LLAMA_CPP_BACKENDS = ("cpu", "cu125", "cu130", "metal")

#: minimal driver versions for each CUDA runtime we can use
MIN_DRIVER = {"cu125": (555,), "cu130": (580,)}

#: cuBLAS/cuDNN as pip wheels. Note the cu12/cu13 split: CTranslate2 (our STT
#: engine) links CUDA 12, while the cu130 llama.cpp wheel links CUDA 13.
CUDA_RUNTIME_PACKAGES = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
)

#: CUDA 13's cuBLAS has no PyPI wheel -- nvidia-cublas-cu13 exists but is an
#: empty 0.0.1 stub with no binaries. So for the cu130 backend we take NVIDIA's
#: own redistributable archive and unpack only the DLLs we need. This is the
#: documented redist for the CUDA version the wheel was built against.
CUBLAS13_REDIST = {
    "13.1.0.3": {
        "windows-x86_64": "libcublas/windows-x86_64/libcublas-windows-x86_64-13.1.0.3-archive.zip",
        "sha256": "4ac4847bbe4f7709b244956fcfc32197a2954ee70b155cb67eebd9ee26f7e339",
    },
}
CUBLAS13_BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"


def nvidia_smi() -> dict | None:
    """GPU name, driver and compute capability, or None without an NVIDIA GPU.

    The only trustworthy answer to "is there a usable NVIDIA GPU here": the
    pip-installed DLLs cannot tell us which card or which driver we got.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    fields = [f.strip() for f in out.stdout.splitlines()[0].split(",")]
    while len(fields) < 3:
        fields.append("")
    name, driver, cap = fields[:3]
    try:
        compute_cap = float(cap)
    except ValueError:
        # older drivers do not implement --query-gpu=compute_cap
        compute_cap = None
    return {
        "name": name,
        "driver": driver,
        "compute_cap": compute_cap,
        "driver_tuple": tuple(int(p) for p in driver.split(".")[:2] if p.isdigit()),
    }


def looks_like_blackwell(name: str) -> bool:
    """Fallback for drivers too old to report compute capability."""
    lowered = name.lower()
    return any(tag in lowered for tag in ("rtx 50", "rtx50", "b40", "b30", "gb20", "blackwell"))


def driver_new_enough(driver: tuple, backend: str) -> bool:
    minimum = MIN_DRIVER.get(backend)
    return minimum is None or driver >= minimum


def resolve_backend(override: str | None) -> tuple[str, list[str]]:
    """Pick the llama.cpp backend for this machine: (backend, notes to print)."""
    notes: list[str] = []
    if override and override != "auto":
        if override not in LLAMA_CPP_BACKENDS:
            die(f"unknown --backend '{override}'. Choose from: {', '.join(LLAMA_CPP_BACKENDS)}, auto")
        return override, [f"backend forced to '{override}' on request"]

    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "metal", ["Apple Silicon detected: using the Metal backend"]

    gpu = nvidia_smi()
    if gpu is None:
        if IS_WIN and platform.system() == "Windows":
            notes.append("No NVIDIA GPU detected (no nvidia-smi): using the CPU backend")
        return "cpu", notes

    cap = gpu["compute_cap"]
    blackwell = (cap is not None and cap >= 12.0) or (cap is None and looks_like_blackwell(gpu["name"]))
    backend = "cu130" if blackwell else "cu125"
    notes.append(f"GPU: {gpu['name']} (driver {gpu['driver']}, compute capability {cap or 'unknown'})")

    if not driver_new_enough(gpu["driver_tuple"], backend):
        wanted = ".".join(str(p) for p in MIN_DRIVER[backend])
        notes.append(
            f"driver {gpu['driver']} predates the {backend} runtime (needs >= {wanted}): "
            "falling back to CPU. Updating the NVIDIA driver will enable the GPU."
        )
        return "cpu", notes

    if backend == "cu130" and platform.system() != "Windows":
        # CUDA 13's cuBLAS is fetched from NVIDIA's zip redistributable, which
        # is only wired up for Windows here; the CPU wheel still runs.
        notes.append(
            "CUDA 13 runtime is only wired up for Windows in this installer: "
            "using the CPU backend.")
        return "cpu", notes

    notes.append(f"backend '{backend}'")
    return backend, notes


def install_llama_cpp(uv: Path, py: Path, backend: str, dry: bool) -> int:
    """Install the wheel matching `backend` from the official per-GPU index.

    Not in uv.lock on purpose: one checkout must be able to install the CPU
    wheel, a CUDA 12.5 wheel or a CUDA 13 wheel without carrying three lockfiles.
    """
    index = f"{LLAMA_CPP_INDEX}/{backend}/llama-cpp-python/"
    info(f"\nInstalling llama-cpp-python {LLAMA_CPP_VERSION} ({backend} backend)")
    return run([uv, "pip", "install", "--python", str(py), "--find-links", index,
                f"llama-cpp-python=={LLAMA_CPP_VERSION}"], dry=dry)


def install_cuda_runtime(uv: Path, py: Path, backend: str, venv: Path, dry: bool) -> None:
    """Put the CUDA libraries the engines link against where they can find them.

    ggml-cuda.dll (CUDA 13) and CTranslate2 (CUDA 12) load these at import/init
    time with no search-path override of their own, so a laptop without the
    CUDA Toolkit would otherwise fail at the first call.
    """
    packages = [p for p in CUDA_RUNTIME_PACKAGES]
    info("\nInstalling CUDA runtime libraries (pip wheels, no admin rights needed)...")
    rc = run([uv, "pip", "install", "--python", str(py), *packages], dry=dry)
    if rc:
        info("warning: the CUDA runtime packages failed to install. GPU acceleration")
        info("         will be unavailable; everything still runs on the CPU.")
        return

    if backend == "cu130":
        install_cublas13_redist(venv, dry=dry)


def install_cublas13_redist(venv: Path, dry: bool) -> None:
    """Unpack cublas64_13.dll from NVIDIA's official Windows redistributable.

    Skipped when the file is already there, so re-running setup is cheap.
    """
    version, platforms = next(iter(CUBLAS13_REDIST.items()))
    key = "windows-x86_64" if IS_WIN else "linux-x86_64"
    relative = platforms.get(key)
    if relative is None:
        info(f"warning: no cuBLAS {version} redistributable known for this platform;")
        info("         GPU inference may fall back to the CPU.")
        return

    target = venv_site_packages(venv) / "nvidia" / "cublas13" / "bin"
    already = list(target.glob("cublas64_13*")) if target.is_dir() else []
    if already:
        info(f"cuBLAS {version} already present in {target}")
        return

    url = CUBLAS13_BASE + relative
    info(f"\nFetching cuBLAS {version} from NVIDIA's redistributable archive (385 MB):")
    info(f"  {url}")
    if dry:
        info(f"[dry-run] would download, verify sha256, and unpack into {target}")
        return

    tmpdir = Path(tempfile.mkdtemp(prefix="v2a-cublas-"))
    try:
        archive = tmpdir / Path(relative).name
        request = urllib.request.Request(url, headers={"User-Agent": "Voice2Anki setup_env"})
        with urllib.request.urlopen(request, timeout=180) as resp, open(archive, "wb") as fh:
            shutil.copyfileobj(resp, fh)

        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != platforms["sha256"]:
            die(f"cuBLAS archive checksum mismatch (expected {platforms['sha256']}, got {digest}).")
        info("  checksum ok")

        target.mkdir(parents=True, exist_ok=True)
        extracted = 0
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name.lower().endswith(".dll"):
                    continue
                with zf.open(member) as src, open(target / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
        info(f"  unpacked {extracted} DLL(s) into {target}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def fetch_models(py: Path, dry: bool) -> None:
    """Pre-download the local model weights so the first launch works offline."""
    info("\nFetching local AI models (no API key ever needed; ~2.5 GB)...")
    if dry:
        info("[dry-run] would download the Qwen GGUF, faster-whisper and BGE-small weights")
        return
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from utils import model_manifest as m; "
        "print('  models directory:', m.models_root()); "
        "m.ensure_all()" % str(REPO)
    )
    rc = run([py, "-c", code], dry=False)
    if rc:
        info("warning: could not fetch the local models. The app will download them")
        info("         on first use instead, which takes longer.")


# Runs INSIDE the freshly built venv (imported/exec'd via `python -c`).
CHECK_CODE = r'''
import importlib, os, shutil, sys
from pathlib import Path
REPO = Path(sys.argv[1]).resolve()
mods = [
    ("gradio", "gradio"), ("pandas", "pandas"), ("numpy", "numpy"),
    ("opencv", "cv2"), ("rapidfuzz", "rapidfuzz"), ("joblib", "joblib"),
    ("platformdirs", "platformdirs"), ("beartype", "beartype"),
    ("aiohttp", "aiohttp"), ("beautifulsoup4", "bs4"), ("rtoml", "rtoml"),
    ("fire", "fire"), ("tqdm", "tqdm"), ("pyclip", "pyclip"),
    ("openai", "openai"), ("litellm", "litellm"), ("scikit-learn", "sklearn"),
    ("ankipandas", "ankipandas"), ("py_ankiconnect", "py_ankiconnect"),
    ("pydub", "pydub"), ("scipy", "scipy"), ("deepgram-sdk", "deepgram"),
    ("async_cache", "cache"), ("pillow", "PIL"),
    ("static-ffmpeg", "static_ffmpeg"),
    ("ocr_with_format", "OCR_with_format"),
    ("faster-whisper", "faster_whisper"), ("fastembed", "fastembed"),
]
rows = []
for label, imp in mods:
    try:
        importlib.import_module(imp)
        rows.append((label, "ok", "", False))
    except Exception as e:
        rows.append((label, "MISSING", type(e).__name__, False))

# local AI: the CUDA libraries must be on the loader path *before*
# llama_cpp is imported -- ggml.dll links ggml-cuda.dll, which links cuBLAS.
sys.path.insert(0, str(REPO))
dll_dirs = []
try:
    from utils.gpu_bootstrap import describe_gpu, enable_cuda_dlls, cuda_device_count
    dll_dirs = enable_cuda_dlls()
    cuda_count = cuda_device_count()
except Exception:
    cuda_count = 0
try:
    import llama_cpp
    rows.append(("llama-cpp-python (local LLM)", "ok", llama_cpp.__version__, False))
except Exception as e:
    rows.append(("llama-cpp-python (local LLM)", "MISSING", type(e).__name__, False))
try:
    detail = f"{describe_gpu()}; {cuda_count} CUDA device(s); {len(dll_dirs)} NVIDIA runtime dir(s)"
    if cuda_count > 0:
        rows.append(("GPU acceleration", "ok", detail, False))
    elif dll_dirs:
        rows.append(("GPU acceleration", "ok", detail + " - CPU fallback", True))
    else:
        rows.append(("GPU acceleration", "MISSING", "no NVIDIA runtime installed (CPU only)", True))
except Exception as e:
    rows.append(("GPU acceleration", "MISSING", type(e).__name__, True))
try:
    from utils import model_manifest as m
    for label, present, detail in m.status():
        rows.append((label, "ok" if present else "MISSING", detail, not present))
except Exception as e:
    rows.append(("local models", "MISSING", type(e).__name__, True))

# resolve ffmpeg exactly the way the app does (system install wins, static fallback)
try:
    from utils.ffmpeg_bootstrap import ensure_ffmpeg
    resolved = ensure_ffmpeg()
    if resolved:
        rows.append(("ffmpeg (binary)", "ok", os.path.basename(resolved[0]), False))
    else:
        rows.append(("ffmpeg (binary)", "MISSING", "no system or static build", False))
except Exception as e:
    rows.append(("ffmpeg (binary)", "MISSING", type(e).__name__, False))
present = shutil.which("tesseract") is not None
note = "" if present else "optional: OCR mode only"
rows.append(("tesseract (binary)", "ok" if present else "MISSING", note, True))
w = max(len(r[0]) for r in rows)
print("  " + "component".ljust(w) + "  status    detail")
print("  " + "-" * w + "  ------    ------")
hard = 0
for label, status, note, warn_only in rows:
    print(f"  {label:<{w}}  {status:<7}  {note}")
    if status != "ok" and not warn_only:
        hard += 1
if hard:
    print(f"\n  {hard} required component(s) missing -> environment is NOT ready.")
    sys.exit(1)
print("\n  all required components present.")
sys.exit(0)
'''


def do_check(py: Path) -> int:
    info("\n== Voice2Anki environment check ==")
    info(f"venv python: {py}")
    return run([py, "-c", CHECK_CODE, str(REPO)], dry=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Bootstrap an isolated Python environment for Voice2Anki.")
    ap.add_argument("--python", help="Python version to install (default: .python-version, else 3.13)")
    ap.add_argument("--venv", default=str(REPO / ".venv"), help="venv location (default: ./.venv)")
    ap.add_argument("--recreate", action="store_true", help="delete an existing venv before rebuilding")
    ap.add_argument("--check", action="store_true", help="only probe the existing venv; do not install")
    ap.add_argument("--no-bootstrap", action="store_true", help="require uv on PATH instead of fetching it")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, change nothing")
    ap.add_argument("--skip-models", action="store_true",
                    help="install packages but do not fetch the local model weights (~2.5 GB)")
    ap.add_argument("--cpu-only", action="store_true",
                    help="skip GPU detection and install the CPU-only llama.cpp wheel")
    ap.add_argument("--backend", default="auto",
                    help=f"llama.cpp backend: auto (default), or one of {', '.join(LLAMA_CPP_BACKENDS)}")
    args = ap.parse_args(argv)

    pyver = desired_python(args.python)
    venv = Path(args.venv)
    py = venv_python(venv)
    req = REPO / "requirements.txt"
    lock = REPO / "uv.lock"

    if args.check:
        if not py.exists():
            die(f"no venv at {venv}. Run 'python setup_env.py' first (or drop --check).")
        return do_check(py)

    if not req.exists() and not lock.exists():
        die(f"neither requirements.txt nor uv.lock found next to setup_env.py ({REPO})")

    uv = find_uv(allow_bootstrap=not args.no_bootstrap, dry=args.dry_run)

    if args.recreate and venv.exists():
        if args.dry_run:
            info(f"[dry-run] would remove existing venv {venv}")
        else:
            info(f"removing existing venv {venv}")
            shutil.rmtree(venv)

    rc = run([uv, "python", "install", pyver], args.dry_run)
    if rc:
        return rc

    if not venv.exists() or args.recreate:
        rc = run([uv, "venv", str(venv), "--python", pyver], args.dry_run)
        if rc:
            return rc

    if lock.exists():
        # sync makes the venv match the lock exactly, and prunes anything extra.
        # It runs BEFORE the llama-cpp-python step below, because pruning is
        # exactly what would undo that install (its wheel is not in the lock).
        rc = run([uv, "pip", "sync", "--python", str(py), str(lock)], args.dry_run)
    else:
        info("warning: uv.lock not found; installing from the ranges in requirements.txt.")
        info("         The result will not be reproducible. Run 'uv pip compile")
        info("         requirements.txt -o uv.lock --universal' to create one.")
        rc = run([uv, "pip", "install", "--python", str(py), "-r", str(req)], args.dry_run)
    if rc:
        return rc

    # Decide the accelerator before installing llama-cpp-python: its wheel
    # depends on the machine, so it cannot come from the universal lock.
    backend, notes = resolve_backend("cpu" if args.cpu_only else args.backend)
    info("\n== Local AI backend ==")
    for note in notes:
        info(f"  {note}")

    rc = install_llama_cpp(uv, py, backend, args.dry_run)
    if rc:
        return rc

    if backend in MIN_DRIVER:
        install_cuda_runtime(uv, py, backend, venv, args.dry_run)
    else:
        info("\nNo CUDA runtime needed for the CPU/Metal backend.")

    if not args.skip_models:
        fetch_models(py, args.dry_run)

    info("\nFetching bundled ffmpeg/ffprobe (no system install required)...")
    rc = run([py, "-c", "from static_ffmpeg import run as r; print('  ffmpeg :', r.get_or_fetch_platform_executables_else_raise()[0])"], dry=args.dry_run)
    if rc:
        info("warning: could not fetch the bundled ffmpeg. Audio decoding will fail until")
        info("         ffmpeg is installed system-wide, or setup is re-run.")

    info("\nDone. Next steps:")
    if IS_WIN:
        info(f"  activate:  & '{venv / 'Scripts' / 'activate.ps1'}'")
    else:
        info(f"  activate:  source {venv / 'bin' / 'activate'}")
    info(f"  run:       {py} Voice2Anki.py    (or: uv run --python {venv} Voice2Anki.py)")
    info("  verify:    python setup_env.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
