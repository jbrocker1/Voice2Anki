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

If uv.lock is present the environment is synced to it exactly (reproducible);
otherwise requirements.txt ranges are used and a warning is printed.
"""
from __future__ import annotations

import argparse
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
]
rows = []
for label, imp in mods:
    try:
        importlib.import_module(imp)
        rows.append((label, "ok", "", False))
    except Exception as e:
        rows.append((label, "MISSING", type(e).__name__, False))
# resolve ffmpeg exactly the way the app does (system install wins, static fallback)
try:
    sys.path.insert(0, str(REPO))
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
        # sync makes the venv match the lock exactly, and prunes anything extra
        rc = run([uv, "pip", "sync", "--python", str(py), str(lock)], args.dry_run)
    else:
        info("warning: uv.lock not found; installing from the ranges in requirements.txt.")
        info("         The result will not be reproducible. Run 'uv pip compile")
        info("         requirements.txt -o uv.lock --universal' to create one.")
        rc = run([uv, "pip", "install", "--python", str(py), "-r", str(req)], args.dry_run)
    if rc:
        return rc

    if args.dry_run:
        return 0

    # Fetch the static ffmpeg/ffprobe now so the first app launch works offline.
    info("\nFetching bundled ffmpeg/ffprobe (no system install required)...")
    rc = run([py, "-c", "from static_ffmpeg import run as r; print('  ffmpeg :', r.get_or_fetch_platform_executables_else_raise()[0])"], dry=False)
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
