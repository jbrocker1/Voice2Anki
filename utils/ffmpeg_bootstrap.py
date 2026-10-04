"""Locate ffmpeg/ffprobe for pydub, installing nothing system-wide.

pydub resolves its encoder exactly once, at import time
(``AudioSegment.converter = get_encoder_name()``), by scanning ``PATH`` for
``ffmpeg``. There is no ``FFMPEG_BINARY``-style override, so this module has to
run *before* pydub is imported -- see ``utils/__init__.py``.

A system install always wins. Only when ffmpeg (or ffprobe) is missing do we fall
back to the static binaries shipped by the ``static-ffmpeg`` package, which are
downloaded into the virtualenv on first use and need no admin rights.
"""
from __future__ import annotations

import os
import shutil
import warnings
from typing import Optional, Tuple

Resolved = Tuple[str, str]

_resolved: Optional[Resolved] = None


def _static_binaries() -> Optional[Resolved]:
    try:
        from static_ffmpeg import run
    except ImportError:
        return None
    try:
        return tuple(run.get_or_fetch_platform_executables_else_raise())
    except Exception:
        return None


def ensure_ffmpeg() -> Optional[Resolved]:
    """Make ffmpeg and ffprobe resolvable on PATH. Returns their paths, or None.

    Idempotent, and safe to call repeatedly. Never raises: a failure here would
    only break audio decoding later, with a much more confusing traceback.
    """
    global _resolved
    if _resolved is not None:
        return _resolved

    system_ffmpeg = shutil.which("ffmpeg")
    system_ffprobe = shutil.which("ffprobe")
    if system_ffmpeg and system_ffprobe:
        _resolved = (system_ffmpeg, system_ffprobe)
        return _resolved

    binaries = _static_binaries()
    if binaries is None:
        warnings.warn(
            "ffmpeg/ffprobe not found, and the 'static-ffmpeg' package could not "
            "provide them. Audio decoding will fail -- run `python setup_env.py` "
            "or install ffmpeg system-wide.",
            RuntimeWarning,
        )
        return None

    ffmpeg, ffprobe = binaries
    # Append rather than prepend, so a partially-present system install keeps
    # winning for whichever of the two it actually provides.
    bin_dir = os.path.dirname(ffmpeg)
    current = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if bin_dir not in current:
        os.environ["PATH"] = os.pathsep.join(current + [bin_dir])

    _resolved = (ffmpeg, ffprobe)
    return _resolved


if __name__ == "__main__":
    for label, path in zip(("ffmpeg", "ffprobe"), ensure_ffmpeg() or (None, None)):
        print(f"{label}: {path}")