"""Local speech-to-text through faster-whisper (CTranslate2), no API key.

The cloud branches in ``utils/main.py`` return an OpenAI-shaped transcript dict;
:func:`transcribe` returns the same two fields the rest of the app reads
(``text``, ``duration``) so callers do not need to know which engine ran.

faster-whisper is loaded lazily (it is a ~50 MB import plus a ~500 MB model) and
cached per model id. GPU (float16) is attempted first and int8 CPU is the
fallback, decided by ``utils.gpu_bootstrap``.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from . import model_manifest as manifest
from .gpu_bootstrap import gpu_then_cpu
from .logger import red, whi
from .typechecker import optional_typecheck

_stt_lock = threading.Lock()
_models: Dict[str, Any] = {}

#: faster-whisper decodes with a temperature and retries with each next value
#: until the transcript stops being implausible. This is its own default order;
#: we reuse it when the user leaves the temperature slider at 0 ("auto").
TEMPERATURE_FALLBACKS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@optional_typecheck
def is_local_stt(choice: str) -> bool:
    """True when ``choice`` is transcribed locally instead of by a cloud API."""
    return manifest.is_local_stt(choice)


@optional_typecheck
def _load_model(choice: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    entry = manifest.STT[choice]
    manifest.ensure_stt(choice)
    whi(f"Loading local STT {choice} on {device} ({compute_type})")
    return WhisperModel(
        entry["hf_model"],
        device=device,
        compute_type=compute_type,
        download_root=str(manifest.models_root() / "faster-whisper"),
    )


@optional_typecheck
def get_model(choice: str) -> Any:
    """The loaded faster-whisper handle for ``choice``, loading it if needed."""
    with _stt_lock:
        if choice not in _models:
            _models[choice], _ = gpu_then_cpu(
                f"STT {choice}",
                lambda: _load_model(choice, "cuda", "float16"),
                lambda: _load_model(choice, "cpu", "int8"),
            )
        return _models[choice]


@optional_typecheck
def _temperatures(requested: float) -> List[float]:
    """Whisper wants a retry ladder, not a single value."""
    if requested and requested > 0:
        return sorted({float(requested), *TEMPERATURE_FALLBACKS})
    return list(TEMPERATURE_FALLBACKS)


@optional_typecheck
def transcribe(
    choice: str,
    audio_path: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Transcribe ``audio_path`` locally, returning an OpenAI-shaped transcript.

    ``language`` is an ISO-639-1 code (``en``, ``fr``, ...) or None to detect it.
    """
    model = get_model(choice)

    segments, info = model.transcribe(
        str(audio_path),
        language=language or None,
        initial_prompt=prompt or None,
        temperature=_temperatures(temperature),
        beam_size=5,
        vad_filter=True,
    )
    # faster-whisper is lazy: nothing has been decoded until we iterate.
    collected = [
        {"start": s.start, "end": s.end, "text": s.text}
        for s in segments
    ]
    text = "".join(s["text"] for s in collected).strip()
    if not text:
        red(f"Local whisper returned no speech for {audio_path}")

    return {
        "text": text,
        "duration": float(info.duration or 0.0),
        "language": info.language,
        "segments": collected,
    }