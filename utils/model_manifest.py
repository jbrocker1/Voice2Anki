"""Single source of truth for the local AI models Voice2Anki can run.

Deliberately free of third-party imports: ``setup_env.py`` must be able to
import this module with whatever interpreter the user happens to have, i.e.
before the virtualenv (and therefore platformdirs, huggingface_hub, gradio)
exists. The app-side wrappers in ``local_llm``/``local_stt``/
``local_embeddings`` import the same registry, so setup and runtime can never
disagree about a filename.

Weights live in the OS user-data directory, never in the repository: a clone
stays a few MB of text and the ~2.5 GB of weights are fetched on first run.
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

APP_DIRNAME = "Voice2Anki"
HF_PREFIX = "https://huggingface.co"

#: faster-whisper model ids (mapped onto Systran/faster-whisper-* repos)
STT: Dict[str, dict] = {
    "local:whisper-small": {
        "hf_model": "small",
        "size_mb": 480,
        "description": "faster-whisper small (multilingual, local, free)",
    },
    "local:whisper-base": {
        "hf_model": "base",
        "size_mb": 145,
        "description": "faster-whisper base (multilingual, local, faster/less accurate)",
    },
    "local:whisper-tiny": {
        "hf_model": "tiny",
        "size_mb": 75,
        "description": "faster-whisper tiny (multilingual, local, fastest/least accurate)",
    },
}
DEFAULT_STT = "local:whisper-small"

#: fastembed model ids
EMBED: Dict[str, dict] = {
    "local/bge-small-en-v1.5": {
        "hf_model": "BAAI/bge-small-en-v1.5",
        "size_mb": 130,
        "description": "BAAI bge-small-en-v1.5 (English, local, CPU)",
    },
}
DEFAULT_EMBED = "local/bge-small-en-v1.5"

#: local GGUF LLMs served through llama.cpp
LLM: Dict[str, dict] = {
    "local/qwen2.5-3b-instruct": {
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "url": f"{HF_PREFIX}/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_mb": 1850,
        "n_ctx": 16384,
        "max_input_tokens": 16384,
        "chat_format": "chatml",
        "description": "Qwen2.5 3B Instruct Q4_K_M (local, free, no API key)",
    },
    # Not fetched by setup, offered in the dropdown for users who want noticeably
    # better cards than 3B and have the RAM/disk for it (8 GB+ recommended).
    "local/qwen2.5-7b-instruct": {
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "url": f"{HF_PREFIX}/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_mb": 4680,
        "n_ctx": 16384,
        "max_input_tokens": 16384,
        "chat_format": "chatml",
        "description": "Qwen2.5 7B Instruct Q4_K_M (local, better quality, needs 8GB+ RAM)",
    },
    # Microsoft's instruction-tuned small model. Strong at structured output
    # and following long, rule-heavy system prompts (the kind cloze extraction
    # needs). Same memory class as the Qwen 3B but materially better at format
    # compliance -- in particular, the multi-card `#####` separator behavior we
    # added. GGUF conversion by bartowski (high-quality imatrix quant).
    "local/phi-3.5-mini-instruct": {
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "url": f"{HF_PREFIX}/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_mb": 2300,
        "n_ctx": 16384,
        "max_input_tokens": 16384,
        # bartowski's GGUF conversion was built so llama.cpp renders the
        # Phi-3.5 chat template via the built-in "chatml" chat format
        # (Phi-3.5 was trained on a ChatML-compatible template). The bare
        # "phi-3" chat format string is not a valid llama-cpp-python name
        # and silently falls back to raw templating.
        "chat_format": "chatml",
        "description": "Microsoft Phi-3.5 mini Instruct Q4_K_M (local, tuned for instruction following and structured output)",
    },
}
#: what setup_env.py prefetches. Phi-3.5-mini is now the default: in our
#: 5-fixture end-to-end harness it produced 4/5 PASS (vs Qwen 3B at 3/5)
#: and crucially passed the MIT OCW lecture opening that Qwen 3B failed
#: (Qwen echoed the prompt; Phi-3.5 produced a structured answer).
#: Same memory class (~2.2 GB Q4_K_M vs Qwen's 1.85 GB), but materially
#: better at following the cloze-format instructions in long transcripts.
PREFETCH_LLM = "local/phi-3.5-mini-instruct"
DEFAULT_LLM = PREFETCH_LLM


def models_root() -> Path:
    """Where model weights are cached. Outside the repo, on purpose."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / APP_DIRNAME / "models"


def llm_path(choice: str) -> Path:
    """Filesystem location of the GGUF backing a local LLM choice."""
    entry = LLM.get(choice)
    if entry is None:
        raise KeyError(f"'{choice}' is not a local LLM. Known: {sorted(LLM)}")
    return models_root() / entry["filename"]


def is_local_llm(choice: str) -> bool:
    return choice in LLM


def local_llm_info(choice: str) -> dict:
    """A litellm.model_cost-shaped row for a local model.

    Shaped like the cloud rows on purpose: the dropdown, the context-limit check
    and the cost accounting then need no special case for local inference, which
    is free. Lives here rather than in ``local_llm`` so that importing it costs
    no third-party dependency (``shared_module`` needs it at class-definition
    time, and ``shared_module`` is imported by ``logger``).
    """
    entry = LLM[choice]
    return {
        "max_input_tokens": entry["max_input_tokens"],
        "max_tokens": entry["max_input_tokens"],
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "local",
        "mode": "chat",
        "local_model": True,
    }


def is_local_stt(choice: str) -> bool:
    return choice in STT


def is_local_embed(choice: str) -> bool:
    return choice in EMBED


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _download(url: str, dest: Path, label: str) -> None:
    """Stream ``url`` to ``dest``, resuming a partial ``dest.part`` if present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for stale in (part, tmp):
        if stale.exists() and stale != part:
            stale.unlink()

    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": f"{APP_DIRNAME} setup"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        print(f"  resuming {label} at {human(resume_from)}", flush=True)

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            # a server that ignored our Range restarts the file
            mode = "ab" if resume_from and response.status == 206 else "wb"
            if mode == "wb":
                resume_from = 0
            total = response.headers.get("Content-Length")
            total = resume_from + int(total) if total else 0
            done = resume_from
            with open(part, mode) as fh:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        sys.stdout.write(f"\r  {label}: {human(done)}/{human(total)} ({pct:5.1f}%)")
                        sys.stdout.flush()
        if total:
            sys.stdout.write("\n")
    except urllib.error.URLError as err:
        raise RuntimeError(f"could not download {url}: {err}") from err

    shutil.move(str(part), str(tmp))
    shutil.move(str(tmp), str(dest))
    print(f"  {label}: done ({human(dest.stat().st_size)})", flush=True)


def ensure_llm(choice: str = DEFAULT_LLM) -> Path:
    """Download the GGUF for ``choice`` unless it is already cached."""
    entry = LLM.get(choice)
    if entry is None:
        raise KeyError(f"'{choice}' is not a local LLM. Known: {sorted(LLM)}")
    dest = llm_path(choice)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    print(f"Fetching local LLM weights ({human(entry['size_mb'] * 1024 * 1024)}):", flush=True)
    _download(entry["url"], dest, entry["filename"])
    return dest


def _slug(text: str) -> str:
    """Comparable form of a cache directory name ('v1.5' == 'v1-5')."""
    return "".join(c for c in text.lower() if c.isalnum())


def _fastembed_cache_dirs() -> List[str]:
    embed_dir = models_root() / "fastembed"
    try:
        return [p.name for p in embed_dir.iterdir() if p.is_dir()]
    except OSError:
        return []


def stt_cached(choice: str = DEFAULT_STT) -> bool:
    """Whether faster-whisper already has this model's weights on disk."""
    entry = STT.get(choice)
    if entry is None:
        return False
    snapshots = models_root() / "faster-whisper" / f"models--Systran--faster-whisper-{entry['hf_model']}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def embedding_cached(choice: str = DEFAULT_EMBED) -> bool:
    """Whether fastembed already has this model's weights on disk."""
    entry = EMBED.get(choice)
    if entry is None:
        return False
    wanted = _slug(entry["hf_model"].split("/")[-1])
    return any(wanted in _slug(name) for name in _fastembed_cache_dirs())


def ensure_stt(choice: str = DEFAULT_STT) -> None:
    """Prefetch a faster-whisper model into the shared models directory."""
    entry = STT.get(choice)
    if entry is None:
        raise KeyError(f"'{choice}' is not a local STT model. Known: {sorted(STT)}")
    if stt_cached(choice):
        return
    from faster_whisper import WhisperModel  # imported late: heavy, optional

    print(f"Fetching local STT weights ({human(entry['size_mb'] * 1024 * 1024)}):", flush=True)
    # device='cpu' on purpose: prefetching must not need a working GPU stack.
    model = WhisperModel(
        entry["hf_model"],
        device="cpu",
        compute_type="int8",
        download_root=str(models_root() / "faster-whisper"),
    )
    del model


def ensure_embedding(choice: str = DEFAULT_EMBED) -> None:
    """Prefetch a fastembed model into the shared models directory."""
    entry = EMBED.get(choice)
    if entry is None:
        raise KeyError(f"'{choice}' is not a local embedding model. Known: {sorted(EMBED)}")
    if embedding_cached(choice):
        return
    from fastembed import TextEmbedding  # imported late: heavy, optional

    print(f"Fetching local embedding weights ({human(entry['size_mb'] * 1024 * 1024)}):", flush=True)
    model = TextEmbedding(entry["hf_model"], cache_dir=str(models_root() / "fastembed"))
    # TextEmbedding is lazy: encoding something is what actually resolves the
    # weights (and fails if the download did not work).
    list(model.embed(["warmup"]))


def ensure_all(
    llm: str = PREFETCH_LLM,
    stt: str = DEFAULT_STT,
    embed: str = DEFAULT_EMBED,
) -> None:
    ensure_llm(llm)
    ensure_stt(stt)
    ensure_embedding(embed)


def total_download_mb(llm: str = PREFETCH_LLM, stt: str = DEFAULT_STT, embed: str = DEFAULT_EMBED) -> float:
    return sum(float(e["size_mb"]) for e in (LLM[llm], STT[stt], EMBED[embed]))


def _slug(text: str) -> str:
    """Comparable form of a cache directory name ('v1.5' == 'v1-5')."""
    return "".join(c for c in text.lower() if c.isalnum())


def status() -> List[Tuple[str, bool, str]]:
    """(label, present, detail) for every local model, for `--check`."""
    rows: List[Tuple[str, bool, str]] = []
    for choice, entry in LLM.items():
        path = llm_path(choice)
        if path.exists():
            rows.append((choice, True, f"{human(path.stat().st_size)} at {path}"))
        else:
            rows.append((choice, False, f"missing ({human(entry['size_mb'] * 1024 * 1024)})"))

    whisper_dir = models_root() / "faster-whisper"
    for choice, entry in STT.items():
        if stt_cached(choice):
            stem = f"models--Systran--faster-whisper-{entry['hf_model']}"
            rows.append((choice, True, f"cached in {whisper_dir / stem / 'snapshots'}"))
        else:
            rows.append((choice, False, f"missing ({human(entry['size_mb'] * 1024 * 1024)})"))

    embed_dir = models_root() / "fastembed"
    for choice, entry in EMBED.items():
        if embedding_cached(choice):
            wanted = _slug(entry["hf_model"].split("/")[-1])
            hit = next(c for c in _fastembed_cache_dirs() if wanted in _slug(c))
            rows.append((choice, True, f"cached in {embed_dir / hit}"))
        else:
            rows.append((choice, False, f"missing ({human(entry['size_mb'] * 1024 * 1024)})"))

    return rows


def register_env() -> None:
    """Point third-party downloaders at our models directory.

    Keeps huggingface_hub (used by both faster-whisper and fastembed) from
    scattering weights into a home-directory cache we never report on. An
    explicit HF_HOME from the user always wins.
    """
    os.environ.setdefault("HF_HOME", str(models_root() / "huggingface"))


register_env()

if __name__ == "__main__":
    print(f"models root: {models_root()}")
    for label, present, detail in status():
        print(f"  {'ok' if present else 'MISSING':<8} {label:<30} {detail}")