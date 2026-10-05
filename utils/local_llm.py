"""One entry point for LLM calls: local llama.cpp, or any cloud via litellm.

Local models are served by llama.cpp in-process (no server, no daemon, no port),
which is what makes "no login, no API key" possible. Two things the naive
version gets wrong and this module therefore handles:

* **Thread safety.** ``alfred`` can fan out over a thread pool when a transcript
  contains several ``#####`` splits. A llama.cpp context is not re-entrant, so
  loading and generation both go through one lock.
* **First-call latency.** Loading a 3B model takes a few seconds and ~2 GB of
  RAM. Models are loaded lazily and then cached for the whole session.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from . import model_manifest as manifest
from .gpu_bootstrap import enable_cuda_dlls, gpu_then_cpu
from .logger import red, whi
from .typechecker import optional_typecheck

# Guards both the model cache and llama.cpp contexts (which are not re-entrant).
_llm_lock = threading.Lock()
_llama: Dict[str, Any] = {}
_tokenizers: Dict[str, Any] = {}

#: enough for a cloze card or two; the app never asked cloud models for a cap
#: either, and a runaway local generation would hang the UI thread pool.
DEFAULT_MAX_TOKENS = 1536


class LocalChatResponse(dict):
    """A local completion that quacks like litellm's response object.

    ``alfred`` reads ``response["choices"][0]["finish_reason"]`` and later calls
    ``response.json()`` to store the raw answer for the fine-tuning buffer, both
    of which assume the OpenAI SDK object the cloud path returns. Subclassing
    ``dict`` keeps those call sites working unchanged for local models.
    """

    def json(self) -> str:
        return json.dumps(dict(self), ensure_ascii=False)


def _close_all() -> None:
    """Release llama.cpp contexts while the interpreter is still alive.

    Left to garbage collection, ``Llama.__del__`` runs during interpreter
    shutdown, when ``llama_cpp._internals`` globals have already been replaced
    by None, and prints a confusing ``TypeError: 'NoneType' object is not
    callable`` traceback every single time the app is closed.
    """
    for handle in list(_llama.values()):
        try:
            handle.close()
        except BaseException:
            # KeyboardInterrupt lands here when the user stops the app with
            # Ctrl+C mid-close: swallow it so the atexit callback does not
            # abort and leave the models to explode in __del__ instead
            pass
    _llama.clear()


atexit.register(_close_all)


def is_local_llm(model: str) -> bool:
    """True when ``model`` is served by llama.cpp instead of a cloud provider."""
    return manifest.is_local_llm(model)


@optional_typecheck
def _load_llama(choice: str, n_gpu_layers: int) -> Any:
    # Must happen before `import llama_cpp`: on Windows ggml.dll links
    # ggml-cuda.dll, which links cuBLAS, and those live in site-packages/nvidia
    # instead of next to the extension. Token counting asks for a tokenizer long
    # before the first completion, so the registration cannot live only inside the
    # GPU/CPU fallback.
    enable_cuda_dlls()
    from llama_cpp import Llama

    path = manifest.ensure_llm(choice)
    entry = manifest.LLM[choice]
    whi(f"Loading local LLM {choice} from {path} (n_gpu_layers={n_gpu_layers})")
    return Llama(
        model_path=str(path),
        n_ctx=int(entry["n_ctx"]),
        n_gpu_layers=n_gpu_layers,
        chat_format=entry.get("chat_format", "chatml"),
        logits_all=False,
        verbose=False,
    )


@optional_typecheck
def get_llama(choice: str) -> Any:
    """The loaded llama.cpp handle for ``choice``, loading it if needed."""
    with _llm_lock:
        if choice not in _llama:
            _llama[choice], _ = gpu_then_cpu(
                f"LLM {choice}",
                lambda: _load_llama(choice, -1),   # offload every layer
                lambda: _load_llama(choice, 0),    # CPU only
            )
        return _llama[choice]


@optional_typecheck
def _load_tokenizer(choice: str) -> Any:
    """Vocabulary-only handle: enough to count tokens, no weights, no GPU."""
    enable_cuda_dlls()
    from llama_cpp import Llama

    path = manifest.ensure_llm(choice)
    whi(f"Loading local tokenizer for {choice} (vocab only)")
    return Llama(model_path=str(path), vocab_only=True, verbose=False)


@optional_typecheck
def _count_tokens(choice: str, text: str) -> int:
    """Tokens in ``text`` per the local model's own tokenizer.

    Falls back to a characters/4 estimate when the GGUF is unavailable (e.g.
    first run before setup fetched it): over-estimating is much safer than
    under-estimating a context window.
    """
    if not text:
        return 0
    try:
        if choice not in _tokenizers:
            _tokenizers[choice] = _load_tokenizer(choice)
        handle = _tokenizers[choice]
        # n_tokens is a plain int on some llama-cpp-python builds and a method
        # on others, so go through tokenize() which is stable across versions.
        tokenize = getattr(handle, "tokenize", None)
        if callable(tokenize):
            return len(tokenize(text.encode("utf-8"), add_bos=False, special=False))
        n_tokens = getattr(handle, "n_tokens", None)
        return int(n_tokens(text.encode("utf-8")) if callable(n_tokens) else n_tokens)
    except Exception as err:
        red(f"Falling back to an estimated token count for {choice} ({type(err).__name__}: {err})")
        return max(1, len(text) // 4)


@optional_typecheck
def count_tokens(model: str, text: str) -> int:
    """Token count for ``model``, local or cloud."""
    if manifest.is_local_llm(model):
        with _llm_lock:
            return _count_tokens(model, text)
    import litellm

    return int(litellm.token_counter(model=model, text=text))


@optional_typecheck
def local_completion(
    choice: str,
    messages: List[dict],
    temperature: float,
    max_tokens: Optional[int] = None,
) -> dict:
    """Run one local chat completion and return a litellm-shaped response."""
    handle = get_llama(choice)
    kwargs: Dict[str, Any] = {
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens or DEFAULT_MAX_TOKENS),
    }
    with _llm_lock:
        response = handle.create_chat_completion(**kwargs)

    raw_choice = response["choices"][0]
    text = raw_choice["message"]["content"] or ""
    usage = response.get("usage") or {}
    # llama.cpp does report usage, but never trust a missing key when the
    # caller is about to add it to a running total.
    prompt_tokens = int(usage.get("prompt_tokens") or _count_tokens(choice, "\n".join(
        str(m.get("content", "")) for m in messages
    )))
    completion_tokens = int(usage.get("completion_tokens") or _count_tokens(choice, text))

    return LocalChatResponse({
        "id": f"local-{os.getpid()}-{id(response) & 0xFFFF:04x}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": choice,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            # 'length' when max_tokens cut it off, 'stop' otherwise. alfred
            # warns the user when this is anything else.
            "finish_reason": raw_choice.get("finish_reason") or "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


@optional_typecheck
def completion(
    model: str,
    messages: List[dict],
    temperature: float = 0.0,
    **kwargs,
) -> dict:
    """Single place to call whichever LLM is selected.

    ``alfred`` and ``audio_edit`` both go through here, so adding a local model
    costs nothing at the call sites.
    """
    if manifest.is_local_llm(model):
        return local_completion(
            model,
            messages,
            temperature,
            max_tokens=kwargs.get("max_tokens"),
        )
    import litellm

    return litellm.completion(model=model, messages=messages, temperature=temperature, **kwargs)