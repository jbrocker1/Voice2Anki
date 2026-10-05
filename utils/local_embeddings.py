"""Local embeddings through fastembed (ONNX Runtime), no API key.

Deliberately CPU-only, unlike STT and the LLM: BGE-small encodes a handful of
short strings per card, which takes single-digit milliseconds on any CPU.
Routing it through CUDA would mean shipping a second GPU runtime (ONNX Runtime
GPU) for a difference nobody can perceive, so ``utils/gpu_bootstrap`` is not
consulted here on purpose.

The vector shape returned matches the cloud branches in ``utils/memory.py``:
a ``(1, dim)`` float array, L2-normalised when requested.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List

import numpy as np

from . import model_manifest as manifest
from .logger import whi
from .typechecker import optional_typecheck

_embed_lock = threading.Lock()
_models: Dict[str, Any] = {}

#: fastembed batches internally, but staying modest keeps peak RAM low on the
#: small laptops this project targets.
BATCH_SIZE = 32


@optional_typecheck
def is_local_embed(choice: str) -> bool:
    """True when ``choice`` is embedded locally instead of by a cloud API."""
    return manifest.is_local_embed(choice)


@optional_typecheck
def get_model(choice: str) -> Any:
    """The loaded fastembed handle for ``choice``, loading it if needed."""
    with _embed_lock:
        if choice not in _models:
            from fastembed import TextEmbedding

            entry = manifest.EMBED[choice]
            manifest.ensure_embedding(choice)
            whi(f"Loading local embedding model {choice} (CPU)")
            _models[choice] = TextEmbedding(
                entry["hf_model"],
                cache_dir=str(manifest.models_root() / "fastembed"),
            )
        return _models[choice]


@optional_typecheck
def embed(choice: str, text_list: List[str]) -> List[np.ndarray]:
    """Embed ``text_list`` locally, returning one ``(1, dim)`` array per text."""
    model = get_model(choice)
    vectors = [np.asarray(v, dtype=np.float32).reshape(1, -1) for v in model.embed(text_list, batch_size=BATCH_SIZE)]
    assert len(vectors) == len(text_list), f"Expected {len(text_list)} vectors, got {len(vectors)}"
    return vectors