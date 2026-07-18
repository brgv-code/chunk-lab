"""Strategy registry + dispatch.

A chunker is a function ``(text, *, chunk_size, overlap) -> ChunkResult`` that
registers itself with ``@register(...)``. Both stages (visualizer and eval)
read the same registry, so adding a strategy is a one-function change and it
shows up everywhere automatically.
"""

from .types import ChunkResult
from .ollama import ollama_available

_REGISTRY: dict[str, dict] = {}


def register(key, label, description, needs_ollama=False):
    """Decorator: register a chunker function under ``key``."""
    def deco(fn):
        _REGISTRY[key] = dict(
            key=key,
            fn=fn,
            label=label,
            description=description,
            needs_ollama=needs_ollama,
        )
        return fn
    return deco


def _ensure_loaded():
    """Import the chunker modules so their @register calls run (once)."""
    if _REGISTRY:
        return
    # Importing the package runs every module's register() side effects.
    from . import chunkers  # noqa: F401


def available_strategies() -> list[dict]:
    """All registered strategies, in registration order, annotated with whether
    they're currently runnable (Ollama-gated ones flip to unavailable when
    Ollama isn't reachable)."""
    _ensure_loaded()
    out = []
    have_ollama = ollama_available()
    for spec in _REGISTRY.values():
        available = (not spec["needs_ollama"]) or have_ollama
        out.append(dict(
            key=spec["key"],
            label=spec["label"],
            description=spec["description"],
            needs_ollama=spec["needs_ollama"],
            available=available,
            note="" if available else "disabled: start Ollama to enable",
        ))
    return out


def run_strategy(key, text, *, chunk_size, overlap, **kwargs) -> ChunkResult:
    """Dispatch to a registered chunker. Never raises for an expected failure
    (disabled strategy, or a chunker that errors on this input), those come
    back as ``ChunkResult(ok=False, note=...)`` so a single bad column can't
    take down the whole app."""
    _ensure_loaded()
    spec = _REGISTRY.get(key)
    if spec is None:
        return ChunkResult(strategy=key, chunks=[], params={}, ok=False,
                           note=f"unknown strategy: {key}")

    if spec["needs_ollama"] and not ollama_available():
        return ChunkResult(strategy=spec["label"], chunks=[], params={},
                           ok=False, note="disabled: start Ollama to enable")

    try:
        return spec["fn"](text, chunk_size=chunk_size, overlap=overlap, **kwargs)
    except Exception as e:  # noqa: BLE001 - surface, don't crash the UI
        return ChunkResult(strategy=spec["label"], chunks=[], params={},
                           ok=False, note=f"error: {type(e).__name__}: {e}")


def strategy_label(key) -> str:
    _ensure_loaded()
    spec = _REGISTRY.get(key)
    return spec["label"] if spec else key
