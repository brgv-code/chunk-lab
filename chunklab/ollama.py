"""Local Ollama detection.

LLM-based chunking is optional and never required. If Ollama isn't running,
the strategy is disabled in the UI and everything else still works. We probe
the local API (no external calls, no keys) and cache the answer briefly so a
re-render doesn't hammer it.
"""

import time
import urllib.request
import json

OLLAMA_HOST = "http://localhost:11434"
_CACHE: dict = {"ts": 0.0, "available": False, "models": []}
_TTL = 5.0  # seconds


def _probe() -> tuple[bool, list[str]]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1.5) as r:
            data = json.loads(r.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        return True, models
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False, []


def _refresh(force: bool = False) -> None:
    now = time.time()
    if force or (now - _CACHE["ts"]) > _TTL:
        available, models = _probe()
        _CACHE.update(ts=now, available=available, models=models)


def ollama_available() -> bool:
    _refresh()
    return _CACHE["available"]


def ollama_models() -> list[str]:
    _refresh()
    return list(_CACHE["models"])


def pick_model(preferred=("llama3.2", "llama3.1", "llama3", "mistral")) -> str | None:
    """Choose a model to use for LLM chunking: first preferred match, else the
    first installed model, else None."""
    models = ollama_models()
    if not models:
        return None
    for pref in preferred:
        for m in models:
            if m == pref or m.startswith(pref + ":"):
                return m
    return models[0]
