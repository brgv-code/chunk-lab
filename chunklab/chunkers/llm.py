"""LLM-based chunking: optional, gated behind local Ollama.

Chonkie's SlumberChunker hands the document to a language model and asks it to
mark boundaries. We point it at a local Ollama model through Ollama's
OpenAI-compatible endpoint, so there are no API keys and nothing leaves the
machine. If Ollama isn't running (or has no model), the strategy reports itself
disabled and the rest of the app is unaffected. The registry gate already
blocks this when Ollama is unreachable; the inner checks are belt-and-braces.
"""

from ..registry import register
from ..types import ChunkResult
from ._common import to_chunks
from ..ollama import ollama_available, pick_model, OLLAMA_HOST

TOKENIZER = "gpt2"


@register(
    "llm", "LLM-based",
    "Hand the document to a local LLM (via Ollama) and let it mark the "
    "boundaries. Highest ceiling, slowest. Disabled unless Ollama is running.",
    needs_ollama=True,
)
def llm(text, *, chunk_size, overlap, **_):
    if not ollama_available():
        return ChunkResult("LLM-based", [], {}, ok=False,
                           note="disabled: start Ollama to enable")
    model = pick_model()
    if not model:
        return ChunkResult("LLM-based", [], {}, ok=False,
                           note="disabled: no Ollama model installed "
                                "(try: ollama pull llama3.2)")

    from chonkie import SlumberChunker
    from chonkie.genie import OpenAIGenie

    genie = OpenAIGenie(model=model, base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")
    chunker = SlumberChunker(genie=genie, tokenizer=TOKENIZER,
                             chunk_size=int(chunk_size), verbose=False)
    chunks = chunker(text)
    return ChunkResult(
        strategy="LLM-based",
        chunks=to_chunks(chunks, base_meta={"model": model}),
        params={"model": model, "chunk_size": chunk_size, "overlap": "n/a (LLM decides)"},
    )
