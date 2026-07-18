"""Fixed-size chunking: the dumb baseline.

Chonkie's TokenChunker cuts every ``chunk_size`` tokens with ``chunk_overlap``
tokens of genuine overlap between neighbours. Fast, predictable, blind to
meaning. Every other strategy is measured against this.
"""

from functools import lru_cache

from chonkie import TokenChunker

from ..registry import register
from ..types import ChunkResult
from ._common import to_chunks

TOKENIZER = "gpt2"


@lru_cache(maxsize=32)
def _chunker(chunk_size: int, overlap: int) -> TokenChunker:
    return TokenChunker(tokenizer=TOKENIZER, chunk_size=chunk_size,
                        chunk_overlap=overlap)


@register(
    "fixed", "Fixed-size",
    "Cut every N tokens, with optional overlap. Fast and predictable, but "
    "blind to meaning, it will happily split a sentence in half.",
)
def fixed(text, *, chunk_size, overlap, **_):
    overlap = min(int(overlap), max(0, int(chunk_size) - 1))
    chunks = _chunker(int(chunk_size), overlap)(text)
    return ChunkResult(
        strategy="Fixed-size",
        chunks=to_chunks(chunks),
        params={"chunk_size": chunk_size, "overlap": overlap, "tokenizer": TOKENIZER},
    )
