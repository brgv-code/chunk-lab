"""Recursive chunking: the sensible prose default.

Chonkie's RecursiveChunker splits on the largest natural boundary first
(paragraphs), falling back to sentences then words only when a piece is still
too big. RecursiveChunker has no built-in overlap, so when overlap is asked for
we add it afterwards with OverlapRefinery (prefix mode): each chunk gets the
tail of the previous chunk prepended. We record how many characters were
prepended so the visualizer can highlight the shared span.
"""

from functools import lru_cache

from chonkie import RecursiveChunker
from chonkie.refinery import OverlapRefinery

from ..registry import register
from ..types import ChunkResult
from ._common import to_chunks

TOKENIZER = "gpt2"


@lru_cache(maxsize=32)
def _chunker(chunk_size: int) -> RecursiveChunker:
    return RecursiveChunker(tokenizer=TOKENIZER, chunk_size=chunk_size)


@register(
    "recursive", "Recursive",
    "Split on paragraph, then sentence, then word boundaries. Keeps related "
    "sentences together far more often than fixed-size, at almost no extra cost.",
)
def recursive(text, *, chunk_size, overlap, **_):
    chunks = _chunker(int(chunk_size))(text)
    overlaps = None
    overlap = int(overlap)
    if overlap > 0 and len(chunks) > 1:
        orig_lens = [len(c.text) for c in chunks]
        refinery = OverlapRefinery(tokenizer=TOKENIZER, context_size=overlap,
                                   method="prefix", merge=True, inplace=False)
        chunks = refinery(chunks)
        overlaps = [0] + [max(0, len(chunks[i].text) - orig_lens[i])
                          for i in range(1, len(chunks))]
    return ChunkResult(
        strategy="Recursive",
        chunks=to_chunks(chunks, overlaps=overlaps),
        params={"chunk_size": chunk_size, "overlap": overlap, "tokenizer": TOKENIZER},
    )
