"""Semantic chunking: let the embeddings choose the boundaries.

Chonkie's SemanticChunker embeds sentences and cuts where the running
embedding similarity drops, so splits follow meaning rather than punctuation.
We hand it the *same* cached MiniLM embedder the eval harness uses, so the
model loads once and both stages measure similarity the same way. Overlap is
not applicable, the boundaries are chosen, not slid.
"""

from ..registry import register
from ..types import ChunkResult
from ._common import to_chunks

_CHUNKERS: dict[int, object] = {}


def _chunker(chunk_size: int):
    ch = _CHUNKERS.get(chunk_size)
    if ch is None:
        from chonkie import SemanticChunker
        from ..embed import get_embedder
        ch = SemanticChunker(embedding_model=get_embedder(), chunk_size=chunk_size)
        _CHUNKERS[chunk_size] = ch
    return ch


@register(
    "semantic", "Semantic",
    "Embed each sentence and cut where the embedding similarity drops sharply. "
    "Boundaries follow topic shifts, not punctuation. Slower, often cleaner.",
)
def semantic(text, *, chunk_size, overlap, **_):
    chunks = _chunker(int(chunk_size))(text)
    from ..embed import SHORT_NAME
    return ChunkResult(
        strategy="Semantic",
        chunks=to_chunks(chunks),
        params={"chunk_size": chunk_size, "overlap": "n/a (similarity splits)",
                "embedding_model": SHORT_NAME},
    )
