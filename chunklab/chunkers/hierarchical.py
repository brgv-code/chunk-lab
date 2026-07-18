"""Hierarchical chunking: search small, return large.

Built on recursive: split the document into large parent chunks (level 0), then
split each parent into small child chunks (level 1). The visualizer nests
children under their parent. The eval harness retrieves on children (precise
match) but can return the parent's text (enough context to use it), a direct
answer to the large-versus-tiny tension. Overlap is not applicable; the two
layers *are* the context mechanism.

The returned list interleaves each parent with its own children:
    [parent0, child0.0, child0.1, ..., parent1, child1.0, ...]
"""

from functools import lru_cache

from chonkie import RecursiveChunker

from ..registry import register
from ..types import Chunk, ChunkResult
from ._common import count_tokens

TOKENIZER = "gpt2"


@lru_cache(maxsize=32)
def _chunker(chunk_size: int) -> RecursiveChunker:
    return RecursiveChunker(tokenizer=TOKENIZER, chunk_size=chunk_size)


@register(
    "hierarchical", "Hierarchical",
    "Two layers: large parents split into small children. Retrieve on the "
    "precise children, return the parent's text for context. Search small, "
    "return large.",
)
def hierarchical(text, *, chunk_size, overlap, **_):
    cs = int(chunk_size)
    parent_size = max(cs * 4, 256)
    child_size = max(cs, 64)

    parents = _chunker(parent_size)(text)
    child_splitter = _chunker(child_size)

    out: list[Chunk] = []
    for pi, p in enumerate(parents):
        p_start = int(getattr(p, "start_index", 0) or 0)
        out.append(Chunk(
            text=p.text, start=p_start,
            end=int(getattr(p, "end_index", p_start + len(p.text))),
            token_count=count_tokens(p.text), level=0,
            meta={"role": "parent", "parent_index": pi, "overlap_chars": 0},
        ))
        for c in child_splitter(p.text):
            c_start = p_start + int(getattr(c, "start_index", 0) or 0)
            c_end = p_start + int(getattr(c, "end_index", len(c.text)))
            out.append(Chunk(
                text=c.text, start=c_start, end=c_end,
                token_count=count_tokens(c.text), level=1,
                meta={"role": "child", "parent_index": pi,
                      "parent_text": p.text, "overlap_chars": 0},
            ))

    return ChunkResult(
        strategy="Hierarchical",
        chunks=out,
        params={"parent_size": parent_size, "child_size": child_size,
                "overlap": "n/a (two layers)"},
    )
