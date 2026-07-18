"""Document-based chunking: use the structure that's already in the file.

Markdown has headings; source code has functions and classes. This strategy
reads those signals and splits along them: markdown via recursive rules that
break on headings first, code via Chonkie's tree-sitter CodeChunker. The
visualizer exposes a "treat input as" selector (prose / markdown / code) that
feeds ``treat_as``; when it's "auto" we detect the kind ourselves.
"""

from functools import lru_cache

from chonkie import RecursiveChunker, CodeChunker

from ..registry import register
from ..types import ChunkResult
from ._common import to_chunks, detect_kind, markdown_rules

TOKENIZER = "gpt2"


@lru_cache(maxsize=16)
def _markdown_chunker(chunk_size: int) -> RecursiveChunker:
    return RecursiveChunker(tokenizer=TOKENIZER, chunk_size=chunk_size,
                            rules=markdown_rules())


@lru_cache(maxsize=16)
def _prose_chunker(chunk_size: int) -> RecursiveChunker:
    return RecursiveChunker(tokenizer=TOKENIZER, chunk_size=chunk_size)


@lru_cache(maxsize=16)
def _code_chunker(chunk_size: int) -> CodeChunker:
    return CodeChunker(tokenizer=TOKENIZER, chunk_size=chunk_size, language="auto")


@register(
    "document", "Document-based",
    "Structure-aware: split markdown on its headings and code on its function "
    "and class boundaries, so a chunk is a section or a function, not an "
    "arbitrary window.",
)
def document(text, *, chunk_size, overlap, treat_as="auto", **_):
    kind = treat_as if treat_as in ("markdown", "code", "prose") else detect_kind(text)
    cs = int(chunk_size)

    if kind == "code":
        chunks = _code_chunker(cs)(text)
    elif kind == "markdown":
        chunks = _markdown_chunker(cs)(text)
    else:
        chunks = _prose_chunker(cs)(text)

    return ChunkResult(
        strategy="Document-based",
        chunks=to_chunks(chunks, base_meta={"kind": kind}),
        params={"treat_as": kind, "chunk_size": chunk_size,
                "overlap": "n/a (structure-aware)"},
    )
