"""Shared data model for chunks and chunking results.

These are the two types every strategy speaks. A chunker maps Chonkie's own
``Chunk`` (which exposes ``start_index`` / ``end_index`` / ``token_count``) onto
this ``Chunk``, so the rest of the app never depends on Chonkie's field names.
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    start: int            # char offset into source (best effort)
    end: int
    token_count: int
    level: int = 0        # 0 = normal; hierarchical uses 0=parent, 1=child
    meta: dict = field(default_factory=dict)


@dataclass
class ChunkResult:
    strategy: str
    chunks: list[Chunk]
    params: dict          # chunk_size, overlap, etc. actually used
    ok: bool = True
    note: str = ""        # e.g. "disabled: Ollama not running"

    @property
    def token_counts(self) -> list[int]:
        return [c.token_count for c in self.chunks]
