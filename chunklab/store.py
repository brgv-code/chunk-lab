"""Vector store interface plus a Chroma implementation.

Stage 2 indexes each strategy's chunks and retrieves top-k per query. `eval.py`
depends only on the `VectorStore` interface here, so Chroma can be swapped for
Qdrant (or anything else) without touching the scoring logic. That swappability
is the point of this file.

The Chroma impl runs fully local and in-process, no server. It uses an
in-memory (ephemeral) client so each eval run starts clean with no leftover
collections on disk. Switching to on-disk persistence is a one-line change:
replace `EphemeralClient()` with `PersistentClient(path=...)`.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

# cosine distance, so retrieval ranks by direction not magnitude
_SPACE = {"hnsw:space": "cosine"}


@dataclass
class Retrieved:
    rank: int          # 1-based rank in the result list
    document: str      # the text returned for this hit (parent text for hierarchical)
    metadata: dict
    distance: float    # cosine distance; lower is closer


class VectorStore(ABC):
    """Minimal interface the eval harness needs. Implement these three and a
    new backend drops in without any change to `eval.py`."""

    @abstractmethod
    def add(self, ids, embeddings, documents, metadatas) -> None: ...

    @abstractmethod
    def query(self, embedding, k: int) -> list[Retrieved]: ...

    @abstractmethod
    def count(self) -> int: ...


def sanitize_name(name: str) -> str:
    """Chroma requires 3-512 chars from [a-zA-Z0-9._-], starting and ending
    alphanumeric. Coerce any strategy key into a valid collection name."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("._-")
    if len(cleaned) < 3:
        cleaned = f"col-{cleaned}"
    return cleaned[:512]


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        import chromadb
        _CLIENT = chromadb.EphemeralClient()
    return _CLIENT


class ChromaStore(VectorStore):
    """A fresh Chroma collection. Any prior collection of the same name is
    dropped on construction so a re-run never sees stale vectors."""

    def __init__(self, name: str):
        client = _client()
        name = sanitize_name(name)
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - "does not exist" is fine
            pass
        self._col = client.create_collection(name, metadata=_SPACE)

    def add(self, ids, embeddings, documents, metadatas) -> None:
        ids = list(ids)
        if not ids:
            return
        self._col.add(ids=ids, embeddings=list(embeddings),
                      documents=list(documents), metadatas=list(metadatas))

    def query(self, embedding, k: int) -> list[Retrieved]:
        n = min(int(k), max(1, self._col.count()))
        res = self._col.query(query_embeddings=[list(embedding)], n_results=n,
                              include=["documents", "metadatas", "distances"])
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            Retrieved(rank=i + 1, document=docs[i], metadata=metas[i] or {},
                      distance=float(dists[i]))
            for i in range(len(docs))
        ]

    def count(self) -> int:
        return self._col.count()


def fresh_store(name: str) -> VectorStore:
    """Factory: the one place eval.py names a backend. Point this at a Qdrant
    implementation to switch stores."""
    return ChromaStore(name)
