"""Embedding model wrapper, loaded once.

We use Chonkie's ``AutoEmbeddings`` around ``sentence-transformers/all-MiniLM-L6-v2``
(384-dim, CPU-fine). The single cached object is reused for two things:
the SemanticChunker (Stage 1) and query/chunk embedding in the eval harness
(Stage 2), so the model never reloads per request and both stages measure
similarity the same way.
"""

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SHORT_NAME = "all-MiniLM-L6-v2"

_EMBEDDER = None


def get_embedder():
    """Return the cached Chonkie embeddings object (a BaseEmbeddings), loading
    it on first use. Safe to pass straight into SemanticChunker."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from chonkie import AutoEmbeddings
        _EMBEDDER = AutoEmbeddings.get_embeddings(MODEL_NAME)
    return _EMBEDDER


def _to_list(vec):
    tolist = getattr(vec, "tolist", None)
    return tolist() if tolist else list(vec)


def embed_one(text: str) -> list[float]:
    return _to_list(get_embedder().embed(text))


def embed_many(texts) -> list[list[float]]:
    return [_to_list(v) for v in get_embedder().embed_batch(list(texts))]


def dimension() -> int:
    return int(get_embedder().dimension)
