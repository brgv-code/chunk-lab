"""Stage 2 scoring: chunk, embed, index, retrieve, score.

For each strategy: chunk the document, embed every chunk with the same MiniLM
model Stage 1 uses, index into a fresh vector-store collection, then for each
query embed it, retrieve top-k, and score.

A retrieved chunk is a hit if it contains the query's short gold answer string,
compared case-insensitively after whitespace normalization. That definition is
deliberately crude and transparent: it needs no labeling of which chunk is
"correct", the gold string does the labeling.

Metrics per strategy:
- recall@k: fraction of queries with a hit anywhere in the top-k
- MRR: mean of 1/rank of the first hit (0 when no hit in top-k)
- hit rate: same as recall@k here; both kept for clarity when k varies
- mean chunk count and mean token size, so quality reads against granularity
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .registry import run_strategy, strategy_label
from .embed import embed_one, embed_many
from .store import fresh_store


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def is_hit(text: str, answer: str) -> bool:
    """The hit definition: whitespace-normalized, case-insensitive substring."""
    if not answer:
        return False
    return normalize(answer) in normalize(text)


def find_span(text: str, answer: str):
    """Best-effort (start, end) char span of the first match of `answer` in the
    raw `text`, for highlighting. Answer whitespace matches any run of
    whitespace. Returns None if not found on the raw text (the hit may still
    count via `is_hit`, which normalizes first)."""
    if not answer:
        return None
    pattern = r"\s+".join(re.escape(w) for w in answer.split())
    m = re.search(pattern, text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


@dataclass
class Retrieval:
    rank: int
    text: str
    distance: float
    hit: bool
    span: tuple | None


@dataclass
class QueryEval:
    query: str
    answer: str
    retrieved: list
    first_hit_rank: int | None
    recall: int          # 1 if any hit in top-k else 0
    rr: float            # reciprocal rank of first hit, else 0


@dataclass
class StrategyEval:
    key: str
    strategy: str
    ok: bool = True
    note: str = ""
    n_units: int = 0
    mean_tokens: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    per_query: list = field(default_factory=list)


def _index_units(result):
    """(embed_text, return_text, token_count) per retrieval unit.

    Hierarchical retrieves on the small children but returns the parent's text,
    so we embed the child and store the parent text as the returned document.
    Every other strategy embeds and returns the same chunk text.
    """
    chunks = result.chunks
    if any(c.level == 1 for c in chunks):
        children = [c for c in chunks if c.level == 1]
        return [(c.text, c.meta.get("parent_text", c.text), c.token_count) for c in children]
    return [(c.text, c.text, c.token_count) for c in chunks]


def evaluate_strategy(key, doc_text, queries, *, chunk_size, overlap, k) -> StrategyEval:
    res = run_strategy(key, doc_text, chunk_size=chunk_size, overlap=overlap)
    label = res.strategy or strategy_label(key)
    if not res.ok:
        return StrategyEval(key=key, strategy=label, ok=False, note=res.note)

    units = _index_units(res)
    if not units:
        return StrategyEval(key=key, strategy=label, ok=False, note="no chunks produced")

    embed_texts = [u[0] for u in units]
    return_texts = [u[1] for u in units]
    tokens = [u[2] for u in units]
    metas = [{"tokens": int(t)} for t in tokens]

    store = fresh_store(f"eval-{key}")
    store.add([f"u{i}" for i in range(len(units))], embed_many(embed_texts),
              return_texts, metas)

    per_query, recalls, rrs = [], [], []
    for q in queries:
        query, answer = q["query"], q["answer"]
        retrieved = store.query(embed_one(query), k)
        rlist, first_hit = [], None
        for r in retrieved:
            h = is_hit(r.document, answer)
            if h and first_hit is None:
                first_hit = r.rank
            rlist.append(Retrieval(rank=r.rank, text=r.document, distance=r.distance,
                                   hit=h, span=find_span(r.document, answer) if h else None))
        recall = 1 if first_hit is not None else 0
        rr = (1.0 / first_hit) if first_hit else 0.0
        recalls.append(recall)
        rrs.append(rr)
        per_query.append(QueryEval(query=query, answer=answer, retrieved=rlist,
                                   first_hit_rank=first_hit, recall=recall, rr=rr))

    n = max(1, len(queries))
    return StrategyEval(
        key=key, strategy=label, ok=True,
        n_units=len(units),
        mean_tokens=round(sum(tokens) / len(tokens), 1),
        recall_at_k=round(sum(recalls) / n, 3),
        mrr=round(sum(rrs) / n, 3),
        hit_rate=round(sum(recalls) / n, 3),
        per_query=per_query,
    )


def run_full_eval(doc_text, queries, keys, *, chunk_size, overlap, k) -> list[StrategyEval]:
    evals = [evaluate_strategy(key, doc_text, queries, chunk_size=chunk_size,
                               overlap=overlap, k=k) for key in keys]
    # ranked: runnable first, then by recall@k, then MRR as the tie-breaker
    evals.sort(key=lambda e: (e.ok, e.recall_at_k, e.mrr), reverse=True)
    return evals


def load_eval_spec(spec_or_path, base_dir):
    """Return (doc_text, queries, doc_path). `spec_or_path` is a parsed dict or a
    path to a JSON eval file; `document` inside it is resolved relative to
    `base_dir`."""
    if isinstance(spec_or_path, (str, Path)):
        spec = json.loads(Path(spec_or_path).read_text())
    else:
        spec = spec_or_path
    doc_path = Path(base_dir) / spec["document"]
    return doc_path.read_text(), spec["queries"], str(doc_path)
