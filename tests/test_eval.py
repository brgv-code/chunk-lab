"""Stage 2 eval-harness tests: the hit definition, the store, and scoring.

These run offline but do load the local embedding model (as the registry tests
already do).
"""

from pathlib import Path

import pytest

from chunklab.eval import (
    normalize, is_hit, find_span, load_eval_spec,
    evaluate_strategy, run_full_eval,
)
from chunklab.store import fresh_store, sanitize_name

ROOT = Path(__file__).parent.parent
DOC, QUERIES, _ = load_eval_spec(str(ROOT / "data" / "sample_eval.json"), ROOT)


# --- hit definition ---

def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Embedding\n Similarity ") == "embedding similarity"


def test_is_hit_is_whitespace_normalized_substring():
    assert is_hit("cuts where the embedding\nsimilarity drops", "embedding similarity")
    assert not is_hit("nothing relevant here", "embedding similarity")


def test_find_span_matches_across_whitespace():
    text = "cuts where the embedding\nsimilarity drops"
    span = find_span(text, "embedding similarity")
    assert span is not None
    s, e = span
    assert text[s:e].lower().split() == ["embedding", "similarity"]


def test_sample_gold_answers_are_all_present_in_doc():
    # the eval set only makes sense if every gold string is really in the doc
    for q in QUERIES:
        assert is_hit(DOC, q["answer"]), q["answer"]


# --- store ---

def test_store_add_and_query_ranks_by_closeness():
    store = fresh_store("eval-unit-test")
    store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]],
              ["alpha", "beta"], [{"i": 0}, {"i": 1}])
    hits = store.query([0.9, 0.1], k=2)
    assert [h.document for h in hits] == ["alpha", "beta"]
    assert hits[0].rank == 1 and hits[0].distance <= hits[1].distance


def test_sanitize_name_makes_valid_collection_names():
    assert sanitize_name("x") == "col-x"           # too short gets padded
    assert sanitize_name("eval/fixed!") == "eval-fixed"


# --- scoring ---

@pytest.mark.parametrize("key", ["fixed", "recursive", "document", "hierarchical"])
def test_evaluate_strategy_produces_valid_metrics(key):
    e = evaluate_strategy(key, DOC, QUERIES, chunk_size=256, overlap=32, k=5)
    assert e.ok, e.note
    assert e.n_units > 0
    assert 0.0 <= e.recall_at_k <= 1.0
    assert 0.0 <= e.mrr <= e.recall_at_k + 1e-9   # MRR can't exceed recall@k
    assert len(e.per_query) == len(QUERIES)


def test_recall_is_monotonic_in_k():
    lo = evaluate_strategy("fixed", DOC, QUERIES, chunk_size=256, overlap=32, k=1)
    hi = evaluate_strategy("fixed", DOC, QUERIES, chunk_size=256, overlap=32, k=5)
    assert hi.recall_at_k >= lo.recall_at_k


def test_run_full_eval_is_ranked_by_recall():
    keys = ["fixed", "recursive", "document", "semantic", "hierarchical"]
    evals = run_full_eval(DOC, QUERIES, keys, chunk_size=256, overlap=32, k=5)
    oks = [e for e in evals if e.ok]
    recalls = [e.recall_at_k for e in oks]
    assert recalls == sorted(recalls, reverse=True)


def test_hierarchical_returns_parent_text_for_retrieval():
    e = evaluate_strategy("hierarchical", DOC, QUERIES, chunk_size=256, overlap=32, k=5)
    assert e.ok
    # at least one retrieved unit should carry substantial (parent-sized) text
    longest = max(len(r.text) for qe in e.per_query for r in qe.retrieved)
    assert longest > 256
