"""Registry + chunker behavior tests.

These run offline (no Ollama). The LLM strategy is expected to report itself
disabled, which is itself a tested behavior.
"""

from pathlib import Path

import pytest

from chunklab.registry import available_strategies, run_strategy
from chunklab.types import ChunkResult

DOC = (Path(__file__).parent.parent / "data" / "sample_doc.md").read_text()

NON_OLLAMA = ["fixed", "recursive", "document", "semantic", "agentic", "hierarchical"]


def test_seven_strategies_registered():
    keys = [s["key"] for s in available_strategies()]
    assert keys == ["fixed", "recursive", "document", "semantic",
                    "llm", "agentic", "hierarchical"]


def test_llm_gated_on_ollama():
    spec = {s["key"]: s for s in available_strategies()}["llm"]
    assert spec["needs_ollama"] is True
    # With Ollama off in CI, it is unavailable and carries a note.
    if not spec["available"]:
        assert spec["note"]


@pytest.mark.parametrize("key", NON_OLLAMA)
def test_each_strategy_runs(key):
    res = run_strategy(key, DOC, chunk_size=200, overlap=32)
    assert isinstance(res, ChunkResult)
    assert res.ok, res.note
    assert res.chunks, "expected at least one chunk"
    for c in res.chunks:
        assert c.text
        assert c.token_count >= 0
        assert c.end >= c.start


def test_unknown_strategy_is_soft_error():
    res = run_strategy("nope", DOC, chunk_size=200, overlap=32)
    assert res.ok is False
    assert "unknown" in res.note.lower()


def test_fixed_overlap_changes_output():
    none = run_strategy("fixed", DOC, chunk_size=200, overlap=0)
    some = run_strategy("fixed", DOC, chunk_size=200, overlap=64)
    shared_none = sum(c.meta.get("overlap_chars", 0) for c in none.chunks)
    shared_some = sum(c.meta.get("overlap_chars", 0) for c in some.chunks)
    assert shared_none == 0
    assert shared_some > 0


def test_recursive_overlap_changes_output():
    none = run_strategy("recursive", DOC, chunk_size=200, overlap=0)
    some = run_strategy("recursive", DOC, chunk_size=200, overlap=64)
    assert sum(c.meta.get("overlap_chars", 0) for c in none.chunks) == 0
    assert sum(c.meta.get("overlap_chars", 0) for c in some.chunks) > 0


def test_hierarchical_has_two_levels():
    res = run_strategy("hierarchical", DOC, chunk_size=200, overlap=0)
    levels = {c.level for c in res.chunks}
    assert levels == {0, 1}
    # every child points at a parent's text for context-return
    children = [c for c in res.chunks if c.level == 1]
    assert children and all(c.meta.get("parent_text") for c in children)


def test_agentic_tags_its_choice():
    res = run_strategy("agentic", DOC, chunk_size=200, overlap=32)
    assert res.ok
    assert res.params.get("chosen") in NON_OLLAMA
    assert all(c.meta.get("chosen") for c in res.chunks)
