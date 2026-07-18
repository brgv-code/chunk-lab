# Progress

Live current-state for Chunk Lab. Session archive lives in `knowledge-base/projects/chunk-lab/sessions/`.

## Status: Stage 1 shipped (GATE 1 passed). Stage 2 not started.

### Done
- Shared foundation: `types`, `registry`, `ollama`, `embed`, `chunkers/_common`, and all seven chunkers.
- Sample data: `data/sample_doc.md`, `data/sample_eval.json` (gold strings verified present).
- Stage 1 visualizer: `viz.py` + `app.py` tab 1. Verified live in a browser; overlap responds; LLM disabled cleanly with Ollama off.
- Tests: `uv run pytest` → 13 passing (`tests/test_registry.py`).
- Blog Posts 1 and 2 drafted, validated, committed to the KB.

### Next (Stage 2, after GATE 1 hand-confirmation)
1. `chunklab/store.py`: vector store interface + Chroma persistent-local impl (ADR-005).
2. `chunklab/eval.py`: chunk → embed → index → retrieve top-k → score (recall@k, MRR, hit rate).
3. `app.py` tab 2: eval file picker, `k` control, ranked table + bar chart + per-query drill-down.
4. `tests/test_eval.py`: scoring logic (hit definition, recall@k, MRR).
5. GATE 2 verify → blog Post 3 with real numbers → finalize `README.md` (with out-of-scope / future work).
