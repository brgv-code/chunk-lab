# Progress

Live current-state for Chunk Lab. Session archive lives in `knowledge-base/projects/chunk-lab/sessions/`.

## Status: Stage 1 and Stage 2 shipped (GATE 1 and GATE 2 passed). Deployed live.

### Done
- Shared foundation: `types`, `registry`, `ollama`, `embed`, `chunkers/_common`, and all seven chunkers.
- Sample data: `data/sample_doc.md`, `data/sample_eval.json` (gold strings verified present).
- Stage 1 visualizer: `viz.py` + `app.py` tab 1. Verified live in a browser; overlap responds; LLM disabled cleanly with Ollama off.
- Stage 2 eval harness: `store.py` (VectorStore interface + Chroma impl), `eval.py` (chunk, embed, index, retrieve, score), and `app.py` tab 2 (ranked table, recall/MRR bar chart, per-query drill-down). Hit is a whitespace-normalized substring match; hierarchical retrieves on children and returns parent text.
- Tests: `uv run pytest` → 26 passing (`tests/test_registry.py`, `tests/test_eval.py`).
- Blog Posts 1, 2, 3 plus a server-vs-serverless post, committed to the KB.
- Deployed live to Streamlit Community Cloud from `main` (Cloud installs from `uv.lock`): https://chunk-lab.streamlit.app . Note: Cloud hot-reload does not re-import changed submodules, so a push that adds names to `chunklab/*` needs a one-time app reboot; Stage 2 symbols are imported lazily so a stale submodule can't crash Stage 1.

### Result on the sample data (chunk_size 256, k 5)
Hierarchical wins (recall 1.0, MRR 0.94). Recursive, fixed-size, and document-based all reach recall 1.0 with lower MRR. Semantic trails at 0.5 because at this size it over-splits into 48 tiny fragments. No universal winner, which is the point.

### Possible next work (out of scope for now)
Reranking, PDF ingestion, hybrid or keyword search, multiple embedding models at once, cloud vector databases, auth.
