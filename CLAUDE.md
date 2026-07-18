# CLAUDE.md: Chunk Lab

Standing instructions for this repo. Growing knowledge (decisions, errors, sessions, blog) lives in `knowledge-base/projects/chunk-lab/`, not here.

## What this is
A local tool to see and evaluate RAG chunking strategies. Two Streamlit tabs over one shared chunker registry: Stage 1 visualizes chunks, Stage 2 scores retrieval. Build Stage 1 → gate → Stage 2.

## Commands
- `uv run streamlit run app.py`: launch the app (http://localhost:8501)
- `uv run pytest`: registry + eval tests
- `uv add <pkg>`: deps (never pip install into the venv directly)

## Rules
- Local-only, zero paid services, no API keys. Must run offline after the one-time model download.
- A strategy is a function `(text, *, chunk_size, overlap) -> ChunkResult` registered with `@register(...)` in `chunklab/chunkers/`. Add one function; it appears in both stages.
- The embedding model loads once in `chunklab/embed.py` and is reused for semantic chunking and Stage 2. Never reload per request.
- Verify Chonkie's API against the installed skill (`.agents/skills/chonkie`) and the venv version, not from memory.
- Small, boring, readable code. This is a tool for looking at things, not a framework.

## Guardrails (see knowledge-base error_logs)
- Chonkie `Chunk` uses `start_index` / `end_index` / `token_count`. Map to our `Chunk` only in `chunkers/_common.to_chunks`.
- `OverlapRefinery` leaves source indices unchanged; compute recursive overlap from the text-length delta, not index intersection.
- Do not use `recipe="markdown"` (HF download breaks offline); build `RecursiveRules` inline.
- Report `token_count` via one canonical gpt2 tokenizer so metrics compare like-for-like.
- LLM chunking is `needs_ollama=True`; it disables cleanly when Ollama is off.

## Out of scope (future work, noted in README)
Reranking, PDF ingestion, hybrid/keyword search, multiple embedding models at once, cloud vector DBs, auth.
