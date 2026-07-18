# Chunk Lab

*See and evaluate RAG chunking strategies on your own data. Local, offline, no API keys.*

**Live demo:** [chunk-lab.streamlit.app](https://chunk-lab.streamlit.app)

Retrieval-augmented generation fails more often at the chunking step than at the retriever, and you rarely notice, because chunks are invisible. They live inside the vector store as numbers. Chunk Lab pulls them back into the open. Paste a document, run several chunking strategies, and look at the actual chunks side by side. Then measure which strategy retrieves the right passage for your questions.

It runs entirely on your machine. The embeddings use a small local model, the vector store is on-disk, and the one strategy that needs a language model talks to a local Ollama install or disables itself. Nothing leaves the laptop and nothing needs a key.

## Why build this?

Because the usual debugging story is backwards. Retrieval gets vague or wrong, so people swap the embedding model, then the vector database, then the number of results, and lose a week. The chunking is upstream of all of it, and it is the one part nobody looks at, because its output was never visible.

So the goal is narrow. Answer two questions, clearly:

1. What do my chunks actually look like?
2. Which strategy actually retrieves the right chunk for my data?

Stage 1 answers the first. Stage 2 answers the second.

## What works today?

Both stages are built.

**Stage 1, the visualizer.** Paste or upload a `.txt` or `.md` file (or load the bundled sample), run up to seven chunking strategies at once, and see every chunk as an alternating colored block with the overlapping span highlighted and hierarchical children nested under their parents. A per-strategy metrics table shows chunk count and the mean, median, min, max, and standard deviation of token counts.

**Stage 2, the retrieval eval harness.** Score each strategy on a query set with known answers. A retrieved chunk is a hit if it contains the query's gold answer string (case-insensitive). You get a ranked table of recall@k and MRR, a bar chart, and a per-query drill-down that shows the top-k retrieved chunks with the matched answer highlighted, so a low score is explainable rather than just a number.

## How do you run it?

You need [uv](https://github.com/astral-sh/uv). Then:

```bash
uv sync
uv run streamlit run app.py
```

That opens the app in your browser. Click "Load sample doc", pick a few strategies, and change the chunk size and overlap to watch the columns respond.

Run the tests with:

```bash
uv run pytest
```

## What are the seven strategies?

Each one makes a different bet about where the meaningful boundaries in a document are.

| Strategy | What it does |
| --- | --- |
| Fixed-size | Cut every N tokens, with optional overlap. Fast, predictable, blind to meaning. The baseline. |
| Recursive | Split on paragraphs, then sentences, then words. The sensible prose default. |
| Document-based | Use the structure already in the file: markdown headings, or code function and class boundaries. |
| Semantic | Embed each sentence and cut where the meaning shifts. Boundaries follow topics, not punctuation. |
| LLM-based | Hand the document to a local model (via Ollama) and let it mark the boundaries. Optional. |
| Agentic | A router. Inspect the document and dispatch to whichever strategy fits it best. |
| Hierarchical | Two layers: large parents split into small children. Retrieve on the children, return the parent. |

## How does the optional LLM strategy work?

It is off by default and never required. If you have [Ollama](https://ollama.com) running with a model pulled (for example `ollama pull llama3.2`), the LLM-based strategy lights up and runs against your local model through Ollama's OpenAI-compatible endpoint. If Ollama is not running, the strategy shows as disabled and everything else works unchanged.

## How is the code organized?

```
app.py                  Streamlit entrypoint, two tabs
chunklab/
  types.py              Chunk and ChunkResult dataclasses
  registry.py           strategy registry and dispatch
  ollama.py             local Ollama detection
  embed.py              embedding model, loaded once
  viz.py                chunk-rendering helpers for the UI
  chunkers/
    _common.py          token counting, Chonkie mapping, structure detection
    fixed.py recursive.py document.py semantic.py
    llm.py agentic.py hierarchical.py
data/
  sample_doc.md         a self-referential explainer about chunking
  sample_eval.json      queries with short verbatim answers
tests/
```

A strategy is one function that takes text and returns chunks, registered with a decorator. Add one function and it shows up in both stages automatically. That is the whole extension model.

## What is out of scope for now?

Reranking, PDF ingestion, hybrid or keyword search, multiple embedding models at once, cloud vector databases, and auth. The surface is kept small on purpose, so the two questions stay easy to answer. These are noted here as possible future work, not as things that exist.

## Where is the write-up?

I am building this in public and writing a three-part series about it: why chunking is the real problem, what the visualizer shows, and which strategy actually wins when you measure. The posts go up on [bhargav.dev](https://bhargav.dev).

## Stack

Python 3.11, [uv](https://github.com/astral-sh/uv) for the environment, [Chonkie](https://github.com/chonkie-inc/chonkie) for chunking, sentence-transformers (`all-MiniLM-L6-v2`) for embeddings, Streamlit for the UI, and Chroma for the vector store behind a thin interface so it can be swapped later.
