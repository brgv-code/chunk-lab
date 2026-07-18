# Chunking for retrieval-augmented generation

Retrieval-augmented generation, or RAG, is the practice of answering a question by first
retrieving relevant passages from your own documents and then handing those passages to a
language model as context. The model never sees your whole corpus. It sees a handful of
passages that some retriever decided were relevant. Everything downstream depends on that
handful being the right one.

Which means everything depends on how you cut the documents up in the first place. That
cutting step is called chunking, and it is the least glamorous and most consequential part
of the whole pipeline. This document explains what chunking is, why it decides retrieval
quality, and how the common strategies differ.

## Why does chunking exist at all?

Embedding models turn a span of text into a single fixed-length vector. That vector is a
lossy summary of the whole span. A short, focused span produces a vector that points
cleanly at one idea. A long span that wanders across three topics produces a vector that
points at the average of those topics, which is to say it points nowhere useful.

So you cannot embed a whole document as one vector and expect precise retrieval. You have
to split the document into pieces small enough that each piece is about one thing. Each
piece is a chunk. The collection of chunks becomes the collection of vectors you search
over.

The core tension is simple to state and hard to resolve. Large chunks average unrelated
ideas into one blurry vector. Tiny chunks lose the surrounding context that made the
sentence meaningful. And even when a large chunk does get retrieved, models tend to ignore
material buried in the middle of a long context, a failure mode known as lost in the
middle. There is no setting that escapes all three problems at once. There is only the
setting that is least wrong for your data.

## What are the common chunking strategies?

There is no single correct way to split a document. Over time a handful of strategies have
become standard, each making a different bet about where the meaningful boundaries in a
document actually are.

### Fixed-size chunking

The simplest approach. Walk through the text and cut every N tokens, optionally letting
adjacent chunks share a few tokens of overlap so a sentence split across a boundary is not
lost entirely. Fixed-size chunking is fast, predictable, and completely blind to meaning.
It will happily cut a sentence in half or merge the end of one section with the start of
the next. It is the baseline every other strategy is measured against.

### Recursive chunking

Recursive chunking tries to respect structure. It splits on the largest natural boundary
first, paragraphs, then falls back to sentences, then to words, only breaking at a finer
level when a piece is still too large. The result keeps related sentences together far more
often than fixed-size chunking does, at almost no extra cost. For most prose it is the
sensible default.

### Document-based chunking

Document-based chunking uses the structure that is already in the file. Markdown has
headings and code fences. Source code has functions and classes. A document-aware chunker
reads those signals and splits along them, so a chunk corresponds to a section or a
function rather than an arbitrary token window. For source code specifically, the
CodeChunker respects function and class boundaries instead of cutting through the middle of
a definition.

### Semantic chunking

Semantic chunking asks the embeddings themselves where to cut. It embeds each sentence,
then walks through the document watching how similar each sentence is to the running group.
Where the embedding similarity drops sharply, it declares a topic boundary and starts a new
chunk. The split points follow meaning rather than punctuation. This is slower because it
requires embedding the text before you can even chunk it, but the boundaries are often much
cleaner.

### LLM-based chunking

The most expensive strategy hands the document to a language model and asks it to mark the
boundaries directly. Given enough capability, the model can split along argument structure
in a way no rule captures. The cost is latency and, usually, money, though running a local
model through Ollama removes the money part. It is the highest-ceiling and slowest option.

### Agentic chunking

Agentic chunking is a router rather than a single method. It inspects the document, notices
whether it is markdown, code, or plain prose, and dispatches to whichever of the above
strategies fits that document best. The point is that no one strategy wins on every input,
so the router picks per document instead of committing globally.

### Hierarchical chunking

Hierarchical chunking keeps two layers at once. It splits the document into large parent
chunks, then splits each parent into small child chunks. Retrieval runs on children so the
match is precise, but the system can return the parent's text so the model gets enough
surrounding context to actually use the match. It is a direct answer to the large-versus-
tiny tension: search small, return large.

## How do you measure which strategy is best?

Looking at chunks tells you what they are. It does not tell you which one retrieves the
right passage for a real question. For that you need an evaluation harness, and the harness
needs a definition of success that is transparent enough to trust.

The simplest honest definition is a substring match. You write a set of questions, and for
each one you write down a short answer string that you know appears in the document. You
chunk the document, embed every chunk, and index them. Then for each question you embed the
query and retrieve the top-k chunks. A retrieved chunk counts as a hit if it contains the
answer string, compared case-insensitively after normalizing whitespace. No manual labeling
of which chunk is correct. The gold string does the labeling for you.

From hits you compute a couple of standard numbers.

| Metric | Question it answers | Range |
| --- | --- | --- |
| Recall@k | Did a correct chunk make the top k at all? | 0 to 1 |
| MRR | How high did the first correct chunk rank? | 0 to 1 |
| Hit rate | Fraction of queries with any hit in top k | 0 to 1 |

Recall@k is the fraction of queries where a hit appears anywhere in the top-k results. MRR,
the mean reciprocal rank, rewards ranking the correct chunk first rather than fifth: it
averages one over the rank of the first hit, scoring zero when nothing in the top-k
matches. Read together, recall tells you whether the answer was retrievable at all and MRR
tells you how hard the model would have to dig to find it.

## What does this look like in code?

The whole point of a chunking library is that swapping strategies is a one-line change.
Here is fixed-size token chunking:

```python
from chonkie import TokenChunker

chunker = TokenChunker(chunk_size=256, chunk_overlap=32)
chunks = chunker(document_text)
for chunk in chunks:
    print(chunk.token_count, chunk.text[:40])
```

And here is the retrieval side, embedding chunks with a local model and scoring a query:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vectors = model.encode([c.text for c in chunks])
query_vector = model.encode("how is a hit defined?")
# rank chunks by cosine similarity to query_vector, take the top k
```

Chunk Lab, the tool this document ships with, uses the all-MiniLM-L6-v2 embedding model for
both semantic chunking and retrieval, and stores vectors in a local Chroma collection so the
whole thing runs offline with no API keys.

## So which strategy should you use?

The honest answer is that it depends on your data, and the only way to know is to measure on
your data. Prose with clean paragraph structure rewards recursive chunking. Technical
documents full of code reward document-based chunking. Dense argumentative text, where
topic shifts do not line up with paragraph breaks, is where semantic chunking earns its
extra cost.

What you should not do is what most people do, which is pick a chunk size of 512 tokens
because a tutorial used it, wire up a vector database, get mediocre retrieval, and then
spend a week swapping embedding models and databases. The embedding model is rarely the
problem. The chunking is upstream of all of it, and it is invisible unless you go looking.

That is the whole reason to make chunks visible: so the most consequential decision in your
pipeline stops being the one you never actually looked at.
