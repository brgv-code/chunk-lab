"""Helpers shared by every chunker.

Two jobs live here:

1. A single canonical token counter (tiktoken/gpt2) so *every* strategy reports
   ``token_count`` in the same unit. Chonkie's own count varies by chunker
   (semantic sizes in the embedding model's wordpiece tokens, others in gpt2),
   which would make the metrics panel compare apples to oranges. Recounting
   here keeps the comparison honest.

2. ``to_chunks``: maps Chonkie ``Chunk`` objects (``start_index`` /
   ``end_index`` / ``.text``) onto our ``Chunk`` and records how many leading
   characters overlap the previous chunk (``meta["overlap_chars"]``), so the
   visualizer can highlight the shared span.
"""

import re
import tiktoken

from ..types import Chunk

_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("gpt2")
    return _ENC


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc().encode(text))


def to_chunks(cc_chunks, *, level=0, base_meta=None, overlaps=None) -> list[Chunk]:
    """Map a list of Chonkie chunks to our Chunk list.

    ``overlaps`` (optional) is an explicit per-chunk leading-overlap length in
    characters, used when overlap was added by a refinery that leaves the
    source indices unchanged. When it's None we infer overlap from the source
    index intersection (how TokenChunker's built-in overlap manifests).
    """
    out: list[Chunk] = []
    prev_end = None
    for i, c in enumerate(cc_chunks):
        text = c.text
        start = getattr(c, "start_index", None)
        start = 0 if start is None else int(start)
        end = getattr(c, "end_index", None)
        end = start + len(text) if end is None else int(end)

        if overlaps is not None:
            ov = overlaps[i] if i < len(overlaps) else 0
        elif prev_end is not None and start < prev_end:
            ov = prev_end - start
        else:
            ov = 0

        meta = dict(base_meta or {})
        meta["overlap_chars"] = max(0, int(ov))
        out.append(Chunk(text=text, start=start, end=end,
                         token_count=count_tokens(text), level=level, meta=meta))
        prev_end = end
    return out


# --- structure detection, shared by document + agentic strategies ---

_CODE_LINE = re.compile(
    r"\s*(def |class |import |from |function |const |let |var |public |private "
    r"|#include|package |func |fn |impl |return |print\(|console\.)"
)
_HEADING = re.compile(r"(?m)^#{1,6}\s")


def _looks_like_code(text: str) -> bool:
    lines = text.splitlines()
    if not lines:
        return False
    code_sig = sum(bool(_CODE_LINE.match(ln)) for ln in lines)
    ratio = code_sig / max(1, len(lines))
    punct = text.count("{") + text.count("}") + text.count(";")
    return ratio > 0.12 or (punct > len(lines) and punct > 12)


def detect_kind(text: str) -> str:
    """Classify input as 'code', 'markdown', or 'prose' for the structure-aware
    strategies. A markdown doc that merely *contains* code fences is still
    markdown; only source-code-dominant input reads as 'code'."""
    if _HEADING.search(text) or "```" in text:
        return "markdown"
    if _looks_like_code(text):
        return "code"
    return "prose"


def markdown_rules():
    """Offline markdown-aware recursive rules: headings first, then paragraphs,
    then sentences, then words. Avoids the HuggingFace-hub recipe download so
    the tool works without network."""
    from chonkie.types import RecursiveRules, RecursiveLevel
    return RecursiveRules(levels=[
        RecursiveLevel(delimiters=["\n# ", "\n## ", "\n### ", "\n#### "]),
        RecursiveLevel(delimiters=["\n\n"]),
        RecursiveLevel(delimiters=[". ", "! ", "? "]),
        RecursiveLevel(whitespace=True),
    ])
