"""Agentic chunking: a router, not a chunker.

No single strategy wins on every input, so this one inspects the document
(markdown headings? code? dense prose?) and dispatches to whichever of the
other strategies fits best, tagging what it picked so the UI can show the
decision. The point is picking per document instead of committing globally.
"""

from ..registry import register, run_strategy
from ._common import detect_kind


def _route(text: str) -> tuple[str, str]:
    """Return (chosen_strategy_key, human_reason)."""
    kind = detect_kind(text)
    if kind == "code":
        return "document", "looks like source code -> document-based (code)"
    if kind == "markdown":
        return "document", "markdown headings present -> document-based (markdown)"
    # Plain prose: dense, long-paragraph prose rewards semantic splits;
    # ordinary prose is fine with recursive.
    paras = [p for p in text.split("\n\n") if p.strip()]
    avg_len = sum(len(p) for p in paras) / max(1, len(paras))
    if avg_len > 700:
        return "semantic", "long, dense paragraphs -> semantic"
    return "recursive", "ordinary prose -> recursive"


@register(
    "agentic", "Agentic (router)",
    "Inspect the document and dispatch to the best-fit strategy, code goes to "
    "document-based, dense prose to semantic, ordinary prose to recursive.",
)
def agentic(text, *, chunk_size, overlap, **_):
    chosen, reason = _route(text)
    res = run_strategy(chosen, text, chunk_size=chunk_size, overlap=overlap,
                       treat_as=detect_kind(text))
    for c in res.chunks:
        c.meta["chosen"] = chosen
    res.strategy = "Agentic (router)"
    res.params = {**res.params, "chosen": chosen, "reason": reason}
    return res
