"""Chunk-rendering helpers for the Streamlit visualizer.

Two things live here: HTML that draws each chunk as an alternating colored
block (with the overlap span highlighted and hierarchical children nested under
their parent), and the per-strategy metrics table. Kept deliberately plain.
This is a tool for looking at things.
"""

import html
import statistics as stats

from .types import ChunkResult

# Alternating block colors (light, readable in Streamlit's default theme).
_BLOCK_A = "#eef2ff"
_BLOCK_B = "#e0f2fe"
_BORDER = "#c7d2fe"
_OVERLAP_BG = "#fde68a"       # amber: the span shared with the previous chunk
_PARENT_BG = "#f1f5f9"
_TEXT = "#0f172a"


def _esc(s: str) -> str:
    return html.escape(s).replace("\n", "<br>")


def _chunk_body(text: str, overlap_chars: int) -> str:
    """Render chunk text, highlighting the leading overlap span if any."""
    overlap_chars = max(0, min(int(overlap_chars), len(text)))
    if overlap_chars:
        shared = _esc(text[:overlap_chars])
        rest = _esc(text[overlap_chars:])
        return (f'<span style="background:{_OVERLAP_BG};border-radius:3px;" '
                f'title="shared with previous chunk">{shared}</span>{rest}')
    return _esc(text)


def _block(idx: int, chunk, bg: str, *, nested: bool = False) -> str:
    role = chunk.meta.get("role")
    tag = f"#{idx}"
    if role == "parent":
        tag = f"parent #{chunk.meta.get('parent_index', idx)}"
    elif role == "child":
        tag = f"child of #{chunk.meta.get('parent_index', '?')}"
    ov = chunk.meta.get("overlap_chars", 0)
    ov_note = f' · <span style="color:#b45309">overlap {ov} chars</span>' if ov else ""
    chosen = chunk.meta.get("chosen")
    chosen_note = f' · via {html.escape(str(chosen))}' if chosen else ""

    margin = "margin-left:26px;" if nested else ""
    header = (f'<div style="font:600 11px/1.4 ui-monospace,monospace;'
              f'color:#475569;margin-bottom:4px;">{tag} · {chunk.token_count} tokens'
              f'{ov_note}{chosen_note}</div>')
    body = (f'<div style="font:13px/1.55 ui-sans-serif,system-ui;color:{_TEXT};'
            f'white-space:pre-wrap;word-break:break-word;">{_chunk_body(chunk.text, ov)}</div>')
    return (f'<div style="{margin}background:{bg};border:1px solid {_BORDER};'
            f'border-radius:8px;padding:10px 12px;margin-bottom:8px;">'
            f'{header}{body}</div>')


def render_chunks_html(result: ChunkResult) -> str:
    """Full HTML for one strategy's chunk column."""
    if not result.ok:
        return (f'<div style="padding:16px;border:1px dashed #cbd5e1;border-radius:8px;'
                f'background:#f8fafc;color:#64748b;font:13px/1.5 ui-sans-serif;">'
                f'<b>{html.escape(result.strategy)}</b><br>{html.escape(result.note)}</div>')
    if not result.chunks:
        return ('<div style="padding:16px;color:#64748b;">No chunks produced.</div>')

    is_hier = any(c.level == 1 for c in result.chunks)
    parts = []
    alt = 0
    for i, c in enumerate(result.chunks):
        if is_hier:
            if c.level == 0:
                parts.append(_block(c.meta.get("parent_index", i), c, _PARENT_BG))
            else:
                bg = _BLOCK_A if alt % 2 == 0 else _BLOCK_B
                alt += 1
                parts.append(_block(i, c, bg, nested=True))
        else:
            bg = _BLOCK_A if i % 2 == 0 else _BLOCK_B
            parts.append(_block(i, c, bg))
    return "".join(parts)


def metrics_row(result: ChunkResult) -> dict:
    """One row of size metrics for a strategy (used in the metrics panel)."""
    row = {"strategy": result.strategy}
    if not result.ok:
        row.update(chunks=0, mean=None, median=None, min=None, max=None, std=None,
                   note=result.note)
        return row
    tc = result.token_counts
    if not tc:
        row.update(chunks=0, mean=None, median=None, min=None, max=None, std=None, note="")
        return row
    row.update(
        chunks=len(tc),
        mean=round(stats.fmean(tc), 1),
        median=int(stats.median(tc)),
        min=min(tc),
        max=max(tc),
        std=round(stats.pstdev(tc), 1) if len(tc) > 1 else 0.0,
        note="",
    )
    return row
