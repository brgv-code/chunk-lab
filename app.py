"""Chunk Lab: Streamlit entrypoint.

Two tabs, one per stage. Tab 1 (Visualize) makes chunk trade-offs visible.
Tab 2 (Evaluate) scores strategies against a query set, built in Stage 2.
Both read the same chunker registry, so a strategy written once shows up in
both places automatically.

Run:  uv run streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from chunklab.registry import available_strategies, run_strategy
from chunklab.viz import render_chunks_html, metrics_row

ROOT = Path(__file__).parent
SAMPLE_DOC = ROOT / "data" / "sample_doc.md"

st.set_page_config(page_title="Chunk Lab", page_icon="🔪", layout="wide")


@st.cache_data(show_spinner=False)
def _run(key, text, chunk_size, overlap, treat_as):
    """Cache chunking per (strategy, text, params) so unrelated reruns are cheap
    (semantic embeds the text, so this matters)."""
    return run_strategy(key, text, chunk_size=chunk_size, overlap=overlap,
                        treat_as=treat_as)


def _strategy_options():
    strategies = available_strategies()
    labels, key_by_label, default = {}, {}, []
    for s in strategies:
        suffix = "" if s["available"] else "  (needs Ollama, off)"
        label = f"{s['label']}{suffix}"
        labels[s["key"]] = label
        key_by_label[label] = s["key"]
    return strategies, labels, key_by_label


# ------------------------------------------------------------------ Visualizer

def render_visualizer():
    st.subheader("What do your chunks actually look like?")
    st.caption("Paste text, pick a few strategies, and see the chunks each one "
               "produces side by side. Alternating colors mark boundaries; the "
               "amber span is text shared with the previous chunk (overlap).")

    strategies, labels, key_by_label = _strategy_options()

    # --- input ---
    if "text" not in st.session_state:
        st.session_state.text = ""

    c_load, c_upload = st.columns([1, 3])
    with c_load:
        if st.button("Load sample doc", use_container_width=True):
            st.session_state.text = SAMPLE_DOC.read_text()
    with c_upload:
        up = st.file_uploader("…or upload a .txt / .md file",
                              type=["txt", "md"], label_visibility="collapsed")
        if up is not None:
            st.session_state.text = up.read().decode("utf-8", errors="replace")

    text = st.text_area("Input text", key="text", height=200,
                        placeholder="Paste a document here, or load the sample.")

    # --- controls ---
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        picked_labels = st.multiselect(
            "Strategies",
            options=list(key_by_label.keys()),
            default=[labels["fixed"], labels["recursive"],
                     labels["document"], labels["semantic"]],
        )
    with col2:
        chunk_size = st.number_input("chunk_size (tokens)", 32, 4096, 256, 32)
    with col3:
        overlap = st.number_input("overlap (tokens)", 0, 1024, 32, 8)
    with col4:
        treat_as = st.selectbox("treat input as", ["auto", "prose", "markdown", "code"])

    picked = [key_by_label[lbl] for lbl in picked_labels]
    unavailable = [s["label"] for s in strategies if not s["available"]]
    if unavailable:
        st.caption(f"Disabled (shown but off): {', '.join(unavailable)}, "
                   f"start Ollama to enable LLM-based chunking.")

    if not text.strip():
        st.info("Load the sample doc or paste some text to begin.")
        return
    if not picked:
        st.info("Select at least one strategy.")
        return

    results = [_run(k, text, int(chunk_size), int(overlap), treat_as) for k in picked]

    # --- side-by-side chunk columns ---
    if len(picked) > 4:
        st.caption("Tip: 3 to 4 strategies at a time are easiest to compare.")
    cols = st.columns(len(results))
    for col, res in zip(cols, results):
        with col:
            st.markdown(f"**{res.strategy}**")
            st.markdown(render_chunks_html(res), unsafe_allow_html=True)

    # --- metrics panel ---
    st.divider()
    st.markdown("**Chunk-size metrics**: where the trade-off becomes concrete: "
                "big chunks average unrelated ideas, tiny chunks lose context.")
    rows = [metrics_row(r) for r in results]
    df = pd.DataFrame(rows)[
        ["strategy", "chunks", "mean", "median", "min", "max", "std", "note"]
    ].rename(columns={"mean": "mean tok", "median": "median tok",
                      "min": "min tok", "max": "max tok", "std": "std tok"})
    st.dataframe(df, hide_index=True, use_container_width=True)


# -------------------------------------------------------------------- Eval tab

def render_eval():
    st.subheader("Which chunking strategy actually wins?")
    st.info("The retrieval eval harness is built in Stage 2 (after GATE 1). "
            "It reuses the same registry to score every strategy on a query "
            "set with recall@k and MRR.")


# ------------------------------------------------------------------------ main

st.title("🔪 Chunk Lab")
st.caption("See and evaluate RAG chunking strategies on your own data, "
           "local, offline, no API keys.")

tab_viz, tab_eval = st.tabs(["Visualize", "Evaluate"])
with tab_viz:
    render_visualizer()
with tab_eval:
    render_eval()
