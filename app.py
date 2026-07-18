"""Chunk Lab: Streamlit entrypoint.

Two tabs, one per stage. Tab 1 (Visualize) makes chunk trade-offs visible.
Tab 2 (Evaluate) scores strategies against a query set, built in Stage 2.
Both read the same chunker registry, so a strategy written once shows up in
both places automatically.

Run:  uv run streamlit run app.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from chunklab.registry import available_strategies, run_strategy
# Stage 1 symbols only at module scope. Stage 2 symbols (render_retrievals_html,
# eval, chromadb) are imported lazily inside the Evaluate tab, so the app always
# loads even if a Stage 2 dependency is missing or a hosted runtime serves a
# stale module cache.
from chunklab.viz import render_chunks_html, metrics_row

ROOT = Path(__file__).parent
SAMPLE_DOC = ROOT / "data" / "sample_doc.md"
SAMPLE_EVAL = ROOT / "data" / "sample_eval.json"

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

@st.cache_data(show_spinner=False)
def _run_eval(doc_text, queries_json, keys, chunk_size, overlap, k):
    """Cache a full eval run keyed on its inputs, so drill-down reruns don't
    recompute. Args are hashable (queries passed as a JSON string)."""
    from chunklab.eval import run_full_eval
    queries = json.loads(queries_json)
    return run_full_eval(doc_text, queries, list(keys), chunk_size=chunk_size,
                         overlap=overlap, k=k)


def render_eval():
    st.subheader("Which chunking strategy actually wins?")
    st.caption("Score each strategy on a query set with known answers. A retrieved "
               "chunk is a hit if it contains the query's gold answer string "
               "(case-insensitive). Read recall@k with MRR: recall asks whether a "
               "correct chunk made the top k at all, MRR how high it ranked.")

    strategies, labels, key_by_label = _strategy_options()

    # --- eval source ---
    from chunklab.eval import load_eval_spec
    from chunklab.viz import render_retrievals_html   # lazy: Stage 2 only
    source = st.radio("Eval set", ["Sample", "Upload JSON"], horizontal=True)
    try:
        if source == "Upload JSON":
            up = st.file_uploader("Eval file (.json)", type=["json"])
            if up is None:
                st.info("Upload an eval JSON, or switch to Sample.")
                return
            doc_text, queries, doc_path = load_eval_spec(json.load(up), ROOT)
        else:
            doc_text, queries, doc_path = load_eval_spec(str(SAMPLE_EVAL), ROOT)
    except Exception as e:  # noqa: BLE001 - bad file shouldn't crash the tab
        st.error(f"Could not load eval set: {e}")
        return
    st.caption(f"Document: `{os.path.basename(doc_path)}` · {len(queries)} queries")

    # --- controls ---
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        eval_defaults = [labels[x] for x in
                         ("fixed", "recursive", "document", "semantic", "hierarchical")
                         if x in labels]
        picked_labels = st.multiselect("Strategies to compare",
                                       options=list(key_by_label.keys()),
                                       default=eval_defaults)
    with c2:
        k = int(st.number_input("k (top-k)", 1, 20, 5))
    with c3:
        chunk_size = int(st.number_input("chunk_size", 32, 4096, 256, 32, key="eval_cs"))
    with c4:
        overlap = int(st.number_input("overlap", 0, 1024, 32, 8, key="eval_ov"))

    picked = [key_by_label[lbl] for lbl in picked_labels]
    unavailable = [s["label"] for s in strategies if not s["available"]]
    if unavailable:
        st.caption(f"Disabled (needs Ollama): {', '.join(unavailable)}.")
    if not picked:
        st.info("Pick at least one strategy to compare.")
        return

    if st.button("Run eval", type="primary"):
        st.session_state.eval_ran = True
    if not st.session_state.get("eval_ran"):
        st.info("Choose k and strategies, then Run eval.")
        return

    try:
        with st.spinner("Chunking, embedding, indexing, and scoring each strategy…"):
            results = _run_eval(doc_text, json.dumps(queries), tuple(picked),
                                chunk_size, overlap, k)
    except Exception as e:  # noqa: BLE001 - surface, don't crash the tab
        st.error(f"Eval could not run: {e}. On a fresh hosted deploy the vector "
                 f"store may still be installing; wait a moment and try again.")
        return

    ok = [e for e in results if e.ok]
    failed = [e for e in results if not e.ok]

    # --- ranked results table ---
    st.markdown(f"**Results, ranked by recall@{k}**")
    rows = [{"strategy": e.strategy, f"recall@{k}": e.recall_at_k, "MRR": e.mrr,
             "hit rate": e.hit_rate, "chunks": e.n_units, "mean tok": e.mean_tokens}
            for e in ok]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    for e in failed:
        st.caption(f"· {e.strategy}: {e.note}")

    # --- bar chart: recall@k and MRR per strategy ---
    if ok:
        chart_df = pd.DataFrame(
            {f"recall@{k}": [e.recall_at_k for e in ok], "MRR": [e.mrr for e in ok]},
            index=[e.strategy for e in ok])
        st.bar_chart(chart_df, height=280)

    # --- drill-down: why a score is what it is ---
    if ok:
        st.divider()
        st.markdown("**Drill-down** — pick a query and see the top-k retrieved chunks "
                    "per strategy, with the matched answer highlighted.")
        qi = st.selectbox("Query", range(len(queries)),
                          format_func=lambda i: queries[i]["query"])
        st.caption(f"Gold answer: `{queries[qi]['answer']}`")
        cols = st.columns(len(ok))
        for col, e in zip(cols, ok):
            with col:
                qe = e.per_query[qi]
                mark = "✓" if qe.recall else "✗"
                st.markdown(f"**{e.strategy}** {mark}")
                st.markdown(render_retrievals_html(qe), unsafe_allow_html=True)


# ------------------------------------------------------------------------ main

st.title("🔪 Chunk Lab")
st.caption("See and evaluate RAG chunking strategies on your own data, "
           "local, offline, no API keys.")

tab_viz, tab_eval = st.tabs(["Visualize", "Evaluate"])
with tab_viz:
    render_visualizer()
with tab_eval:
    render_eval()
