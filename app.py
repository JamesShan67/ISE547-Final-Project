# app.py
"""
Smart FAQ Assistant – user-facing web app.

Features:
- Upload a PDF or text document.
- Automatically generates FAQs from the content.
- Clean, collapsible FAQ view.
- Search box: keyword filter; if nothing matches, show top-3 most similar FAQs.
- Optional CSV download.
"""

import io
import os
import tempfile
import difflib

import streamlit as st
import pandas as pd

from faq_core import (
    generate_faq_for_document,
    MAX_CHARS_PER_CHUNK,
)

# ----------- Page config -----------

st.set_page_config(
    page_title="Smart FAQ Assistant",
    layout="wide",
)

# ----------- Session state init -----------

if "faqs_df" not in st.session_state:
    st.session_state["faqs_df"] = None
if "current_file_name" not in st.session_state:
    st.session_state["current_file_name"] = None

# ----------- Header -----------

st.title("Smart FAQ Assistant")

st.markdown(
    """
Turn long documents into a quick, readable FAQ.

1. **Upload** a PDF or text file  
2. **Generate** FAQs with one click  
3. **Browse & search** the questions you care about  
"""
)

# ----------- Sidebar / Options -----------

max_chars = MAX_CHARS_PER_CHUNK

with st.sidebar:
    st.header("Options")

    with st.expander("Advanced settings", expanded=False):
        max_chars = st.slider(
            "Max characters per chunk",
            min_value=500,
            max_value=4000,
            value=max_chars,
            step=100,
            help="Larger chunks keep more context, smaller chunks create more localized FAQs.",
        )

# ----------- Helper: similarity-based search -----------

def find_top_similar_faqs(df: pd.DataFrame, query: str, top_n: int = 3) -> pd.DataFrame:
    """
    Rank FAQs by string similarity to the query (question or answer) and return top_n.
    """
    query = (query or "").strip().lower()
    if not query or df.empty:
        return pd.DataFrame(columns=df.columns)

    scores = []
    for idx, row in df.iterrows():
        q_text = str(row.get("question", "")).lower()
        a_text = str(row.get("answer", "")).lower()

        s_q = difflib.SequenceMatcher(None, query, q_text).ratio()
        s_a = difflib.SequenceMatcher(None, query, a_text).ratio()
        score = max(s_q, s_a)

        scores.append((score, idx))

    scores.sort(reverse=True, key=lambda x: x[0])

    best_rows = []
    for score, idx in scores[:top_n]:
        if score > 0:
            best_rows.append(df.loc[idx])

    if not best_rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(best_rows).reset_index(drop=True)


# ----------- File upload + generate button -----------

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "md"],
    help="Supported formats: PDF, TXT, Markdown.",
)

if uploaded_file is not None:
    # Save uploaded content to a temp file so the core pipeline can read it by path
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success(f"File uploaded: **{uploaded_file.name}**")

    if st.button("✨ Generate FAQs"):
        with st.spinner("Analyzing your document and generating FAQs..."):
            faqs = generate_faq_for_document(
                file_path=tmp_path,
                max_chars_per_chunk=max_chars,
            )

        if not faqs:
            st.warning(
                "No FAQs were generated. Try a different document or adjust the settings."
            )
            st.session_state["faqs_df"] = None
            st.session_state["current_file_name"] = None
        else:
            st.session_state["faqs_df"] = pd.DataFrame(faqs)
            st.session_state["current_file_name"] = uploaded_file.name

# ----------- Show FAQs + search (if we have any) -----------

df = st.session_state["faqs_df"]

if df is None or df.empty:
    st.info("Upload a document and click **Generate FAQs** to get started.")
else:
    st.subheader("Your FAQs")

    # Search bar
    search_query = st.text_input(
        "Search within FAQs",
        "",
        placeholder="Type a keyword or phrase (e.g., refund, deadline, camera settings)...",
    )
    query = search_query.strip()

    if not query:
        # No query -> show all
        filtered_df = df.reset_index(drop=True)
        st.caption(f"Showing all {len(filtered_df)} FAQs")
    else:
        # First try keyword filter
        mask_q = df["question"].str.contains(query, case=False, na=False)
        mask_a = df["answer"].str.contains(query, case=False, na=False)
        filtered_df = df[mask_q | mask_a].reset_index(drop=True)

        if filtered_df.empty:
            # Fallback: top-3 similar FAQs
            similar_df = find_top_similar_faqs(df, query, top_n=3)
            if similar_df.empty:
                st.warning(
                    f"No FAQs match or are similar to **“{query}”**. Try another keyword."
                )
            else:
                st.info(
                    f"No exact matches found. Showing the **{len(similar_df)} most similar** FAQs instead."
                )
                filtered_df = similar_df
        else:
            st.caption(
                f"Found {len(filtered_df)} FAQs matching **“{query}”**."
            )

    # Show FAQs as expanders
    if not filtered_df.empty:
        for idx, row in filtered_df.iterrows():
            q = row["question"]
            a = row["answer"]
            with st.expander(f"Q{idx+1}. {q}"):
                st.write(a)

    # Download section
    st.markdown("---")
    st.subheader("Download")

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    file_base = (
        st.session_state["current_file_name"].rsplit(".", 1)[0]
        if st.session_state["current_file_name"]
        else "faqs"
    )
    st.download_button(
        label="Download all FAQs as CSV",
        data=csv_buf.getvalue(),
        file_name=f"{file_base}_faqs.csv",
        mime="text/csv",
    )

    st.caption("FAQs are generated automatically from your document content.")
