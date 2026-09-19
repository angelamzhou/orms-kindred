"""Streamlit UI for ORMatch. Launched via `ormatch ui`; run directly with `streamlit run ui_app.py`."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ormatch.cli import suggest_for_pdf

st.set_page_config(page_title="ORMatch", layout="wide")
st.title("ORMatch: reviewer suggestions")
st.caption("Your manuscript never leaves this machine. Only the public paper index is used.")

with st.sidebar:
    index_dir = Path(st.text_input("Index directory", os.environ.get("ORMATCH_INDEX_DIR", "data/index")))
    backend = st.selectbox("Embedding backend", ["specter2", "scincl", "tfidf"])
    n = st.slider("Number of reviewers", 5, 50, 20)
    excl_inst = st.text_area("Exclude institutions (OpenAlex IDs, one per line)", height=80)
    excl_auth = st.text_area("Exclude authors (OpenAlex IDs, one per line)", height=80)

uploaded = st.file_uploader("Drag and drop a manuscript PDF", type=["pdf"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)
    try:
        with st.spinner("Extracting text, embedding, and searching the index..."):
            res = suggest_for_pdf(
                tmp_path, n,
                exclude_institutions={s.strip() for s in excl_inst.splitlines() if s.strip()},
                exclude_authors={s.strip() for s in excl_auth.splitlines() if s.strip()},
                index_dir=index_dir, backend=backend,
            )
    except Exception as e:  # show, do not crash
        st.error(f"Failed: {e!r}")
        st.stop()
    finally:
        tmp_path.unlink(missing_ok=True)

    st.subheader(res["title"] or "(no title found)")
    with st.expander("Extracted abstract"):
        st.write(res["abstract"])

    rows = []
    for i, r in enumerate(res["reviewers"], 1):
        ev = r.get("evidence") or []
        rows.append({
            "#": i,
            "Reviewer": r.get("author_name"),
            "Institution": r.get("institution") or "",
            "Score": round(float(r["score"]), 3),
            "Top evidence": ev[0][1] if ev else "",
            "OpenAlex ID": r.get("author_id"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Evidence")
    for r in res["reviewers"]:
        with st.expander(f"{r.get('author_name')}  ({r.get('institution') or 'unknown'})  score={float(r['score']):.3f}"):
            for pid, title, score in r.get("evidence") or []:
                st.write(f"- {title}  (sim {float(score):.2f}, {pid})")

    st.download_button("Download JSON", data=pd.Series(res).to_json(), file_name="ormatch_suggestions.json")
