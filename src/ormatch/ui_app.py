"""Streamlit UI for ORMatch. Launched via `ormatch ui`; run directly with `streamlit run ui_app.py`."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ormatch.cli import prepare_query, rank_prepared

st.set_page_config(page_title="ORMatch", layout="wide")
st.title("ORMatch: reviewer suggestions")
st.caption("Your manuscript never leaves this machine. Only the public paper index is used.")


@st.cache_resource(show_spinner="Reading the PDF, embedding it and matching its bibliography...")
def _prepare(pdf_bytes: bytes, index_dir: str, backend: str | None) -> dict:
    """Expensive part, cached on the PDF's content: moving a slider only re-ranks."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        return prepare_query(tmp_path, Path(index_dir), backend)
    finally:
        tmp_path.unlink(missing_ok=True)


with st.sidebar:
    index_dir = st.text_input("Index directory", os.environ.get("ORMATCH_INDEX_DIR", "data/index"))
    backend = st.selectbox("Embedding backend", ["index default", "specter2", "scincl", "tfidf"],
                           help="'index default' uses the backend recorded in the index's meta.json")
    backend = None if backend == "index default" else backend
    n = st.slider("Number of reviewers", 5, 50, 20)

    st.markdown("### What to prioritise")
    cite_weight = st.slider(
        "Cited-author bonus", 0.0, 0.5, 0.1, 0.01,
        help="Bonus for authors of papers the manuscript cites: weight x (1 - 0.5^n) for n cited papers. "
             "0 ignores the bibliography. Leave-one-out MRR peaks around 0.1-0.2.")
    lam = st.slider(
        "Best paper vs. body of work", 0.0, 1.0, 0.3, 0.05,
        help="0 scores an author by their single most similar paper (favours specialists with one "
             "close paper); 1 uses the mean of their top-k papers (favours sustained work on the topic).")
    k = st.slider("Papers per author in the mean (k)", 1, 10, 3)
    half_life_on = st.checkbox("Prefer recently active authors", value=False)
    half_life = st.slider("Recency half-life (years)", 2.0, 30.0, 8.0, 1.0,
                          help="An author's paper this many years old keeps half its advantage over an average paper.") if half_life_on else None
    min_papers = st.slider("Minimum indexed papers per author", 1, 10, 1)

    st.markdown("### Conflicts")
    excl_inst = st.text_area("Exclude institutions (OpenAlex IDs, one per line)", height=80)
    excl_auth = st.text_area("Exclude authors (OpenAlex IDs, one per line)", height=80)
    exclude_coauthors = st.checkbox("Also exclude their co-authors", value=True)

uploaded = st.file_uploader("Drag and drop a manuscript PDF", type=["pdf"])

if uploaded is not None:
    try:
        prep = _prepare(uploaded.getvalue(), index_dir, backend)
        res = rank_prepared(
            prep, n,
            exclude_institutions={s.strip() for s in excl_inst.splitlines() if s.strip()},
            exclude_authors={s.strip() for s in excl_auth.splitlines() if s.strip()},
            exclude_coauthors=exclude_coauthors, lam=lam, k=k, half_life=half_life,
            cite_weight=cite_weight, min_papers=min_papers,
        )
    except Exception as e:  # show, do not crash
        st.error(f"Failed: {e!r}")
        st.stop()

    st.subheader(res["title"] or "(no title found)")
    with st.expander("Extracted abstract"):
        st.write(res["abstract"])
    with st.expander(f"Bibliography: {res['n_references']} entries parsed, {res['n_references_in_index']} matched to indexed papers"):
        for pid, title in res["cited_papers"]:
            st.write(f"- {title}  ({pid})")

    rows = []
    for i, r in enumerate(res["reviewers"], 1):
        ev = r.get("evidence") or []
        rows.append({
            "#": i,
            "Reviewer": r.get("author_name"),
            "Institution": r.get("institution") or "",
            "Score": round(float(r["score"]), 3),
            "Cited": r.get("n_cited", 0),
            "Papers": r.get("n_papers"),
            "Top evidence": ev[0][1] if ev else "",
            "OpenAlex ID": r.get("author_id"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Evidence")
    for r in res["reviewers"]:
        cited = f", cites {r['n_cited']} of their papers" if r.get("n_cited") else ""
        with st.expander(f"{r.get('author_name')}  ({r.get('institution') or 'unknown'})  score={float(r['score']):.3f}{cited}"):
            for pid, title, score in r.get("evidence") or []:
                st.write(f"- {title}  (sim {float(score):.2f}, {pid})")

    st.download_button("Download JSON", data=pd.Series(res).to_json(), file_name="ormatch_suggestions.json")
