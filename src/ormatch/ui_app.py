"""Streamlit UI for ORMatch. Launched via `ormatch ui`; run directly with `streamlit run ui_app.py`."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ormatch.cli import discover_index_dirs, prepare_query, rank_prepared

st.set_page_config(page_title="ORMatch", layout="wide")
st.title("ORMatch: reviewer suggestions")
st.caption("Your manuscript never leaves this machine. Only the public paper index is used.")


def _code_version() -> str:
    """mtimes of the ormatch sources; part of the cache key so a code edit never leaves a stale
    matcher object (old class) inside Streamlit's cache."""
    import ormatch
    root = Path(ormatch.__file__).parent
    return str(max(p.stat().st_mtime for p in root.glob("*.py")))


@st.cache_resource(show_spinner="Reading the PDF, embedding it and matching its bibliography...")
def _prepare(pdf_bytes: bytes, index_dirs: tuple[str, ...], backend: str | None, code_version: str) -> dict:
    """Expensive part, cached on the PDF's content: moving a slider only re-ranks."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        prep = prepare_query(tmp_path, [Path(d) for d in index_dirs], backend)
    finally:
        tmp_path.unlink(missing_ok=True)
    pp = prep["papers"]
    ids = pp["openalex_work_id"].astype(str)
    prep["abstracts"] = dict(zip(ids, pp["abstract"].fillna("")))
    prep["venues"] = dict(zip(ids, pp["venue"].fillna("").astype(str) + " " + pp["year"].astype("Int64").astype(str)))
    prep["rank_cache"] = {}
    return prep


def _rank_cached(prep: dict, **kw) -> dict:
    """Re-rank only when a parameter changed; clicking a row must not re-rank."""
    key = json.dumps({k: sorted(v) if isinstance(v, (set, list)) else v for k, v in kw.items()}, sort_keys=True, default=str)
    cache = prep["rank_cache"]
    if key not in cache:
        if len(cache) > 50:
            cache.clear()
        cache[key] = rank_prepared(prep, **kw)
    return cache[key]


with st.sidebar:
    index_dir = st.text_input("Core index directory", os.environ.get("ORMATCH_INDEX_DIR", "data/index"))
    available = [str(d) for d in discover_index_dirs(Path(index_dir))]
    index_dirs = st.multiselect("Indexes to search", available, default=available[:1] or None,
                                help="The core OR/MS index plus any add-on collections you have downloaded (data/index_<name>)")
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
    volume_correction = st.slider(
        "Correct for prolific authors", 0.0, 1.0, 0.0, 0.05,
        help="Removes the chance advantage of having many indexed papers (expected best-of-n cosine). "
             "Turn up to surface people who are a close fit but have few papers instead of the usual suspects.")
    editor_weight = st.slider("Penalty per current editorial role", 0.0, 0.2, 0.0, 0.01,
                              help="Needs data/editors.csv (author or name, journal, role).")
    seniority_weight = st.slider("Penalty for seniority (log works count above median)", 0.0, 0.1, 0.0, 0.005,
                                 help="Needs data/authors.parquet from scripts/fetch_author_stats.py.")
    early_career_weight = st.slider("Boost early-career researchers", 0.0, 0.1, 0.0, 0.005,
                                    help="Bonus for people with few OpenAlex works (PhD students, postdocs); full at 1 work, zero at the cutoff below.")
    early_career_max_works = st.slider("Early-career cutoff (OpenAlex works)", 5, 40, 15)
    min_or_links = st.slider("Add-on authors: minimum links to OR literature", 0, 10, 1,
                             help="Core-venue papers plus citations to/from core papers. Only applies when add-on collections are searched.")

    st.markdown("### Seeds and diversity")
    seeds_txt = st.text_area("Reviewers you already have in mind (names or IDs, one per line)", height=70,
                             help="Pulls the list toward people whose body of work resembles these seeds.")
    seed_weight = st.slider("Seed pull", 0.0, 0.5, 0.2, 0.05)
    diversity = st.slider("Diversity", 0.0, 1.0, 0.0, 0.05,
                          help="Re-ranks so consecutive picks cover different neighbourhoods rather than clones of the seeds (maximal marginal relevance).")

    st.markdown("### Conflicts of interest")
    ms_authors = st.text_area("Manuscript authors (names or OpenAlex IDs, one per line)", height=90,
                              help="Resolved against the index. Their co-authors, colleagues and genealogy links are flagged.")
    coi_years = st.slider("Co-authorship window (years)", 1, 20, 5)
    coi_same_inst = st.checkbox("Flag same institution", value=True)
    coi_mode = st.radio("Conflicted candidates", ["flag with evidence", "exclude"], index=0, horizontal=True)
    excl_inst = st.text_area("Always exclude institutions (OpenAlex IDs, one per line)", height=60)
    excl_auth = st.text_area("Always exclude authors (OpenAlex IDs, one per line)", height=60)

uploaded = st.file_uploader("Drag and drop a manuscript PDF", type=["pdf"])

if uploaded is not None:
    try:
        if not index_dirs:
            st.error("Select at least one index"); st.stop()
        prep = _prepare(uploaded.getvalue(), tuple(index_dirs), backend, _code_version())
        res = _rank_cached(
            prep, n=n,
            exclude_institutions={s.strip() for s in excl_inst.splitlines() if s.strip()},
            exclude_authors={s.strip() for s in excl_auth.splitlines() if s.strip()},
            lam=lam, k=k, half_life=half_life, cite_weight=cite_weight, min_papers=min_papers,
            manuscript_authors=[a.strip() for a in ms_authors.splitlines() if a.strip()],
            coi_years=float(coi_years), coi_same_institution=coi_same_inst,
            coi_mode="exclude" if coi_mode == "exclude" else "flag",
            editor_weight=editor_weight, seniority_weight=seniority_weight,
            min_or_links=min_or_links, volume_correction=volume_correction,
            early_career_weight=early_career_weight, early_career_max_works=early_career_max_works,
            seed_reviewers=[a.strip() for a in seeds_txt.splitlines() if a.strip()],
            seed_weight=seed_weight, diversity=diversity,
        )
    except Exception as e:  # show, do not crash
        st.error(f"Failed: {e!r}")
        st.stop()

    abstracts, venues = prep["abstracts"], prep["venues"]

    # ---- header -----------------------------------------------------------------------
    st.subheader(res["title"] or "(no title found)")
    st.caption(f"{res['n_index_papers']:,} papers searched from: {', '.join(res['collections'])}  ·  "
               f"bibliography: {res['n_references']} entries, {res['n_references_in_index']} matched")
    with st.expander("Manuscript abstract and matched references"):
        st.write(res["abstract"])
        if res["cited_papers"]:
            st.markdown("**Cited papers found in the index**")
            for pid, title in res["cited_papers"]:
                st.write(f"- {title} ({venues.get(pid, '')})")
    if res.get("manuscript_authors"):
        unresolved = [q for q, ids in res["manuscript_authors"].items() if not ids]
        multi = {q: len(ids) for q, ids in res["manuscript_authors"].items() if len(ids) > 1}
        msg = f"{res['n_conflicts_flagged']} potential conflicts flagged (COI column; evidence is a hint to follow up, not a verdict)."
        if unresolved:
            msg += f" Not found in index: {', '.join(unresolved)}."
        if multi:
            msg += " Ambiguous names matched several OpenAlex authors: " + ", ".join(f"{q} ({v})" for q, v in multi.items()) + "."
        st.info(msg)
    if prep.get("n_editors_known") == 0 and editor_weight:
        st.warning("No data/editors.csv found; the editorial penalty has no effect.")
    if prep.get("n_author_stats") == 0 and seniority_weight:
        st.warning("No data/authors.parquet found; run scripts/fetch_author_stats.py for the seniority penalty.")

    # ---- the objective with the current weights, and the scale it operates on -------------
    prm = res["params"]
    base = [r["components"].get("best paper (1-lam)*max", 0) + r["components"].get("top-k mean lam*mean", 0) for r in res["reviewers"]]
    terms = [f"{1 - prm['lam']:.2f} · max_sim + {prm['lam']:.2f} · mean_top{prm['k']}"]
    if prm["cite_weight"]:
        terms.append(f"+ {prm['cite_weight']:.2f} · (1 − 0.5^n_cited)")
    if prm.get("seed_weight"):
        terms.append(f"+ {prm['seed_weight']:.2f} · sim_to_seed")
    if prm.get("early_career_weight"):
        terms.append(f"+ {prm['early_career_weight']:.3f} · early_career")
    if prm.get("volume_correction"):
        terms.append(f"− {prm['volume_correction']:.2f} · E[max of n random]")
    if prm.get("editor_weight"):
        terms.append(f"− {prm['editor_weight']:.2f} · editor_roles")
    if prm.get("seniority_weight"):
        terms.append(f"− {prm['seniority_weight']:.3f} · log_works_above_median")
    with st.container(border=True):
        st.markdown("**score = " + " ".join(terms) + "**")
        if base:
            st.caption(f"Scale: among the {len(base)} listed, the text-similarity part ranges {min(base):.3f} to {max(base):.3f} "
                       f"(spread {max(base) - min(base):.3f}). A weight of 0.05 is therefore about "
                       f"{(0.05 / max(max(base) - min(base), 1e-6)):.1f}x the whole gap between #1 and #{len(base)}; "
                       "the detail panel shows each term per person."
                       + (f"  Diversity {prm['diversity']:.2f}: order re-ranked by marginal relevance." if prm.get("diversity") else ""))

    # ---- ranked list (left) + details of the selected row (right) -----------------------
    rows = []
    for i, r in enumerate(res["reviewers"], 1):
        rows.append({
            "#": i,
            "Reviewer": r.get("author_name"),
            "Institution": (r.get("institution") or "")[:60],
            "Score": round(float(r["score"]), 3),
            "Cited": r.get("n_cited", 0) or None,
            "Papers": r.get("n_papers"),
            "COI?": "⚠" if r.get("coi") else "",
        })
    table = pd.DataFrame(rows)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Ranked reviewers")
        st.caption("Click a row to see why.")
        sel = st.dataframe(table, use_container_width=True, hide_index=True,
                           on_select="rerun", selection_mode="single-row", height=min(38 * (len(rows) + 1), 900))
        picked = sel.selection.rows[0] if sel and sel.selection and sel.selection.rows else 0
    with right:
        r = res["reviewers"][picked] if res["reviewers"] else None
        if r:
            st.markdown(f"#### {r['author_name']}")
            st.caption((r.get("institution") or "institution unknown") + f"  ·  {r['n_papers']} indexed papers"
                       + (f"  ·  cited {r['n_cited']}x here" if r.get("n_cited") else ""))
            if r.get("coi"):
                st.warning(f"Possible conflict: {r['coi']}")
            for pid, title, score in r.get("evidence") or []:
                with st.expander(f"{title}  ({venues.get(pid, '')}, sim {float(score):.2f})", expanded=False):
                    st.write(abstracts.get(pid) or "_no abstract in the index_")
            with st.expander("Score breakdown", expanded=False):
                comp = r.get("components") or {}
                st.table(pd.DataFrame({"term": list(comp), "value": [f"{v:+.3f}" for v in comp.values()]}).set_index("term"))
                st.caption(f"total {float(r['score']):.3f}")
            with st.expander("More about this person", expanded=False):
                facts = [f"score {float(r['score']):.3f}"]
                if r.get("seniority") is not None:
                    facts.append(f"{int(r['seniority'])} works on OpenAlex")
                if r.get("n_editor_roles"):
                    facts.append(f"{r['n_editor_roles']:g} editorial role(s)")
                if r.get("or_links") is not None:
                    facts.append(f"{r['or_links']} links to OR literature")
                facts.append(f"OpenAlex {r['author_id']}")
                st.write(" · ".join(facts))

    # ---- downloads ------------------------------------------------------------------------
    flat = []
    for i, r in enumerate(res["reviewers"], 1):
        row = {"rank": i, "author_name": r["author_name"], "openalex_author_id": r["author_id"],
               "institution": r.get("institution") or "", "score": round(float(r["score"]), 4),
               "n_indexed_papers": r["n_papers"], "n_cited_by_manuscript": r.get("n_cited", 0),
               "coi_flag": r.get("coi") or "", "or_links": r.get("or_links"), "openalex_works": r.get("seniority"),
               "editor_roles": r.get("n_editor_roles")}
        for j, (pid, title, score) in enumerate(r.get("evidence") or [], 1):
            row[f"evidence{j}_title"] = title
            row[f"evidence{j}_venue"] = venues.get(pid, "")
            row[f"evidence{j}_similarity"] = round(float(score), 4)
            row[f"evidence{j}_abstract"] = abstracts.get(pid, "")
        flat.append(row)
    full = dict(res)
    for r in full["reviewers"]:
        r["evidence"] = [{"paper_id": pid, "title": t, "venue": venues.get(pid, ""), "similarity": float(sc),
                          "abstract": abstracts.get(pid, "")} for pid, t, sc in r.get("evidence") or []]
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV (names, evidence titles and abstracts)", data=pd.DataFrame(flat).to_csv(index=False),
                       file_name="ormatch_suggestions.csv", mime="text/csv")
    c2.download_button("Download JSON (everything, incl. weights used)", data=json.dumps(full, indent=1, default=str),
                       file_name="ormatch_suggestions.json", mime="application/json")
