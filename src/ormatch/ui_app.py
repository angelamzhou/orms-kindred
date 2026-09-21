"""Streamlit UI for ORMatch. Launched via `ormatch ui`; run directly with `streamlit run ui_app.py`."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ormatch import learn
from ormatch.cli import PERSONAL_COI, discover_index_dirs, load_personal_conflicts, prepare_query, rank_prepared, save_personal_conflicts

st.set_page_config(page_title="ORMatch", layout="wide")
st.title("ORMatch: reviewer suggestions")
st.caption("Your manuscript never leaves this machine. Only the public paper index is used.")


def _code_version() -> str:
    """mtimes of the ormatch sources; part of the cache key so a code edit never leaves a stale
    matcher object (old class) inside Streamlit's cache."""
    import ormatch
    root = Path(ormatch.__file__).parent
    return str(max(p.stat().st_mtime for p in root.glob("*.py")))


@st.cache_resource(show_spinner="Embedding the manuscript and matching its bibliography...")
def _prepare(pdf_bytes: bytes | None, title: str, abstract: str, refs_text: str,
             index_dirs: tuple[str, ...], backend: str | None, code_version: str) -> dict:
    """Expensive part, cached on the manuscript content: moving a slider only re-ranks."""
    dirs = [Path(d) for d in index_dirs]
    if pdf_bytes:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)
        try:
            prep = prepare_query(tmp_path, dirs, backend)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        prep = prepare_query(None, dirs, backend, title=title, abstract=abstract, references_text=refs_text or None)
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


def _describe(prep: dict, aid: str) -> str:
    m = prep["matcher"]
    rows = m.author_rows.get(aid)
    n = len(rows) if rows is not None else 0
    inst = "; ".join(sorted(m.author_inst_names.get(aid, set())))[:45] or "institution unknown"
    sample = ""
    if n:
        pid = prep["index"].paper_ids[int(rows[0])]
        sample = (prep["titles"].get(pid) or "")[:60]
    return f"{m.author_name.get(aid, aid)} — {inst} — {n} paper{'s' if n != 1 else ''} — e.g. {sample} [{aid}]"


def _confirm(prep: dict, names: list[str], label: str) -> list[str]:
    """For each typed name, show the OpenAlex authors it matches and let the user pick one, all, or none.
    Returns the confirmed author ids. Ambiguous names default to 'all matches' (conservative)."""
    if not names:
        return []
    resolved = prep["matcher"].resolve_authors(names)
    chosen: list[str] = []
    st.markdown(f"**Confirm {label}**")
    for q, ids in resolved.items():
        if not ids:
            st.warning(f"{q}: no author with this name in the index (check spelling, or paste the OpenAlex ID).")
            continue
        if len(ids) == 1:
            st.caption(f"{q} → {_describe(prep, ids[0])}")
            chosen.append(ids[0]); continue
        opts = [f"all {len(ids)} matches"] + [_describe(prep, i) for i in ids] + ["none of these"]
        pick = st.selectbox(f"{q} matches {len(ids)} OpenAlex authors", opts, index=0, key=f"confirm_{label}_{q}")
        if pick.startswith("all "):
            chosen.extend(ids)
        elif pick != "none of these":
            chosen.append(ids[opts.index(pick) - 1])
    return chosen


SAVED = learn.load_params() or {}
def _d(name, default):
    """slider default: learned/saved weight if present, else the built-in default"""
    v = SAVED.get(name)
    return float(v) if isinstance(v, (int, float)) else default

with st.sidebar:
    if SAVED:
        st.caption(f"Slider defaults come from your learned weights ({learn.WEIGHTS}).")
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
        "Cited-author bonus", 0.0, 0.5, min(0.5, _d("cite_weight", 0.1)), 0.01,
        help="Bonus for authors of papers the manuscript cites: weight x (1 - 0.5^n) for n cited papers. "
             "0 ignores the bibliography. Leave-one-out MRR peaks around 0.1-0.2.")
    lam = st.slider(
        "Best paper vs. body of work", 0.0, 1.0, _d("lam", 0.3), 0.05,
        help="0 scores an author by their single most similar paper (favours specialists with one "
             "close paper); 1 uses the mean of their top-k papers (favours sustained work on the topic).")
    k = st.slider("Papers per author in the mean (k)", 1, 10, 3)
    half_life_on = st.checkbox("Prefer recently active authors", value=False)
    half_life = st.slider("Recency half-life (years)", 2.0, 30.0, 8.0, 1.0,
                          help="An author's paper this many years old keeps half its advantage over an average paper.") if half_life_on else None
    min_papers = st.slider("Minimum indexed papers per author", 1, 10, 1)
    volume_correction = st.slider(
        "Correct for prolific authors", 0.0, 1.0, min(1.0, _d("volume_correction", 0.0)), 0.05,
        help="Removes the chance advantage of having many indexed papers (expected best-of-n cosine). "
             "Turn up to surface people who are a close fit but have few papers instead of the usual suspects.")
    editor_weight = st.slider("Penalty per current editorial role", 0.0, 0.2, min(0.2, _d("editor_weight", 0.0)), 0.01,
                              help="Needs data/editors.csv (author or name, journal, role).")
    seniority_weight = st.slider("Penalty for seniority (log works count above median)", 0.0, 0.1, min(0.1, _d("seniority_weight", 0.0)), 0.005,
                                 help="Needs data/authors.parquet from scripts/fetch_author_stats.py.")
    early_career_weight = st.slider("Boost early-career researchers", 0.0, 0.1, min(0.1, _d("early_career_weight", 0.0)), 0.005,
                                    help="Bonus for people with few OpenAlex works (PhD students, postdocs); full at 1 work, zero at the cutoff below.")
    early_career_max_works = st.slider("Early-career cutoff (OpenAlex works)", 5, 40, 15)
    min_or_links = st.slider("Add-on authors: minimum links to OR literature", 0, 10, 1,
                             help="Core-venue papers plus citations to/from core papers. Only applies when add-on collections are searched.")

    st.markdown("### Seeds and diversity")
    seed_weight = st.slider("Seed pull", 0.0, 0.5, min(0.5, _d("seed_weight", 0.2)), 0.05, help="How strongly the seed reviewers entered in step 1 tilt the list.")
    diversity = st.slider("Diversity", 0.0, 1.0, 0.0, 0.05,
                          help="Re-ranks so consecutive picks cover different neighbourhoods rather than clones of the seeds (maximal marginal relevance).")

    st.markdown("### Conflicts of interest")
    coi_years = st.slider("Co-authorship window (years)", 1, 20, 5)
    coi_same_inst = st.checkbox("Flag same institution", value=True)
    coi_mode = st.radio("Conflicted candidates", ["flag with evidence", "exclude"], index=0, horizontal=True)

st.markdown("## 1. Who is involved")
c1, c2, c3 = st.columns(3)
ms_authors = c1.text_area("Manuscript authors", height=110, placeholder="one name or OpenAlex ID per line",
                          help="Resolved against the index. Their co-authors, colleagues and genealogy links are flagged as conflicts.")
personal = c2.text_area("My declared conflicts", value="\n".join(load_personal_conflicts()), height=110,
                        placeholder="people the data cannot know about",
                        help=f"Always flagged (or excluded). Remembered between sessions in {PERSONAL_COI}.")
seeds_txt = c3.text_area("Reviewers already in mind (optional)", height=110, placeholder="seeds: pulls the list toward similar people",
                         help="Seeds are removed from the output; the list tilts toward people whose work resembles them.")
personal_list = [a.strip() for a in personal.splitlines() if a.strip()]
if personal_list != load_personal_conflicts():
    save_personal_conflicts(personal_list)
with st.expander("Always exclude by OpenAlex ID (optional)"):
    excl_inst = st.text_area("Institution IDs, one per line", height=60)
    excl_auth = st.text_area("Author IDs, one per line", height=60)

st.markdown("## 2. Manuscript")
mode = st.radio("Provide the manuscript as", ["title and abstract (typed)", "PDF upload"], horizontal=True,
                help="Typing only the title and abstract means the tool never touches the full manuscript. "
                     "A PDF also gives the reference list, which improves the ranking.")
uploaded = None; q_title = q_abstract = q_refs = ""
if mode == "PDF upload":
    uploaded = st.file_uploader("Drag and drop the PDF", type=["pdf"])
else:
    q_title = st.text_input("Title")
    q_abstract = st.text_area("Abstract", height=150)
    q_refs = st.text_area("Reference list (optional; paste the bibliography to enable the citation bonus)", height=100)
have_input = uploaded is not None or bool(q_title.strip() or q_abstract.strip())

if have_input:
    try:
        if not index_dirs:
            st.error("Select at least one index"); st.stop()
        prep = _prepare(uploaded.getvalue() if uploaded is not None else None, q_title, q_abstract, q_refs,
                        tuple(index_dirs), backend, _code_version())
        typed_authors = [a.strip() for a in ms_authors.splitlines() if a.strip()]
        typed_seeds = [a.strip() for a in seeds_txt.splitlines() if a.strip()]
        if typed_authors or typed_seeds:
            with st.container(border=True):
                author_ids = _confirm(prep, typed_authors, "manuscript authors")
                seed_ids = _confirm(prep, typed_seeds, "seed reviewers")
        else:
            author_ids, seed_ids = [], []
        res = _rank_cached(
            prep, n=n,
            exclude_institutions={s.strip() for s in excl_inst.splitlines() if s.strip()},
            exclude_authors={s.strip() for s in excl_auth.splitlines() if s.strip()},
            lam=lam, k=k, half_life=half_life, cite_weight=cite_weight, min_papers=min_papers,
            manuscript_authors=author_ids,
            coi_years=float(coi_years), coi_same_institution=coi_same_inst,
            coi_mode="exclude" if coi_mode == "exclude" else "flag",
            editor_weight=editor_weight, seniority_weight=seniority_weight,
            min_or_links=min_or_links, volume_correction=volume_correction,
            early_career_weight=early_career_weight, early_career_max_works=early_career_max_works,
            seed_reviewers=seed_ids,
            seed_weight=seed_weight, diversity=diversity, personal_conflicts=personal_list,
        )
    except Exception as e:  # show, do not crash
        st.error(f"Failed: {e!r}")
        st.stop()

    abstracts, venues = prep["abstracts"], prep["venues"]

    # ---- header -----------------------------------------------------------------------
    st.markdown("## 3. Suggested reviewers")
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
        multi = {}
        if unresolved:
            st.warning(f"Not found in index: {', '.join(unresolved)}.")
    if prep.get("n_editors_known") == 0 and editor_weight:
        st.warning("No data/editors.csv found; the editorial penalty has no effect.")
    if prep.get("n_author_stats") == 0 and seniority_weight:
        st.warning("No data/authors.parquet found; run scripts/fetch_author_stats.py for the seniority penalty.")

    # ---- the objective with the current weights, and the scale it operates on -------------
    prm = res["params"]
    base = [r["components"].get("best paper (1-lam)*max", 0) + r["components"].get("top-k mean lam*mean", 0) for r in res["reviewers"]]
    lam = prm["lam"]
    terms = []
    if lam < 1:
        terms.append(f"{1 - lam:.2f} × (best-paper similarity)")
    if lam > 0:
        terms.append(f"{lam:.2f} × (mean similarity of top {prm['k']} papers)")
    if prm["cite_weight"]:
        terms.append(f"{prm['cite_weight']:.2f} × (1 − 0.5^(papers cited by manuscript))")
    if prm.get("seed_weight"):
        terms.append(f"{prm['seed_weight']:.2f} × (similarity to closest seed − pool median)")
    if prm.get("early_career_weight"):
        terms.append(f"{prm['early_career_weight']:.3f} × (early-career factor, 1 → 0 by {prm['early_career_max_works']} works)")
    if prm.get("volume_correction"):
        terms.append(f"− {prm['volume_correction']:.2f} × (expected best-of-n similarity for n random papers)")
    if prm.get("editor_weight"):
        terms.append(f"− {prm['editor_weight']:.2f} × (editorial roles; past roles count ½)")
    if prm.get("seniority_weight"):
        terms.append(f"− {prm['seniority_weight']:.3f} × (log publications above median author)")
    objective = "score = " + " ".join(t if i == 0 or t.startswith("−") else "+ " + t for i, t in enumerate(terms))
    notes = []
    if prm.get("half_life"):
        notes.append(f"similarities shrink toward the average by ½ every {prm['half_life']:g} years of paper age")
    if prm.get("diversity"):
        notes.append(f"list re-ordered with diversity {prm['diversity']:.2f} (marginal relevance against seeds and earlier picks)")
    with st.container(border=True):
        st.markdown(f"**{objective}**" + (("  \n_" + "; ".join(notes) + "_") if notes else ""))
        if base:
            spread = max(base) - min(base)
            st.caption(f"Scale: the text-similarity part ranges {min(base):.3f} to {max(base):.3f} among the {len(base)} listed "
                       f"(spread {spread:.3f}). A bonus of 0.05 is about {0.05 / max(spread, 1e-6):.1f} times the whole gap between #1 and #{len(base)}. "
                       "Click a row and open 'Score breakdown' to see each term for that person.")

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
            "Conflict?": (r.get("coi") or "")[:48],
        })
    table = pd.DataFrame(rows)
    n_coi = sum(1 for r in res["reviewers"] if r.get("coi"))
    def _hl(row):
        return ["background-color: #ffe0e0; color: #7a0000" if row["Conflict?"] else "" for _ in row]
    styled = table.style.apply(_hl, axis=1)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Ranked reviewers")
        st.caption("Click a row to see why this person is suggested."
                   + (f"  Red rows ({n_coi}) have a potential conflict; the reason is in the last column." if n_coi else ""))
        sel = st.dataframe(styled, use_container_width=True, hide_index=True,
                           on_select="rerun", selection_mode="single-row", height=min(38 * (len(rows) + 1), 900))
        picked = sel.selection.rows[0] if sel and sel.selection and sel.selection.rows else 0
    with right:
        r = res["reviewers"][picked] if res["reviewers"] else None
        if r:
            st.markdown(f"#### {r['author_name']}")
            st.caption((r.get("institution") or "institution unknown") + f"  ·  {r['n_papers']} indexed papers"
                       + (f"  ·  cited {r['n_cited']}x here" if r.get("n_cited") else ""))
            if r.get("coi"):
                st.error(f"⚠ Potential conflict of interest: {r['coi']}. Verify before inviting.")
            for pid, title, score in r.get("evidence") or []:
                with st.expander(f"{title}  ({venues.get(pid, '')}, sim {float(score):.2f})", expanded=False):
                    st.write(abstracts.get(pid) or "_no abstract in the index_")
            mkey = learn.manuscript_key(res["title"], res["abstract"])
            b1, b2, b3 = st.columns([1, 1, 3])
            if b1.button("👍 good fit", key=f"up_{r['author_id']}"):
                learn.record(mkey, r["author_id"], r["author_name"], +1, r.get("features", {}))
                for o in res["reviewers"]:  # unrated shown candidates become weak negatives
                    if o["author_id"] != r["author_id"] and not any(x["author_id"] == o["author_id"] and x["manuscript"] == mkey and x["label"] != 0 for x in learn.load()):
                        learn.record(mkey, o["author_id"], o["author_name"], 0, o.get("features", {}))
                st.toast(f"Recorded: {r['author_name']} is a good fit")
            if b2.button("👎 poor fit", key=f"down_{r['author_id']}"):
                learn.record(mkey, r["author_id"], r["author_name"], -1, r.get("features", {}))
                st.toast(f"Recorded: {r['author_name']} is a poor fit")
            rated = {x["author_id"]: x["label"] for x in learn.load() if x["manuscript"] == mkey and x["label"] != 0}
            if r["author_id"] in rated:
                b3.caption("your rating: " + ("👍" if rated[r["author_id"]] > 0 else "👎"))
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

    # ---- feedback and learned weights ------------------------------------------------------------
    fb = learn.load()
    n_rated = sum(1 for x in fb if x["label"] != 0)
    with st.expander(f"Learn locally from my ratings ({n_rated} ratings on {len({x['manuscript'] for x in fb})} manuscripts)", expanded=False):
        st.caption("Rate candidates with the buttons in the detail panel. The fit is a pairwise logistic regression "
                   "that pulls the current slider settings toward weights ranking your 👍 above your 👎 (and above unrated "
                   "candidates shown at the time). Everything stays on this machine.")
        l2 = st.slider("Stay close to current settings (regularisation)", 0.1, 10.0, 1.0, 0.1)
        if st.button("Fit weights", disabled=n_rated < 2):
            fitres = learn.fit(fb, res["params"], l2=l2)
            st.session_state["fit"] = fitres
        fitres = st.session_state.get("fit")
        if fitres:
            keys = ["lam", "cite_weight", "volume_correction", "editor_weight", "seniority_weight", "early_career_weight", "seed_weight"]
            st.table(pd.DataFrame({"weight": keys, "current": [round(float(res["params"].get(k, 0) or 0), 3) for k in keys],
                                   "learned": [round(float(fitres["params"].get(k, 0) or 0), 3) for k in keys]}).set_index("weight"))
            if fitres.get("train_acc") is not None:
                st.caption(f"{fitres['n_pairs']} preference pairs; pairs ordered correctly: current {fitres['base_acc']:.0%} -> learned {fitres['train_acc']:.0%}")
            if st.button("Save learned weights as my defaults"):
                learn.save_params(fitres["params"])
                st.success(f"Saved to {learn.WEIGHTS}. Reload the page to use them as slider defaults.")
        if fb and st.button("Delete all my ratings"):
            learn.FEEDBACK.unlink(missing_ok=True); st.session_state.pop("fit", None); st.rerun()

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
    full = copy.deepcopy(res)  # res is memoised; never mutate it
    for r in full["reviewers"]:
        r["evidence"] = [{"paper_id": pid, "title": t, "venue": venues.get(pid, ""), "similarity": float(sc),
                          "abstract": abstracts.get(pid, "")} for pid, t, sc in r.get("evidence") or []]
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV (names, evidence titles and abstracts)", data=pd.DataFrame(flat).to_csv(index=False),
                       file_name="ormatch_suggestions.csv", mime="text/csv")
    c2.download_button("Download JSON (everything, incl. weights used)", data=json.dumps(full, indent=1, default=str),
                       file_name="ormatch_suggestions.json", mime="application/json")
