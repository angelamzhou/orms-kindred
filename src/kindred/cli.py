"""Kindred command-line interface (typer)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Kindred: local reviewer matching for OR/MS/OM papers.", no_args_is_help=True)
console = Console(stderr=True)

DEFAULT_INDEX = Path("data/index")
# Public index releases: https://github.com/angelamzhou/orms-kindred/releases (GitHub release
# assets, not git LFS: assets allow 2 GB files and unlimited downloads).
INDEX_RELEASES = "https://github.com/angelamzhou/orms-kindred/releases/download"
DEFAULT_INDEX_URL = f"{INDEX_RELEASES}/index-v1/kindred-index-v1.tar.gz"


# --------------------------------------------------------------------------- core
def _load_tables(index_dir: Path):
    """Load authorships/papers parquet tables that live next to the index."""
    import pandas as pd

    def _find(name: str) -> Path:
        for cand in (index_dir / f"{name}.parquet", index_dir.parent / f"{name}.parquet",
                     index_dir.parent / "processed" / f"{name}.parquet"):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"{name}.parquet not found under {index_dir} or its parent")

    return pd.read_parquet(_find("authorships")), pd.read_parquet(_find("papers"))


def discover_index_dirs(root: Path = DEFAULT_INDEX) -> list[Path]:
    """The core index dir plus any add-on collection dirs: siblings named like `index_<name>`
    (e.g. data/index_applied-or) and subdirectories of root that contain a meta.json."""
    dirs = [root] if (root / "meta.json").exists() else []
    if root.parent.exists():
        for d in sorted(root.parent.glob(f"{root.name}_*")):
            if (d / "meta.json").exists() and d != root and not d.name.endswith("_scincl"):
                dirs.append(d)
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                dirs.append(d)
    return dirs


def load_indexes(index_dirs: list[Path]):
    """Load one or more self-contained index dirs and return (index, authorships, papers)."""
    import pandas as pd

    from kindred.index import PaperIndex, concat_indexes

    idxs, auths, paps, refs = [], [], [], []
    for d in index_dirs:
        ix = PaperIndex.load(str(d))
        ix.meta.setdefault("collection", d.name)
        a, pp = _load_tables(d)
        idxs.append(ix); auths.append(a); paps.append(pp)
        for cand in (d / "references.parquet", d.parent / "references.parquet"):
            if cand.exists():
                refs.append(pd.read_parquet(cand)); break
    references = pd.concat(refs, ignore_index=True).drop_duplicates() if refs else None
    core_ids = set(idxs[0].paper_ids)  # the first index dir is the core collection
    if len(idxs) == 1:
        return idxs[0], auths[0], paps[0], references, core_ids
    papers = pd.concat(paps, ignore_index=True).drop_duplicates("openalex_work_id")
    authorships = pd.concat(auths, ignore_index=True).drop_duplicates(["work_id", "position"])
    return concat_indexes(idxs), authorships, papers, references, core_ids


PERSONAL_COI = Path.home() / ".kindred" / "conflicts.txt"


def load_personal_conflicts(path: Path = PERSONAL_COI) -> list[str]:
    """Names/ids the user declares as conflicts, one per line (# comments allowed). Local only."""
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def save_personal_conflicts(names: list[str], path: Path = PERSONAL_COI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# orms-kindred personal conflicts: one name or OpenAlex author id per line\n" + "\n".join(names) + "\n", encoding="utf-8")


def _load_side_tables(index_dir: Path) -> tuple[dict, dict, list]:
    """Optional local tables next to the core index:
    editors.csv (author_id or name, journal, role) from scripts/scrape_editorial_boards.py,
    authors.parquet (author_id, works_count, ...) from scripts/fetch_author_stats.py,
    coi/genealogy.csv (person_a, person_b, relation, source) from scripts/fetch_genealogy.py or by hand."""
    import csv

    import pandas as pd

    root = index_dir.parent
    editors: dict = {}
    p = root / "editors.csv"
    if p.exists():
        import datetime
        this_year = datetime.date.today().year
        for r in csv.DictReader(open(p, encoding="utf-8")):
            key = (r.get("author_id") or "").strip() or ("name:" + (r.get("name") or "").strip())
            if not key.strip(":"):
                continue
            # a role seen in the last two snapshots counts fully; a past role counts half
            # (the "held many editorial positions" proxy)
            last = int(r["last_year"]) if (r.get("last_year") or "").isdigit() else this_year
            editors[key] = editors.get(key, 0) + (1.0 if last >= this_year - 1 else 0.5)
    stats: dict = {}
    p = root / "authors.parquet"
    if p.exists():
        a = pd.read_parquet(p)
        col = "works_count" if "works_count" in a.columns else a.columns[1]
        stats = dict(zip(a["author_id"].astype(str), a[col].astype(float)))
    pairs: list = []
    p = root / "coi" / "genealogy.csv"
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("person_a") and r.get("person_b"):
                pairs.append((r["person_a"], r["person_b"], r.get("relation", ""), r.get("source", "")))
    return editors, stats, pairs


def prepare_query(pdf: Optional[Path] = None, index_dir: Path | list[Path] = DEFAULT_INDEX, backend: Optional[str] = None,
                  parse_references: bool = True, title: Optional[str] = None, abstract: Optional[str] = None,
                  references_text: Optional[str] = None) -> dict:
    """Expensive, parameter-free half of the pipeline: manuscript -> title/abstract -> embedding, plus
    the bibliography matched to indexed papers. The manuscript is either a PDF or typed
    title/abstract (and optionally a pasted reference list); typing avoids handling the full
    PDF at all. The result can be re-ranked many times with different weights (rank_prepared)."""
    from kindred.embed import Embedder
    from kindred.index import PaperIndex
    from kindred.pdf import extract_references, match_references, parse_reference_text, pdf_to_query

    if pdf is not None:
        q = pdf_to_query(pdf)
    elif title or abstract:
        q = {"title": (title or "").strip(), "abstract": (abstract or "").strip()}
    else:
        raise ValueError("give a PDF or a title/abstract")
    index_dirs = [Path(d) for d in (index_dir if isinstance(index_dir, (list, tuple)) else [index_dir])]
    idx, authorships, papers, references, core_ids = load_indexes(index_dirs)
    index_dir = index_dirs[0]
    backend = backend or idx.meta.get("backend") or "specter2"
    if idx.meta.get("backend") and backend != idx.meta["backend"]:
        console.print(f"[yellow]warning:[/yellow] index was built with {idx.meta['backend']} but querying with {backend}")
    embedder = Embedder(backend)
    if embedder.backend == "tfidf":  # fitted vectorizer is stored beside the index
        tfidf_path = index_dir / "tfidf.pkl"
        if not tfidf_path.exists():
            raise FileNotFoundError(f"tfidf backend needs {tfidf_path} (built with the index)")
        embedder.load(str(tfidf_path))
    qvec = embedder.encode([q["title"]], [q["abstract"]])[0]
    if references_text:
        refs = parse_reference_text(references_text)
    elif pdf is not None and parse_references:
        refs = extract_references(pdf)
    else:
        refs = []
    cited_ids = match_references(refs, papers) if refs else []
    titles = dict(zip(papers["openalex_work_id"].astype(str), papers["title"].astype(str))) if "title" in papers.columns else {}
    from kindred.match import ReviewerMatcher

    # Built once; rank_prepared only flips its cheap parameters, so re-ranking is ~instant.
    matcher = ReviewerMatcher(idx, authorships, papers)
    editors, stats, pairs = _load_side_tables(index_dir)
    # editors keyed by name -> resolve to ids in this index
    ed_ids: dict = {}
    for key, n in editors.items():
        ids = [key] if not key.startswith("name:") else matcher.resolve_authors([key[5:]]).get(key[5:], [])
        for i in ids:
            ed_ids[i] = ed_ids.get(i, 0) + n
    matcher.set_editor_roles(ed_ids)
    matcher.set_author_stats(stats)
    if len(index_dirs) > 1:
        matcher.compute_or_links(core_ids, references)
    return {"pdf": str(pdf) if pdf else None, "title": q["title"], "abstract": q["abstract"], "qvec": qvec, "index": idx,
            "authorships": authorships, "papers": papers, "titles": titles, "references": refs,
            "cited_ids": cited_ids, "backend": embedder.backend, "matcher": matcher,
            "genealogy_pairs": pairs, "n_editors_known": len(ed_ids), "n_author_stats": len(stats),
            "multi": len(index_dirs) > 1}


def rank_prepared(prep: dict, n: int = 20, exclude_institutions: set[str] | None = None,
                  exclude_authors: set[str] | None = None, exclude_coauthors: bool = True,
                  lam: float = 0.3, k: int = 3, half_life: Optional[float] = None,
                  cite_weight: float = 0.1, min_papers: int = 1,
                  manuscript_authors: Optional[list[str]] = None, coi_years: Optional[float] = 5.0,
                  coi_same_institution: bool = True, coi_mode: str = "flag",
                  editor_weight: float = 0.0, seniority_weight: float = 0.0,
                  min_or_links: Optional[int] = None, volume_correction: float = 0.0,
                  early_career_weight: float = 0.0, early_career_max_works: int = 15,
                  seed_reviewers: Optional[list[str]] = None, seed_weight: float = 0.2,
                  diversity: float = 0.0, personal_conflicts: Optional[list[str]] = None) -> dict:
    """Cheap half: aggregate similarities into reviewer scores with the given weights.

    lam          0 = score an author by their single most similar paper; 1 = by the mean of their
                 top-k papers. In between mixes the two (OpenReview/TPMS-style pooling choices).
    k            how many of an author's papers enter the mean.
    half_life    years; shrinks older papers' advantage over an average paper by half every
                 half_life years (None = off). Prefers active reviewers at some cost in accuracy.
    cite_weight  bonus for authors of papers the manuscript cites: cite_weight * (1 - 0.5**n).
    min_papers   drop authors with fewer indexed papers.
    manuscript_authors  names or OpenAlex ids of the manuscript's authors; resolved against the
                 index. Their co-authors (papers within coi_years), colleagues at the same
                 institution and genealogy links are potential conflicts.
    coi_mode     "flag" (default): keep conflicted candidates in the list with the evidence in
                 a `coi` field so a human can follow up; "exclude": drop them.
    editor_weight     penalty per known current editorial role (needs data/editors.csv).
    seniority_weight  penalty per log-unit of works_count above the median author (needs
                 data/authors.parquet).
    min_or_links  when add-on collections are searched, drop authors with fewer links to the
                 core OR literature (core papers + citations to/from core); default 1.
    volume_correction  0..1: subtract the chance advantage of having many indexed papers
                 (expected best-of-n cosine for random papers). 1 = fully chance-corrected;
                 surfaces close fits with few papers instead of the usual suspects.
    """
    matcher = prep["matcher"]
    matcher.lam, matcher.k, matcher.min_papers = lam, k, min_papers
    matcher.set_recency(half_life, prep["papers"])
    resolved = matcher.resolve_authors(manuscript_authors or [])
    seeds_resolved = matcher.resolve_authors(seed_reviewers or [])
    seed_ids = {i for ids in seeds_resolved.values() for i in ids}
    ms_ids = {i for ids in resolved.values() for i in ids} | set(exclude_authors or [])
    coi = matcher.conflicts_for(ms_ids, coauthor_years=coi_years, same_institution=coi_same_institution,
                                extra_pairs=[(a, b) for a, b, *_ in prep.get("genealogy_pairs", [])]) if ms_ids else {}
    # user-declared conflicts (personal list): always flagged, whatever the data says
    pc = personal_conflicts if personal_conflicts is not None else load_personal_conflicts()
    for q, ids in matcher.resolve_authors(pc).items():
        for i in ids:
            coi[i] = f"declared conflict ({q})"
    if min_or_links is None:
        min_or_links = 1 if prep.get("multi") else 0
    hard_exclude = set(coi) if coi_mode == "exclude" else set(ms_ids)  # authors themselves always out
    hard_exclude |= seed_ids  # the user already has the seeds; show who else is like them
    cands = matcher.rank(
        prep["qvec"], top_n=n,
        manuscript_author_ids=None,
        excluded_institution_ids=exclude_institutions or None,
        exclude_coauthors=False,  # handled by conflicts_for (windowed, with reasons)
        cited_papers=prep["cited_ids"], cite_weight=cite_weight,
        exclude_ids=hard_exclude, editor_weight=editor_weight, seniority_weight=seniority_weight,
        min_or_links=min_or_links, volume_correction=volume_correction,
        early_career_weight=early_career_weight, early_career_max_works=early_career_max_works,
        seed_author_ids=seed_ids, seed_weight=seed_weight if seed_ids else 0.0, diversity=diversity,
    )
    titles = prep["titles"]
    reviewers = []
    for c in cands:
        reviewers.append({
            "coi": coi.get(c.author_id),
            "components": {k: round(v, 4) for k, v in c.components.items()},
            "features": {k: round(v, 5) for k, v in c.features.items()},
            "n_editor_roles": c.n_editor_roles,
            "seniority": c.seniority,
            "or_links": c.or_links,
            "author_id": c.author_id,
            "author_name": c.author_name,
            "institution": "; ".join(c.institutions) if c.institutions else None,
            "score": float(c.score),
            "n_papers": c.n_papers,
            "n_cited": c.n_cited,
            "evidence": [(pid, titles.get(pid, pid), float(s)) for pid, s in c.evidence],
        })
    return {"pdf": prep["pdf"], "title": prep["title"], "abstract": prep["abstract"], "backend": prep["backend"],
            "collections": prep["index"].meta.get("collections") or [prep["index"].meta.get("collection", "core")],
            "n_index_papers": len(prep["index"]),
            "n_references": len(prep["references"]), "n_references_in_index": len(prep["cited_ids"]),
            "cited_papers": [(pid, titles.get(pid, pid)) for pid in prep["cited_ids"]],
            "manuscript_authors": {q: ids for q, ids in resolved.items()},
            "seed_reviewers": {q: ids for q, ids in seeds_resolved.items()},
            "n_conflicts_flagged": sum(1 for r in reviewers if r["coi"]),
            "params": {"n": n, "lam": lam, "k": k, "half_life": half_life, "cite_weight": cite_weight,
                       "min_papers": min_papers, "coi_years": coi_years, "coi_mode": coi_mode,
                       "editor_weight": editor_weight, "seniority_weight": seniority_weight,
                       "min_or_links": min_or_links, "volume_correction": volume_correction,
                       "early_career_weight": early_career_weight, "early_career_max_works": early_career_max_works,
                       "seed_weight": seed_weight if seed_ids else 0.0, "diversity": diversity},
            "reviewers": reviewers}


def suggest_for_pdf(pdf: Path, n: int = 20, exclude_institutions: set[str] | None = None,
                    exclude_authors: set[str] | None = None, index_dir: Path | list[Path] = DEFAULT_INDEX,
                    backend: Optional[str] = None, k_hits: int = 200, cite_weight: float = 0.1,
                    lam: float = 0.3, k: int = 3, half_life: Optional[float] = None, **rank_kw) -> dict:
    """Full pipeline: PDF -> title/abstract -> embed -> search -> rank reviewers (prepare + rank)."""
    prep = prepare_query(pdf, index_dir, backend, parse_references=bool(cite_weight))
    return rank_prepared(prep, n, exclude_institutions, exclude_authors, lam=lam, k=k,
                         half_life=half_life, cite_weight=cite_weight, **rank_kw)


def _print_result(res: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(res, indent=2, default=str))
        return
    console.print(f"[bold]Title:[/bold] {res['title']}")
    console.print(f"[bold]Abstract:[/bold] {res['abstract'][:300]}{'...' if len(res['abstract']) > 300 else ''}")
    console.print(f"[bold]Index:[/bold] {res.get('n_index_papers', '?')} papers from {', '.join(res.get('collections', []))}")
    console.print(f"[bold]References:[/bold] {res.get('n_references', 0)} parsed, {res.get('n_references_in_index', 0)} matched to indexed papers\n")
    if res.get("manuscript_authors"):
        for q, ids in res["manuscript_authors"].items():
            console.print(f"[bold]Author[/bold] {q} -> {', '.join(ids) if ids else '[red]not found in index[/red]'}")
        console.print(f"[bold]Potential conflicts flagged:[/bold] {res.get('n_conflicts_flagged', 0)} (follow up; --coi exclude drops them)\n")
    t = Table(title=f"Suggested reviewers ({len(res['reviewers'])})")
    for col in ("#", "Reviewer", "Institution", "Score", "Cited", "COI?", "Evidence (top paper)"):
        t.add_column(col)
    for i, r in enumerate(res["reviewers"], 1):
        ev = r.get("evidence") or []
        top = f"{ev[0][1][:60]} ({ev[0][2]:.2f})" if ev else ""
        flag = f"[red]{r['coi']}[/red]" if r.get("coi") else ""
        t.add_row(str(i), str(r.get("author_name")), str(r.get("institution") or ""), f"{r['score']:.3f}",
                  str(r.get("n_cited", 0) or ""), flag, top)
    Console().print(t)


# ----------------------------------------------------------------------- commands
@app.command()
def suggest(
    pdf: Optional[Path] = typer.Argument(None, exists=True, readable=True, help="Manuscript PDF (omit and pass --title/--abstract instead)"),
    title: Optional[str] = typer.Option(None, "--title", help="Manuscript title (alternative to a PDF)"),
    abstract: Optional[str] = typer.Option(None, "--abstract", help="Manuscript abstract (alternative to a PDF)"),
    references: Optional[Path] = typer.Option(None, "--references", help="Text file with the reference list (optional, with --title/--abstract)"),
    n: int = typer.Option(20, "--n", help="Number of reviewers"),
    exclude_institution: list[str] = typer.Option([], "--exclude-institution", help="OpenAlex institution ID (repeatable)"),
    exclude_author: list[str] = typer.Option([], "--exclude-author", help="OpenAlex author ID (repeatable)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    index_dir: list[Path] = typer.Option([DEFAULT_INDEX], "--index-dir", help="Index directory; repeat to query the core index plus add-on collections together"),
    all_collections: bool = typer.Option(False, "--all-collections", help="Also use every add-on index found next to the first --index-dir (data/index_<name>)"),
    backend: Optional[str] = typer.Option(None, "--backend", help="specter2 | scincl | tfidf (default: backend recorded in the index)"),
    cite_weight: float = typer.Option(0.1, "--cite-weight", help="Boost authors of papers the manuscript cites (0 disables bibliography parsing)"),
    lam: float = typer.Option(0.3, "--lam", min=0.0, max=1.0, help="0 = best single paper, 1 = mean of top-k papers"),
    k: int = typer.Option(3, "--k", min=1, help="Papers per author entering the mean"),
    half_life: Optional[float] = typer.Option(None, "--half-life", help="Recency half-life in years (off by default)"),
    author: list[str] = typer.Option([], "--author", help="Manuscript author name or OpenAlex id (repeatable); used to flag conflicts"),
    coi_years: float = typer.Option(5.0, "--coi-years", help="Co-authorship window in years for conflict flags"),
    coi: str = typer.Option("flag", "--coi", help="flag = keep conflicted candidates with the evidence; exclude = drop them"),
    editor_weight: float = typer.Option(0.0, "--editor-weight", help="Penalty per current editorial role (needs data/editors.csv)"),
    seniority_weight: float = typer.Option(0.0, "--seniority-weight", help="Penalty per log-unit of works_count above median (needs data/authors.parquet)"),
    min_or_links: Optional[int] = typer.Option(None, "--min-or-links", help="With add-on collections: required links to core OR literature (default 1)"),
    volume_correction: float = typer.Option(0.0, "--volume-correction", min=0.0, max=1.0, help="0..1: remove the chance advantage of prolific authors (surfaces close fits with few papers)"),
    early_career_weight: float = typer.Option(0.0, "--early-career-weight", help="Bonus for authors with few OpenAlex works (PhD students, postdocs); fades to 0 at --early-career-max-works"),
    early_career_max_works: int = typer.Option(15, "--early-career-max-works"),
    seed: list[str] = typer.Option([], "--seed", help="Reviewer you already have in mind (name or OpenAlex id, repeatable); pulls the list toward similar people"),
    seed_weight: float = typer.Option(0.2, "--seed-weight", help="Bonus = weight x cosine between a candidate's profile and the closest seed profile"),
    diversity: float = typer.Option(0.0, "--diversity", min=0.0, max=1.0, help="0..1 maximal-marginal-relevance re-ranking: penalise candidates close to seeds or already-picked reviewers"),
    conflict: list[str] = typer.Option([], "--conflict", help="Declare a conflict (name or id, repeatable); added to ~/.kindred/conflicts.txt for future runs"),
):
    """Suggest reviewers for one PDF. Runs entirely offline against the local index."""
    dirs = discover_index_dirs(index_dir[0]) if all_collections else list(index_dir)
    if conflict:
        save_personal_conflicts(sorted(set(load_personal_conflicts()) | set(conflict)))
    if pdf is None and not (title or abstract):
        raise typer.BadParameter("give a PDF path or --title/--abstract")
    prep = prepare_query(pdf, dirs, backend, parse_references=bool(cite_weight), title=title, abstract=abstract,
                         references_text=references.read_text(encoding="utf-8") if references else None)
    res = rank_prepared(prep, n, set(exclude_institution), set(exclude_author),
                          cite_weight=cite_weight, lam=lam, k=k, half_life=half_life,
                          manuscript_authors=list(author), coi_years=coi_years, coi_mode=coi,
                          editor_weight=editor_weight, seniority_weight=seniority_weight, min_or_links=min_or_links,
                          volume_correction=volume_correction, early_career_weight=early_career_weight,
                          early_career_max_works=early_career_max_works, seed_reviewers=list(seed),
                          seed_weight=seed_weight, diversity=diversity)
    _print_result(res, json_out)


@app.command()
def batch(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    n: int = typer.Option(20, "--n"),
    index_dir: Path = typer.Option(DEFAULT_INDEX, "--index-dir"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write JSONL here (default: stdout)"),
):
    """Suggest reviewers for every PDF in DIR; one JSON object per line."""
    fh = open(out, "w") if out else sys.stdout
    for pdf in sorted(directory.glob("*.pdf")):
        try:
            res = suggest_for_pdf(pdf, n, index_dir=index_dir, backend=backend)
        except Exception as e:  # keep going on bad PDFs
            res = {"pdf": str(pdf), "error": repr(e)}
        fh.write(json.dumps(res, default=str) + "\n")
    if out:
        fh.close()
        console.print(f"wrote {out}")


@app.command("verify-offline")
def verify_offline(
    pdf: Path = typer.Argument(..., exists=True),
    index_dir: Path = typer.Option(DEFAULT_INDEX, "--index-dir"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    n: int = typer.Option(5, "--n"),
):
    """Run `suggest` with all socket creation disabled to prove no network is used."""
    import socket

    class _NoNetwork(RuntimeError):
        pass

    def _blocked(*a, **k):
        raise _NoNetwork("network access attempted during offline verification")

    real_socket = socket.socket

    class _BlockedSocket(real_socket):  # subclass so `class X(socket.socket)` in libs still imports
        def __init__(self, *a, **k):
            raise _NoNetwork("network access attempted during offline verification")

    real = (socket.socket, socket.create_connection, socket.getaddrinfo)
    socket.socket = _BlockedSocket  # type: ignore[assignment,misc]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    try:
        res = suggest_for_pdf(pdf, n, index_dir=index_dir, backend=backend)
    except _NoNetwork as e:
        console.print(f"[red]FAIL[/red] {e}")
        raise typer.Exit(2)
    finally:
        socket.socket, socket.create_connection, socket.getaddrinfo = real  # type: ignore[assignment]
    console.print(f"[green]OK[/green] suggest completed with sockets blocked; {len(res['reviewers'])} reviewers returned.")


@app.command("fetch-index")
def fetch_index(
    url: str = typer.Argument(DEFAULT_INDEX_URL, help="URL of an index tarball (.tar.gz); default: the latest core index release"),
    sha256: Optional[str] = typer.Option(None, "--sha256", help="Expected digest; if omitted, tries URL + '.sha256'"),
    index_dir: Path = typer.Option(Path("data"), "--index-dir", help="Unpack here (core tarball contains index/ plus the tables)"),
    collection: Optional[str] = typer.Option(None, "--collection", help="Fetch an add-on collection instead, e.g. stats-ml (unpacks into data/index_<name>)"),
):
    """Download the public reviewer index (about 165 MB), verify its sha256, unpack into --index-dir.

    This is one of only two downloads the tool ever makes; the other is the embedding model
    (about 440 MB from Hugging Face, on first use). No manuscript data is ever uploaded.
    """
    import tarfile
    import tempfile

    import requests

    if collection:
        if url == DEFAULT_INDEX_URL:
            url = f"{INDEX_RELEASES}/index-v1/kindred-index-{collection}-v1.tar.gz"
        index_dir = index_dir / f"index_{collection}" if index_dir.name == "data" else index_dir
    console.print(f"downloading {url} -> {index_dir}")
    index_dir.mkdir(parents=True, exist_ok=True)
    if sha256 is None:
        r = requests.get(url + ".sha256", timeout=30)
        r.raise_for_status()
        sha256 = r.text.split()[0]
    h = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                tmp.write(chunk)
                h.update(chunk)
        tmp_path = Path(tmp.name)
    if h.hexdigest() != sha256.lower():
        tmp_path.unlink(missing_ok=True)
        console.print(f"[red]sha256 mismatch[/red]: got {h.hexdigest()}, expected {sha256}")
        raise typer.Exit(1)
    with tarfile.open(tmp_path) as tf:
        tf.extractall(index_dir, filter="data") if sys.version_info >= (3, 12) else tf.extractall(index_dir)
    tmp_path.unlink(missing_ok=True)
    console.print(f"[green]OK[/green] index unpacked into {index_dir} (sha256 verified)")


@app.command("learn-weights")
def learn_weights(
    l2: float = typer.Option(1.0, "--l2", help="Pull toward the current defaults (higher = smaller change)"),
    save: bool = typer.Option(False, "--save", help="Write the learned weights to ~/.kindred/weights.json"),
):
    """Fit ranking weights to the +/- reviewer ratings recorded in the UI (~/.kindred/feedback.jsonl)."""
    from kindred import learn

    rows = learn.load()
    base = learn.load_params() or {"lam": 0.3, "cite_weight": 0.1, "volume_correction": 0.0, "editor_weight": 0.0,
                                   "seniority_weight": 0.0, "early_career_weight": 0.0, "seed_weight": 0.2}
    res = learn.fit(rows, base, l2=l2)
    console.print(f"{res['n_rated']} ratings on {res['n_manuscripts']} manuscripts -> {res['n_pairs']} preference pairs")
    if res.get("train_acc") is not None:
        console.print(f"pairs ordered correctly: current {res['base_acc']:.0%} -> learned {res['train_acc']:.0%}")
    for k in ("lam", "cite_weight", "volume_correction", "editor_weight", "seniority_weight", "early_career_weight", "seed_weight"):
        console.print(f"  {k:<22} {float(base.get(k, 0) or 0):.3f} -> {float(res['params'].get(k, 0) or 0):.3f}")
    if save:
        learn.save_params(res["params"])
        console.print(f"saved {learn.WEIGHTS}")


@app.command()
def ui(index_dir: Path = typer.Option(DEFAULT_INDEX, "--index-dir"), port: int = typer.Option(8501, "--port"),
       watch: bool = typer.Option(True, "--watch/--no-watch", help="Rerun the app automatically when kindred source files change")):
    """Launch the Streamlit drag-and-drop UI (requires `pip install "orms-kindred[ui]"`)."""
    import os
    import subprocess

    app_path = Path(__file__).with_name("ui_app.py")
    env = {**os.environ, "KINDRED_INDEX_DIR": str(index_dir)}
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port),
           "--browser.gatherUsageStats", "false",
           "--server.runOnSave", "true" if watch else "false",
           # watch the package directory itself (editable install), not just the entry script
           "--server.folderWatchList", str(app_path.parent)]
    subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
    app()
