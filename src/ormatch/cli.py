"""ORMatch command-line interface (typer)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="ORMatch: local reviewer matching for OR/MS/OM papers.", no_args_is_help=True)
console = Console(stderr=True)

DEFAULT_INDEX = Path("data/index")


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


def suggest_for_pdf(pdf: Path, n: int = 20, exclude_institutions: set[str] | None = None,
                    exclude_authors: set[str] | None = None, index_dir: Path = DEFAULT_INDEX,
                    backend: str = "specter2", k_hits: int = 200) -> dict:
    """Full pipeline: PDF -> title/abstract -> embed -> search -> rank reviewers."""
    from ormatch.embed import Embedder
    from ormatch.index import PaperIndex
    from ormatch.match import ReviewerMatcher
    from ormatch.pdf import pdf_to_query

    q = pdf_to_query(pdf)
    idx = PaperIndex.load(str(index_dir))
    embedder = Embedder(backend)
    if embedder.backend == "tfidf":  # fitted vectorizer is stored beside the index
        tfidf_path = index_dir / "tfidf.pkl"
        if not tfidf_path.exists():
            raise FileNotFoundError(f"tfidf backend needs {tfidf_path} (built with the index)")
        embedder.load(str(tfidf_path))
    qvec = embedder.encode([q["title"]], [q["abstract"]])[0]
    authorships, papers = _load_tables(index_dir)
    matcher = ReviewerMatcher(idx, authorships, papers)
    cands = matcher.rank(
        qvec, top_n=n,
        manuscript_author_ids=exclude_authors or None,
        excluded_institution_ids=exclude_institutions or None,
    )
    titles = {}
    if "openalex_work_id" in papers.columns and "title" in papers.columns:
        titles = dict(zip(papers["openalex_work_id"].astype(str), papers["title"].astype(str)))
    reviewers = []
    for c in cands:
        reviewers.append({
            "author_id": c.author_id,
            "author_name": c.author_name,
            "institution": "; ".join(c.institutions) if c.institutions else None,
            "score": float(c.score),
            "n_papers": c.n_papers,
            "evidence": [(pid, titles.get(pid, pid), float(s)) for pid, s in c.evidence],
        })
    return {"pdf": str(pdf), "title": q["title"], "abstract": q["abstract"], "reviewers": reviewers}


def _print_result(res: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(res, indent=2, default=str))
        return
    console.print(f"[bold]Title:[/bold] {res['title']}")
    console.print(f"[bold]Abstract:[/bold] {res['abstract'][:300]}{'...' if len(res['abstract']) > 300 else ''}\n")
    t = Table(title=f"Suggested reviewers ({len(res['reviewers'])})")
    for col in ("#", "Reviewer", "Institution", "Score", "Evidence (top paper)"):
        t.add_column(col)
    for i, r in enumerate(res["reviewers"], 1):
        ev = r.get("evidence") or []
        top = f"{ev[0][1][:70]} ({ev[0][2]:.2f})" if ev else ""
        t.add_row(str(i), str(r.get("author_name")), str(r.get("institution") or ""), f"{r['score']:.3f}", top)
    Console().print(t)


# ----------------------------------------------------------------------- commands
@app.command()
def suggest(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Manuscript PDF"),
    n: int = typer.Option(20, "--n", help="Number of reviewers"),
    exclude_institution: list[str] = typer.Option([], "--exclude-institution", help="OpenAlex institution ID (repeatable)"),
    exclude_author: list[str] = typer.Option([], "--exclude-author", help="OpenAlex author ID (repeatable)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON on stdout"),
    index_dir: Path = typer.Option(DEFAULT_INDEX, "--index-dir"),
    backend: str = typer.Option("specter2", "--backend", help="specter2 | scincl | tfidf"),
):
    """Suggest reviewers for one PDF. Runs entirely offline against the local index."""
    res = suggest_for_pdf(pdf, n, set(exclude_institution), set(exclude_author), index_dir, backend)
    _print_result(res, json_out)


@app.command()
def batch(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    n: int = typer.Option(20, "--n"),
    index_dir: Path = typer.Option(DEFAULT_INDEX, "--index-dir"),
    backend: str = typer.Option("specter2", "--backend"),
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
    backend: str = typer.Option("specter2", "--backend"),
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
    url: str = typer.Argument(..., help="URL of an index tarball (.tar.gz)"),
    sha256: Optional[str] = typer.Option(None, "--sha256", help="Expected digest; if omitted, tries URL + '.sha256'"),
    index_dir: Path = typer.Option(DEFAULT_INDEX, "--index-dir"),
):
    """Download the public reviewer index, verify sha256, unpack into --index-dir."""
    import tarfile
    import tempfile

    import requests

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


@app.command()
def ui(index_dir: Path = typer.Option(DEFAULT_INDEX, "--index-dir"), port: int = typer.Option(8501, "--port")):
    """Launch the Streamlit drag-and-drop UI (requires `pip install ormatch[ui]`)."""
    import os
    import subprocess

    app_path = Path(__file__).with_name("ui_app.py")
    env = {**os.environ, "ORMATCH_INDEX_DIR": str(index_dir)}
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port),
                    "--browser.gatherUsageStats", "false"], env=env, check=False)


if __name__ == "__main__":
    app()
