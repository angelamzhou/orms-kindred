"""End-to-end smoke test: build a tiny TF-IDF index in a temp dir, then run the CLI on the fixture PDF.

Run with:  python -m pytest tests/  (or  python tests/test_cli_e2e.py  to print the table)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

FIX = Path(__file__).parent / "fixtures" / "sample_or_paper.pdf"

PAPERS = [
    ("W1", "Robust vehicle routing with budgeted demand uncertainty", "We propose a two-stage robust optimization model for the capacitated vehicle routing problem with uncertain demands and solve it with column-and-constraint generation.", 2021),
    ("W2", "Branch-and-cut for the robust capacitated vehicle routing problem", "Exact branch-and-cut algorithm for vehicle routing under demand uncertainty using robust knapsack cuts.", 2019),
    ("W3", "Adjustable robust optimization: a survey", "We survey two-stage and multistage adjustable robust optimization with recourse, including column-and-constraint generation methods.", 2020),
    ("W4", "Dynamic pricing for ride-hailing platforms", "We study spatial pricing and matching in two-sided ride-hailing markets with strategic drivers.", 2022),
    ("W5", "Queueing models of hospital emergency departments", "We analyse patient flow in emergency departments using multi-server queues with abandonment.", 2018),
    ("W6", "Inventory control with censored demand observations", "Learning algorithms for newsvendor problems with censored demand and regret bounds.", 2021),
]
AUTHORSHIPS = [
    ("W1", "A1", "Alice Robustova", "I1", "Univ. A"), ("W1", "A2", "Bob Routing", "I2", "Univ. B"),
    ("W2", "A2", "Bob Routing", "I2", "Univ. B"), ("W2", "A3", "Carla Cuts", "I3", "Univ. C"),
    ("W3", "A1", "Alice Robustova", "I1", "Univ. A"), ("W3", "A4", "Dan Survey", "I4", "Univ. D"),
    ("W4", "A5", "Eve Pricing", "I5", "Univ. E"), ("W5", "A6", "Frank Queue", "I6", "Univ. F"),
    ("W6", "A7", "Grace Newsvendor", "I7", "Univ. G"),
]


def build_tiny_index(index_dir: Path) -> None:
    from ormatch.embed import Embedder
    from ormatch.index import build_index

    index_dir.mkdir(parents=True, exist_ok=True)
    papers = pd.DataFrame(PAPERS, columns=["openalex_work_id", "title", "abstract", "year"])
    auth = pd.DataFrame(AUTHORSHIPS, columns=["work_id", "author_id", "author_name", "institution_id", "institution_name"])
    emb = Embedder("tfidf", tfidf_dim=8).fit(papers["title"], papers["abstract"])
    X = emb.encode(papers["title"], papers["abstract"])
    build_index(X, papers["openalex_work_id"].tolist(), {"backend": "tfidf"}).save(str(index_dir))
    emb.save(str(index_dir / "tfidf.pkl"))
    papers.to_parquet(index_dir / "papers.parquet")
    auth.to_parquet(index_dir / "authorships.parquet")


def test_pdf_extraction():
    from ormatch.pdf import extract_text, guess_title_abstract

    title, abstract = guess_title_abstract(extract_text(FIX))
    assert title.startswith("Robust Vehicle Routing under Demand Uncertainty")
    assert abstract.startswith("We study the capacitated vehicle routing problem")
    assert "Keywords" not in abstract and "Introduction" not in abstract


def test_suggest_and_verify_offline(tmp_path: Path):
    idx = tmp_path / "index"
    build_tiny_index(idx)
    out = subprocess.run(
        [sys.executable, "-m", "ormatch.cli", "suggest", str(FIX), "--backend", "tfidf", "--index-dir", str(idx),
         "--json", "--n", "5", "--exclude-institution", "I2"],
        capture_output=True, text=True, check=True,
    )
    res = json.loads(out.stdout)
    names = [r["author_name"] for r in res["reviewers"]]
    assert "Bob Routing" not in names  # excluded institution
    assert names[0] == "Alice Robustova"  # robust VRP author ranks first
    assert res["reviewers"][0]["evidence"][0][1].startswith("Robust vehicle routing")

    off = subprocess.run(
        [sys.executable, "-m", "ormatch.cli", "verify-offline", str(FIX), "--backend", "tfidf", "--index-dir", str(idx)],
        capture_output=True, text=True,
    )
    assert off.returncode == 0, off.stderr
    assert "OK" in off.stderr


if __name__ == "__main__":
    import tempfile

    d = Path(tempfile.mkdtemp()) / "index"
    build_tiny_index(d)
    subprocess.run([sys.executable, "-m", "ormatch.cli", "suggest", str(FIX), "--backend", "tfidf", "--index-dir", str(d), "--n", "5"])
    subprocess.run([sys.executable, "-m", "ormatch.cli", "verify-offline", str(FIX), "--backend", "tfidf", "--index-dir", str(d)])
