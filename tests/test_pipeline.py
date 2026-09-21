"""Smoke tests on the synthetic fixture (run: PYTHONPATH=src python3 -m pytest tests -q, or python3 tests/test_pipeline.py)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from kindred.embed import Embedder  # noqa: E402
from kindred.eval import leave_one_out  # noqa: E402
from kindred.index import PaperIndex, build_index  # noqa: E402
from kindred.match import ReviewerMatcher  # noqa: E402

FIX = os.path.join(HERE, "fixtures")


def _data():
    if not os.path.exists(os.path.join(FIX, "papers.parquet")):
        sys.path.insert(0, FIX)
        import make_fixture
        make_fixture.main()
    return pd.read_parquet(os.path.join(FIX, "papers.parquet")), pd.read_parquet(os.path.join(FIX, "authorships.parquet"))


def test_index_roundtrip(tmp_path=None):
    papers, auth = _data()
    emb = Embedder("tfidf")
    X = emb.encode(papers.title, papers.abstract)
    assert X.dtype == np.float32 and np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4)
    idx = build_index(X, papers.openalex_work_id)
    d = str(tmp_path or os.path.join(FIX, "_tmp_index"))
    idx.save(d)
    idx2 = PaperIndex.load(d)
    assert idx2.paper_ids == idx.paper_ids
    top = idx2.top_k(X[0], k=5)
    assert top[0][0] == idx.paper_ids[0] and abs(top[0][1] - 1) < 1e-2
    return idx2, papers, auth, X


def test_match_conflicts():
    idx, papers, auth, X = test_index_roundtrip()
    m = ReviewerMatcher(idx, auth, papers, recency_half_life=8)
    pid = idx.paper_ids[0]
    true_auth = auth[auth.work_id == pid].author_id.tolist()
    ranked = m.rank(X[0], top_n=10, exclude_papers=[pid], manuscript_author_ids=true_auth)
    ids = {c.author_id for c in ranked}
    assert not ids & set(true_auth)
    for a in true_auth:
        assert not ids & m.coauthors[a], "co-authors should be filtered"
    inst = auth.institution_id.iloc[0]
    ranked2 = m.rank(X[0], top_n=50, excluded_institution_ids=[inst])
    assert all(inst not in m.author_inst[c.author_id] for c in ranked2)
    assert all(len(c.evidence) <= 3 for c in ranked)


def test_eval_runs():
    idx, papers, auth, X = test_index_roundtrip()
    res = leave_one_out(idx, auth, papers, n=20)
    assert 0 <= res["MRR"] <= 1 and res["n_evaluated"] > 0


if __name__ == "__main__":
    test_match_conflicts()
    test_eval_runs()
    print("ok")
