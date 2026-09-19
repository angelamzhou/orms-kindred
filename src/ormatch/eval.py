"""Leave-one-out evaluation of reviewer ranking.

For each sampled paper: remove it from the index (its similarity is masked so
none of its authors get credit from it), treat its embedding as the manuscript,
rank reviewers, and check where the paper's true authors land.

Metrics: MRR (rank of the first true author), recall@10, recall@20 (fraction of
true authors in the top 10/20), averaged over papers. Authors whose *only*
indexed paper is the held-out one are unreachable; they are dropped from the
targets and the count is reported as ``unreachable_authors``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .index import PaperIndex
from .match import ReviewerMatcher


def leave_one_out(
    index: PaperIndex,
    authorships: pd.DataFrame,
    papers: Optional[pd.DataFrame] = None,
    n: int = 50,
    seed: int = 0,
    top_n: int = 20,
    **matcher_kwargs,
) -> Dict:
    matcher = ReviewerMatcher(index, authorships, papers, **matcher_kwargs)
    rng = np.random.default_rng(seed)
    eligible = [p for p in index.paper_ids if p in matcher.paper_authors]
    sample = rng.choice(eligible, size=min(n, len(eligible)), replace=False)

    rr, r10, r20, used, unreachable = [], [], [], 0, 0
    for pid in sample:
        truth = set(matcher.paper_authors[pid])
        reachable = {a for a in truth if len(matcher.author_rows[a]) > 1}
        unreachable += len(truth) - len(reachable)
        if not reachable:
            continue
        used += 1
        q = index.embeddings[index.position(pid)]
        ranked = matcher.rank(q, top_n=top_n, exclude_papers=[pid], exclude_coauthors=False)
        ids = [c.author_id for c in ranked]
        first = next((i for i, a in enumerate(ids) if a in reachable), None)
        rr.append(1.0 / (first + 1) if first is not None else 0.0)
        r10.append(len(reachable & set(ids[:10])) / len(reachable))
        r20.append(len(reachable & set(ids[:20])) / len(reachable))

    return {
        "n_sampled": int(len(sample)),
        "n_evaluated": used,
        "unreachable_authors": unreachable,
        "n_index_papers": len(index),
        "n_authors": len(matcher.author_rows),
        "MRR": float(np.mean(rr)) if rr else float("nan"),
        "recall@10": float(np.mean(r10)) if r10 else float("nan"),
        "recall@20": float(np.mean(r20)) if r20 else float("nan"),
        "params": {"lam": matcher.lam, "k": matcher.k, "recency_half_life": matcher.recency_half_life},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Leave-one-out reviewer ranking eval")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--authorships", default="data/authorships.parquet")
    ap.add_argument("--papers", default="data/papers.parquet")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--half-life", type=float, default=None)
    args = ap.parse_args(argv)

    index = PaperIndex.load(args.index)
    auth = pd.read_parquet(args.authorships)
    papers = pd.read_parquet(args.papers) if os.path.exists(args.papers) else None
    res = leave_one_out(index, auth, papers, n=args.n, seed=args.seed, lam=args.lam, k=args.k,
                        recency_half_life=args.half_life)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
