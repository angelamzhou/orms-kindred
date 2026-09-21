#!/usr/bin/env python3
"""Build the Kindred paper index: papers.parquet -> data/index/{embeddings.npy,paper_ids.json,meta.json}.

Usage: python scripts/build_index.py [--papers data/papers.parquet] [--out data/index] [--backend auto|specter2|scincl|tfidf]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from kindred.embed import Embedder  # noqa: E402
from kindred.index import build_index  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="data/papers.parquet")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto (cuda if available)")
    ap.add_argument("--no-fallback", action="store_true", help="fail instead of falling back specter2 -> scincl")
    ap.add_argument("--limit", type=int, default=None, help="only embed first N papers (debug)")
    ap.add_argument("--collection", default="core", help="name recorded in meta.json (core, applied-or, ...)")
    args = ap.parse_args(argv)

    papers = pd.read_parquet(args.papers)
    papers = papers.dropna(subset=["title"]).drop_duplicates("openalex_work_id")
    if args.limit:
        papers = papers.head(args.limit)
    ids = papers["openalex_work_id"].astype(str).tolist()
    print(f"{len(ids)} papers; loading embedder ({args.backend})...", flush=True)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"device={device}", flush=True)

    t0 = time.time()
    try:
        emb = Embedder(args.backend, device=device, batch_size=args.batch_size)
    except RuntimeError as e:
        if args.backend != "specter2" or args.no_fallback:
            raise
        print(f"specter2 unavailable ({e}); falling back to scincl", flush=True)
        emb = Embedder("scincl", device=device, batch_size=args.batch_size)
    t_load = time.time() - t0
    print(f"backend={emb.backend} loaded in {t_load:.1f}s", flush=True)

    t0 = time.time()
    X = emb.encode(papers["title"].tolist(), papers["abstract"].tolist(), show_progress=True)
    t_enc = time.time() - t0
    per100 = 100 * t_enc / max(1, len(ids))
    print(f"encoded {len(ids)} in {t_enc:.1f}s ({per100:.1f}s / 100 papers), dim={X.shape[1]}", flush=True)

    meta = {"backend": emb.backend, "dim": int(X.shape[1]), "n": len(ids), "collection": args.collection,
            "encode_seconds": round(t_enc, 2), "seconds_per_100": round(per100, 2),
            "source": os.path.abspath(args.papers)}
    idx = build_index(X, ids, meta)
    idx.save(args.out)
    if emb.backend == "tfidf":
        emb.save(os.path.join(args.out, "tfidf.pkl"))
    print(f"saved index to {args.out}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
