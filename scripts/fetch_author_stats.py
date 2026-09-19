#!/usr/bin/env python3
"""Fetch per-author statistics from OpenAlex (works_count, cited_by_count, h_index, first/last
publication year) for every author in the index tables -> data/authors.parquet.

Used as a seniority proxy (`--seniority-weight`) and shown in the UI. 50 ids per request; ~40k
authors take a few minutes. Resumable: already-fetched ids are skipped.
Usage: python scripts/fetch_author_stats.py [--authorships data/authorships.parquet ...]
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ormatch.openalex import OpenAlexClient  # noqa: E402

log = logging.getLogger("authors")
OUT = ROOT / "data" / "authors.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorships", nargs="*", default=None, help="authorships.parquet files (default: core + every data/index_*/)")
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = args.authorships or [str(ROOT / "data" / "authorships.parquet")] + glob.glob(str(ROOT / "data" / "index_*" / "authorships.parquet"))
    ids = sorted({str(a) for f in files for a in pd.read_parquet(f)["author_id"].dropna().unique()})
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["author_id"])
    todo = [i for i in ids if i not in set(done["author_id"])]
    log.info("%d authors in %d tables; %d already fetched; %d to fetch", len(ids), len(files), len(done), len(todo))
    client = OpenAlexClient()
    rows = done.to_dict("records")
    t0 = time.time()
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        try:
            data = client.get("authors", filter="openalex:" + "|".join(chunk), per_page=50,
                              select="id,display_name,works_count,cited_by_count,summary_stats,last_known_institutions")
        except Exception as e:  # noqa: BLE001
            log.warning("chunk %d failed: %s", i, e)
            continue
        for a in data.get("results", []):
            ss = a.get("summary_stats") or {}
            inst = (a.get("last_known_institutions") or [{}])[0] or {}
            rows.append({"author_id": a["id"].rsplit("/", 1)[-1], "display_name": a.get("display_name"),
                         "works_count": a.get("works_count"), "cited_by_count": a.get("cited_by_count"),
                         "h_index": ss.get("h_index"), "i10_index": ss.get("i10_index"),
                         "last_institution": inst.get("display_name")})
        if (i // 50) % 100 == 0:
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            log.info("%d/%d fetched (%.0fs)", min(i + 50, len(todo)), len(todo), time.time() - t0)
        time.sleep(args.sleep)
    pd.DataFrame(rows).to_parquet(OUT, index=False)
    log.info("wrote %d authors -> %s", len(rows), OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
