#!/usr/bin/env python3
"""Normalize raw OpenAlex works JSONL into papers.parquet + authorships.parquet.

Usage:
  python scripts/normalize_works.py [--raw-dir data/raw] [--out-dir data]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ormatch.openalex import reconstruct_abstract  # noqa: E402
from ormatch.sources import source_name  # noqa: E402

log = logging.getLogger("normalize")


def _short_id(url: str | None) -> str | None:
    if not url:
        return None
    return url.rsplit("/", 1)[-1]


def _clean_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.replace("https://doi.org/", "").lower()


def iter_works(raw_dir: Path):
    for path in sorted(raw_dir.glob("works_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping malformed line in %s", path.name)


def normalize(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    papers: list[dict] = []
    auths: list[dict] = []
    for w in iter_works(raw_dir):
        wid = _short_id(w.get("id"))
        if not wid:
            continue
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        sid = _short_id(src.get("id"))
        oa = w.get("open_access") or {}
        best = w.get("best_oa_location") or {}
        pdf_url = best.get("pdf_url") or oa.get("oa_url") or None
        papers.append(
            {
                "openalex_work_id": wid,
                "doi": _clean_doi(w.get("doi")),
                "title": w.get("title"),
                "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                "year": w.get("publication_year"),
                "source_id": sid,
                "venue": src.get("display_name") or source_name(sid or ""),
                "oa_pdf_url": pdf_url,
                "cited_by_count": w.get("cited_by_count"),
            }
        )
        for pos, a in enumerate(w.get("authorships") or []):
            author = a.get("author") or {}
            insts = a.get("institutions") or []
            inst = insts[0] if insts else {}
            auths.append(
                {
                    "work_id": wid,
                    "author_id": _short_id(author.get("id")),
                    "author_name": author.get("display_name"),
                    "position": pos,
                    "author_position": a.get("author_position"),
                    "institution_id": _short_id(inst.get("id")),
                    "institution_name": inst.get("display_name"),
                }
            )

    papers_df = pd.DataFrame(papers, columns=["openalex_work_id", "doi", "title", "abstract", "year", "source_id", "venue", "oa_pdf_url", "cited_by_count"])
    if len(papers_df):
        papers_df = papers_df.drop_duplicates("openalex_work_id", keep="last")
        papers_df["year"] = papers_df["year"].astype("Int64")
        papers_df["cited_by_count"] = papers_df["cited_by_count"].astype("Int64")
    auth_df = pd.DataFrame(auths, columns=["work_id", "author_id", "author_name", "position", "author_position", "institution_id", "institution_name"])
    if len(auth_df):
        auth_df = auth_df[auth_df["work_id"].isin(papers_df["openalex_work_id"])].drop_duplicates(["work_id", "position"], keep="last")
    return papers_df, auth_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    papers, auths = normalize(raw_dir)
    papers.to_parquet(out_dir / "papers.parquet", index=False)
    auths.to_parquet(out_dir / "authorships.parquet", index=False)

    n = len(papers)
    has_abs = papers["abstract"].fillna("").str.strip().ne("").sum() if n else 0
    has_pdf = papers["oa_pdf_url"].fillna("").ne("").sum() if n else 0
    log.info("papers: %d rows -> %s", n, out_dir / "papers.parquet")
    log.info("authorships: %d rows -> %s", len(auths), out_dir / "authorships.parquet")
    if n:
        log.info("abstract coverage: %d/%d = %.1f%%", has_abs, n, 100 * has_abs / n)
        log.info("OA pdf/url coverage: %d/%d = %.1f%%", has_pdf, n, 100 * has_pdf / n)
        log.info("by source:\n%s", papers.groupby(["source_id", "venue"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
