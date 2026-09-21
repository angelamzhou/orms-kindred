#!/usr/bin/env python3
"""Backfill missing abstracts (mostly Elsevier venues) from Semantic Scholar, Elsevier, Crossref.

Usage:
  python scripts/backfill_abstracts.py [--limit N] [--no-crossref] [--papers data/papers.parquet]

Reads papers.parquet, finds rows with a DOI but no abstract, asks the Semantic Scholar
Graph API in batches of 500 DOIs (set S2_API_KEY in .env for a higher rate limit), and
for anything still missing asks Crossref one DOI at a time (skipping Elsevier 10.1016/*, which has none). Results are appended to
data/raw/abstracts_backfill.jsonl (resumable: DOIs already present are skipped) and
picked up by scripts/normalize_works.py on its next run.

Elsevier (EJOR, ORL) deposits abstracts neither in OpenAlex nor Crossref. If ELSEVIER_API_KEY
is set in .env (free key from https://dev.elsevier.com, used from the USC network or with an
ELSEVIER_INSTTOKEN), the remaining 10.1016/* DOIs are fetched from the ScienceDirect Article
Retrieval API (view=META_ABS, ~10k requests/week quota).
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

log = logging.getLogger("backfill")
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
CROSSREF = "https://api.crossref.org/works/"
ELSEVIER = "https://api.elsevier.com/content/article/doi/"
OUT = ROOT / "data" / "raw" / "abstracts_backfill.jsonl"
TAG_RE = re.compile(r"<[^>]+>")


def _clean(s: str | None) -> str | None:
    if not s:
        return None
    s = html.unescape(TAG_RE.sub(" ", s))
    s = re.sub(r"^\s*abstract\s*[:.]?\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _get_with_retry(session, method, url, tries=6, **kw):
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            r = session.request(method, url, timeout=60, **kw)
        except requests.RequestException as e:
            log.warning("%s %s: %s (attempt %d)", method, url[:60], type(e).__name__, attempt)
        else:
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code not in (429, 500, 502, 503, 504):
                log.warning("HTTP %s for %s: %s", r.status_code, url[:60], r.text[:200])
                return None
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra and ra.isdigit() else delay
            log.warning("HTTP %s (attempt %d); sleeping %.0fs", r.status_code, attempt, wait)
            time.sleep(wait)
        delay = min(delay * 2, 60)
    return None


def s2_batch(session, dois: list[str]) -> dict[str, str]:
    r = _get_with_retry(session, "POST", S2_BATCH, params={"fields": "abstract,externalIds"},
                        json={"ids": [f"DOI:{d}" for d in dois]})
    out: dict[str, str] = {}
    if r is None:
        return out
    for doi, item in zip(dois, r.json()):
        if item and item.get("abstract"):
            out[doi] = _clean(item["abstract"])
    return out


def crossref_one(session, doi: str) -> str | None:
    r = _get_with_retry(session, "GET", CROSSREF + requests.utils.quote(doi, safe=""), tries=3)
    if r is None:
        return None
    return _clean((r.json().get("message") or {}).get("abstract"))


def elsevier_one(session, doi: str, key: str, insttoken: str | None) -> str | None:
    headers = {"X-ELS-APIKey": key, "Accept": "application/json"}
    if insttoken:
        headers["X-ELS-Insttoken"] = insttoken
    r = _get_with_retry(session, "GET", ELSEVIER + requests.utils.quote(doi, safe=""), tries=3,
                        params={"view": "META_ABS"}, headers=headers)
    if r is None:
        return None
    core = ((r.json().get("full-text-retrieval-response") or {}).get("coredata") or {})
    return _clean(core.get("dc:description"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(ROOT / "data" / "papers.parquet"))
    ap.add_argument("--limit", type=int, default=None, help="only try the first N missing DOIs")
    ap.add_argument("--no-crossref", action="store_true")
    ap.add_argument("--crossref-skip-prefix", default="10.1016",
                    help="comma-separated DOI prefixes to skip in the Crossref pass (Elsevier deposits no abstracts)")
    ap.add_argument("--s2-sleep", type=float, default=1.2, help="pause between S2 batch calls")
    ap.add_argument("--crossref-sleep", type=float, default=0.1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    papers = pd.read_parquet(args.papers)
    missing = papers[papers["abstract"].fillna("").str.strip().eq("") & papers["doi"].notna()]
    found: dict[str, str] = {}
    tried: dict[str, set] = collections.defaultdict(set)  # doi -> sources that were asked
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            rec = json.loads(line)
            tried[rec["doi"]].add(rec.get("source") or "crossref")  # legacy null records = Crossref miss
            if rec.get("abstract"):
                found[rec["doi"]] = rec["abstract"]
    missing = missing[~missing["doi"].isin(found)]
    todo = missing[~missing["doi"].isin(tried)]  # never asked anywhere yet -> S2 first
    if args.limit:
        todo = todo.head(args.limit)
    els_key, els_tok = os.environ.get("ELSEVIER_API_KEY"), os.environ.get("ELSEVIER_INSTTOKEN")
    els_todo = [d for d in missing["doi"] if d.startswith("10.1016/") and "elsevier" not in tried[d]] if els_key else []
    if args.limit:
        els_todo = els_todo[: args.limit]
    log.info("%d papers lack an abstract; %d have a DOI and no backfilled abstract; %d untried (S2 first); %d Elsevier DOIs for the Elsevier API (%s)",
             papers["abstract"].fillna("").str.strip().eq("").sum(), len(missing), len(todo), len(els_todo),
             "key set" if els_key else "no ELSEVIER_API_KEY")
    if todo.empty and not els_todo:
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = "kindred-backfill/0.1 (https://github.com/; mailto:kindred@localhost)"
    s2_key = os.environ.get("S2_API_KEY")
    if s2_key:
        session.headers["x-api-key"] = s2_key

    id_by_doi = dict(zip(missing["doi"], missing["openalex_work_id"]))
    found_s2 = found_cr = found_els = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fh:
        def emit(doi, abstract, source):
            fh.write(json.dumps({"openalex_work_id": id_by_doi[doi], "doi": doi, "abstract": abstract, "source": source if abstract else (source or None)}, ensure_ascii=False) + "\n")
            fh.flush()

        dois = todo["doi"].tolist()
        still: list[str] = []
        for i in range(0, len(dois), 500):
            chunk = dois[i:i + 500]
            hits = s2_batch(session, chunk)
            found_s2 += len(hits)
            for d in chunk:
                if d in hits:
                    emit(d, hits[d], "s2")
                else:
                    still.append(d)
            log.info("S2 batch %d/%d: %d/%d found (cum %d)", i // 500 + 1, (len(dois) + 499) // 500, len(hits), len(chunk), found_s2)
            time.sleep(args.s2_sleep)

        if not args.no_crossref:
            skip = tuple(x.strip() + "/" for x in args.crossref_skip_prefix.split(",") if x.strip())
            skipped = [d for d in still if d.startswith(skip)]
            still = [d for d in still if not d.startswith(skip)]
            log.info("Crossref fallback for %d DOIs (%d skipped by prefix %s)", len(still), len(skipped), skip)
            for d in skipped:
                emit(d, None, None)  # record as tried so reruns do not retry them
            for j, d in enumerate(still, 1):
                a = crossref_one(session, d)
                if a:
                    found_cr += 1
                emit(d, a, "crossref" if a else None)
                if j % 200 == 0:
                    log.info("crossref %d/%d: %d found", j, len(still), found_cr)
                time.sleep(args.crossref_sleep)
        # With --no-crossref the S2 misses are deliberately not recorded, so a later run
        # with Crossref enabled will still try them.

        if els_todo:
            log.info("Elsevier Article Retrieval (META_ABS) for %d DOIs", len(els_todo))
            for j, d in enumerate(els_todo, 1):
                a = elsevier_one(session, d, els_key, els_tok)
                if a:
                    found_els += 1
                emit(d, a, "elsevier")
                if j % 200 == 0:
                    log.info("elsevier %d/%d: %d found", j, len(els_todo), found_els)
                time.sleep(args.crossref_sleep)

    log.info("done: S2 %d, Crossref %d, Elsevier %d; %d DOIs still lack an abstract",
             found_s2, found_cr, found_els, len(missing) - found_s2 - found_cr - found_els)
    log.info("rerun scripts/normalize_works.py to merge %s into papers.parquet", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
