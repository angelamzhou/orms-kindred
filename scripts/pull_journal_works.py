#!/usr/bin/env python3
"""Stream works for one OpenAlex source into JSONL, with a resumable cursor.

Usage:
  python scripts/pull_journal_works.py --source S125775545 --from-year 2014 \
      --out data/raw/works_S125775545.jsonl

State is saved to data/raw/state_<source>.json after every page; rerunning
resumes from the stored cursor. Once finished, `done: true` is recorded and a
rerun exits immediately (use --restart to start over).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ormatch.openalex import WORK_FIELDS, OpenAlexClient  # noqa: E402
from ormatch.sources import resolve_source, source_name  # noqa: E402


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="OpenAlex source ID (e.g. S125775545) or alias (OR, MSOM, ...)")
    ap.add_argument("--from-year", type=int, default=2014)
    ap.add_argument("--to-year", type=int, default=None)
    ap.add_argument("--out", default=None, help="default data/raw/works_<source>.jsonl")
    ap.add_argument("--max-pages", type=int, default=None, help="stop after N pages (for testing)")
    ap.add_argument("--restart", action="store_true", help="ignore saved state and overwrite output")
    ap.add_argument("--sleep", type=float, default=0.1, help="pause between pages (s)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pull")

    source = resolve_source(args.source)
    out = Path(args.out) if args.out else ROOT / "data" / "raw" / f"works_{source}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.parent / f"state_{source}.json"

    year_filter = f"publication_year:>{args.from_year - 1}"
    if args.to_year:
        year_filter = f"publication_year:{args.from_year}-{args.to_year}"
    flt = f"primary_location.source.id:{source},{year_filter}"

    state = {} if args.restart else load_state(state_path)
    if args.restart:
        out.unlink(missing_ok=True)
    if state.get("done") and state.get("filter") == flt:
        log.info("%s already complete (%d works); nothing to do", source, state.get("n_written", 0))
        return 0
    if state and state.get("filter") != flt:
        log.warning("saved state has a different filter; restarting from scratch")
        state = {}
        out.unlink(missing_ok=True)

    cursor = state.get("next_cursor", "*")
    n_written = int(state.get("n_written", 0))
    n_pages = int(state.get("n_pages", 0))

    client = OpenAlexClient()
    if not client.api_key:
        log.warning("OPENALEX_API_KEY not set; running unauthenticated")

    total = client.count_works(flt)
    log.info("%s (%s): %d works match, resuming at page %d with %d already written", source, source_name(source), total, n_pages + 1, n_written)

    state.update({"source": source, "filter": flt, "total": total, "next_cursor": cursor, "n_written": n_written, "n_pages": n_pages})
    save_state(state_path, state)

    pages_this_run = 0
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for results, next_cursor in client.works(flt, select=WORK_FIELDS, cursor=cursor):
            for w in results:
                fh.write(json.dumps(w, ensure_ascii=False) + "\n")
            fh.flush()
            n_written += len(results)
            n_pages += 1
            pages_this_run += 1
            state.update({"next_cursor": next_cursor, "n_written": n_written, "n_pages": n_pages, "done": next_cursor is None, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            save_state(state_path, state)
            log.info("page %d: +%d (total %d/%d) calls=%d", n_pages, len(results), n_written, total, client.calls)
            if next_cursor is None:
                break
            if args.max_pages and pages_this_run >= args.max_pages:
                log.info("max-pages reached; state saved, rerun to resume")
                break
            time.sleep(args.sleep)

    log.info("finished: %d works written to %s in %.1fs using %d API calls (done=%s)", n_written, out, time.time() - t0, client.calls, state.get("done"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
