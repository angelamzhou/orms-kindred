#!/usr/bin/env python3
"""Compare saved per-source pull state against live OpenAlex counts (detects caps/drift).

Usage: python scripts/check_pull.py [--from-year 2014]
Exit code 1 if any source is not done or the live count exceeds what was written;
rerun `pull_journal_works.py --source S --restart` for those sources.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kindred.openalex import OpenAlexClient  # noqa: E402
from kindred.sources import SOURCES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2014)
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    client = OpenAlexClient()
    bad = 0
    print(f"{'source':<12} {'venue':<46} {'written':>8} {'saved':>8} {'live':>8} {'any-loc':>8} status")
    for sid, name in SOURCES.items():
        st_path = Path(args.raw_dir) / f"state_{sid}.json"
        st = json.loads(st_path.read_text()) if st_path.exists() else {}
        yr = f"publication_year:>{args.from_year - 1}"
        live = client.count_works(f"primary_location.source.id:{sid},{yr}")
        anyloc = client.count_works(f"locations.source.id:{sid},{yr}")
        written, saved, done = st.get("n_written", 0), st.get("total", 0), st.get("done", False)
        status = "ok"
        if not done:
            status = "NOT DONE"
        elif live > written:
            status = f"+{live - written} new"
        elif written < saved:
            status = "SHORT"
        if status != "ok":
            bad += 1
        print(f"{sid:<12} {name[:46]:<46} {written:>8} {saved:>8} {live:>8} {anyloc:>8} {status}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
