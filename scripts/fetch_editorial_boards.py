#!/usr/bin/env python3
"""Editorial-board membership over time from Internet Archive snapshots -> data/editors.csv.

Publisher board pages (INFORMS/Atypon, Elsevier, Wiley, SIAM) refuse scripted access. The
Wayback Machine keeps yearly snapshots and its APIs are public, so we read those instead:
one snapshot per year per journal (availability API, gentle pacing), parse the visible text
for role headings (Editor-in-Chief, Area/Department/Associate Editor, ...) followed by names,
and write one row per (journal, role, name) with the first and last year seen and the number
of snapshots. `--seniority`-style penalties then use current roles (last_year >= this year - 1)
and the total number of (journal, role) rows as the "many editorial positions" proxy.

Usage: python scripts/fetch_editorial_boards.py [--years 2016-2026] [--journals opre,mnsc,...]
Resumable: existing (journal, year) snapshots in data/editorial_snapshots/ are not re-fetched.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ormatch.match import normalize_name  # noqa: E402

log = logging.getLogger("boards")
SNAP = ROOT / "data" / "editorial_snapshots"
OUT = ROOT / "data" / "editors.csv"
UA = "ORMatch-boards/0.1 (academic; reads Internet Archive snapshots only)"

BOARDS = {  # journal key -> (display name, live URL that the archive indexes)
    "opre": ("Operations Research", "https://pubsonline.informs.org/page/opre/editorial-board"),
    "mnsc": ("Management Science", "https://pubsonline.informs.org/page/mnsc/editorial-board"),
    "msom": ("Manufacturing & Service Operations Management", "https://pubsonline.informs.org/page/msom/editorial-board"),
    "moor": ("Mathematics of Operations Research", "https://pubsonline.informs.org/page/moor/editorial-board"),
    "trsc": ("Transportation Science", "https://pubsonline.informs.org/page/trsc/editorial-board"),
    "ijoc": ("INFORMS Journal on Computing", "https://pubsonline.informs.org/page/ijoc/editorial-board"),
    "stsy": ("Stochastic Systems", "https://pubsonline.informs.org/page/stsy/editorial-board"),
    "ejor": ("European Journal of Operational Research", "https://www.sciencedirect.com/journal/european-journal-of-operational-research/about/editorial-board"),
    "orl": ("Operations Research Letters", "https://www.sciencedirect.com/journal/operations-research-letters/about/editorial-board"),
    "cor": ("Computers & Operations Research", "https://www.sciencedirect.com/journal/computers-and-operations-research/about/editorial-board"),
    "omega": ("Omega", "https://www.sciencedirect.com/journal/omega/about/editorial-board"),
    "poms": ("Production and Operations Management", "https://onlinelibrary.wiley.com/page/journal/19375956/homepage/editorialboard.html"),
    "nrl": ("Naval Research Logistics", "https://onlinelibrary.wiley.com/page/journal/15206750/homepage/editorialboard.html"),
    "mp": ("Mathematical Programming", "https://link.springer.com/journal/10107/editorial-board"),
    "siopt": ("SIAM Journal on Optimization", "https://www.siam.org/publications/siam-journals/siam-journal-on-optimization/editorial-board/"),
}
ROLE_RE = re.compile(r"^(editor[- ]in[- ]chief|co[- ]?editors?(?:[- ]in[- ]chief)?|area editors?|department editors?|"
                     r"associate editors?|senior editors?|advisory editors?|editorial board|editors?)\b[:\s]*$", re.I)
NAME_RE = re.compile(r"^(?:[A-Z][\w'’\-\.]+\s+){1,3}[A-Z][\w'’\-]+(?:,\s*(?:Jr|Sr|II|III))?\.?$")
STOP_RE = re.compile(r"university|institute|school|college|department|business|technology|@|http|www\.|\d{4}|\bof\b|"
                     r"editor|board|journal|informs|elsevier|wiley|springer|siam|copyright|sign in|subscribe|search|"
                     r"menu|home|about|help|cookies|privacy|terms|access|login|issues?|current|available|"
                     r"indigenous|american|eastern|asian|african|hispanic|latino|white|black|pacific|native|"
                     r"prefer not|other|unknown|volume|article|special|call for|submit|author|reviewer", re.I)


def get(session, url, tries=5, **kw):
    delay = 5
    for i in range(tries):
        try:
            r = session.get(url, timeout=60, **kw)
            if r.status_code == 200 and r.text.strip():
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException as e:
            log.warning("%s: %s", url[:80], type(e).__name__)
        time.sleep(delay); delay = min(delay * 2, 60)
    return None


def snapshots(session, url: str) -> dict[int, str]:
    """year -> latest 200-status snapshot timestamp for the URL, from the CDX index (one call)."""
    r = get(session, "https://web.archive.org/cdx/search/cdx",
            params={"url": url, "output": "json", "fl": "timestamp,statuscode", "filter": "statuscode:200"})
    if r is None:
        return {}
    try:
        rows = r.json()[1:]
    except ValueError:
        return {}
    out: dict[int, str] = {}
    for ts, _ in rows:
        out[int(ts[:4])] = max(ts, out.get(int(ts[:4]), ""))
    return out


def parse_board(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()
    lines = [re.sub(r"\s+", " ", x).strip(" ,;·•") for x in soup.get_text("\n").split("\n")]
    lines = [x for x in lines if x]
    out, role = [], None
    for ln in lines:
        if ROLE_RE.match(ln):
            role = re.sub(r"s\b", "", ln.strip(": ").title())  # "Associate Editors" -> "Associate Editor"
            continue
        if role and 5 <= len(ln) <= 50 and not STOP_RE.search(ln):
            cand = re.sub(r"^(Dr|Prof|Professor)\.?\s+", "", ln)
            cand = re.sub(r"\s*\(.*?\)\s*$", "", cand)
            if NAME_RE.match(cand):
                out.append((role, cand))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2016-2026")
    ap.add_argument("--journals", default=",".join(BOARDS))
    ap.add_argument("--sleep", type=float, default=4.0, help="pause between archive requests")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    y0, y1 = (int(x) for x in args.years.split("-"))
    SNAP.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers["User-Agent"] = UA
    rows: dict[tuple[str, str, str], dict] = {}
    for key in args.journals.split(","):
        jname, url = BOARDS[key]
        missing = [y for y in range(y0, y1 + 1) if not (SNAP / f"{key}_{y}.html").exists()]
        avail = snapshots(s, url) if missing else {}
        if missing:
            time.sleep(args.sleep)
            log.info("%s: snapshots available for %s", key, sorted(y for y in avail if y0 <= y <= y1))
        for year in range(y0, y1 + 1):
            path = SNAP / f"{key}_{year}.html"
            if not path.exists():
                ts = avail.get(year)
                if not ts:
                    continue
                r = get(s, f"https://web.archive.org/web/{ts}id_/{url}")
                time.sleep(args.sleep)
                if r is None:
                    log.info("%s %d: fetch failed", key, year); continue
                path.write_text(r.text, encoding="utf-8")
            people = parse_board(path.read_text(encoding="utf-8", errors="ignore"))
            log.info("%s %d: %d board entries parsed", key, year, len(people))
            for role, name in people:
                k = (jname, role, normalize_name(name))
                rec = rows.setdefault(k, {"name": name, "journal": jname, "role": role, "first_year": year, "last_year": year, "n_snapshots": 0})
                rec["first_year"] = min(rec["first_year"], year); rec["last_year"] = max(rec["last_year"], year); rec["n_snapshots"] += 1
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "author_id", "journal", "role", "first_year", "last_year", "n_snapshots"])
        w.writeheader()
        for rec in sorted(rows.values(), key=lambda r: (r["journal"], r["role"], r["name"])):
            w.writerow({"author_id": "", **rec})
    log.info("wrote %d (journal, role, person) rows -> %s", len(rows), OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
