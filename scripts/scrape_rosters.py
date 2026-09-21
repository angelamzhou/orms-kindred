#!/usr/bin/env python3
"""Generic faculty-roster scraper.

Usage:
    python scripts/scrape_rosters.py [dept_id ...] [--html-dir DIR]

Reads data/departments.csv, fetches each roster_url (respecting robots.txt, with
timeouts, retries and a browser-like User-Agent), extracts candidate faculty
(name, title, profile_url) using heuristics, and writes
data/rosters_raw/<dept_id>.csv.

--html-dir lets you point at pre-downloaded HTML files named <dept_id>.html
(useful when direct egress is blocked); the file is used instead of fetching.
"""
import argparse, csv, re, sys, time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DEPTS = ROOT / "data" / "departments.csv"
OUT = ROOT / "data" / "rosters_raw"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36 Kindred-roster-bot/0.1 (academic; contact: local)")

TITLE_RE = re.compile(
    r"\b(Professor|Prof\.|Lecturer|Reader|Chair|Fellow|Instructor|Scientist|"
    r"Emerit(?:us|a)|Faculty|Postdoc|Dean|Director)\b", re.I)
# Two to four capitalised tokens, allowing initials, hyphens, apostrophes, accents.
NAME_RE = re.compile(
    r"^(?:(?:Dr|Prof|Professor)\.?\s+)?"
    r"(?:[A-Z][\w'’\-\.]{0,20}\s+){1,3}[A-Z][\w'’\-]{1,25}(?:,\s*(?:Jr|Sr|II|III|Ph\.?D\.?)\.?)?$")
STOP = re.compile(r"\b(read more|view|profile|email|website|cv|home|about|news|"
                  r"contact|apply|login|search|menu|skip|faculty|staff|people|"
                  r"directory|department|school|university|college|research|"
                  r"students?|program|admissions?|events?)\b", re.I)


def robots_ok(url, session):
    rp = robotparser.RobotFileParser()
    p = urlparse(url)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    try:
        r = session.get(robots_url, timeout=10)
        if r.status_code >= 400:
            return True  # no robots.txt -> allowed
        rp.parse(r.text.splitlines())
        return rp.can_fetch(UA.split()[0], url) and rp.can_fetch("*", url)
    except requests.RequestException:
        return True


def fetch(url, session, retries=3, timeout=20):
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"fetch failed for {url}: {last}")


def clean(s):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s.strip(" ,;|·•")


def looks_like_name(text):
    t = clean(text)
    if not (4 <= len(t) <= 60) or STOP.search(t) or TITLE_RE.search(t):
        return False
    t = re.sub(r"^(Dr|Prof|Professor)\.?\s+", "", t)
    return bool(NAME_RE.match(t))


def nearby_title(node):
    """Search the anchor's container (up to 3 ancestors) for a title-like string."""
    cur = node
    for _ in range(3):
        if cur is None or cur.name in ("body", "html", "main", "section", "table", "ul"):
            break
        # Only consider containers small enough to describe one person.
        strings = [clean(s) for s in cur.stripped_strings]
        if len(strings) > 12:
            break
        for s in strings:
            if TITLE_RE.search(s) and 4 <= len(s) <= 120 and not looks_like_name(s):
                return s
        cur = cur.parent
    # Fallback: the next sibling element (e.g. <h3>Name</h3><p>Title</p>).
    sib = node.find_next_sibling()
    if sib is not None:
        s = clean(sib.get_text(" "))
        if TITLE_RE.search(s) and 4 <= len(s) <= 120 and not looks_like_name(s):
            return s
    return ""


def extract(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()
    seen, rows = set(), []
    # Pass 1: anchors with name-like text.
    for a in soup.find_all("a", href=True):
        text = clean(a.get_text(" "))
        if not text:
            # image-only anchors: try alt text
            img = a.find("img")
            text = clean(img.get("alt", "")) if img else ""
        if not looks_like_name(text):
            continue
        href = urljoin(base_url, a["href"])
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        title = nearby_title(a)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"name": text, "title": title, "profile_url": href})
    # Pass 2: headings/strong tags near title keywords without links.
    for h in soup.find_all(["h2", "h3", "h4", "h5", "strong", "b"]):
        text = clean(h.get_text(" "))
        if looks_like_name(text) and text.lower() not in seen:
            title = nearby_title(h)
            if title:
                seen.add(text.lower())
                rows.append({"name": text, "title": title, "profile_url": ""})
    # Prefer rows that have a title if the page yields many hits.
    with_title = [r for r in rows if r["title"]]
    return with_title if len(with_title) >= 5 else rows


def write_rows(dept_id, rows, source):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dept_id}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dept_id", "name", "title", "profile_url", "source"])
        w.writeheader()
        for r in rows:
            w.writerow({"dept_id": dept_id, **r, "source": source})
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dept_ids", nargs="*")
    ap.add_argument("--html-dir", help="directory of <dept_id>.html files to use instead of fetching")
    args = ap.parse_args()
    depts = list(csv.DictReader(open(DEPTS, newline="", encoding="utf-8")))
    if args.dept_ids:
        depts = [d for d in depts if d["dept_id"] in set(args.dept_ids)]
    session = requests.Session()
    session.headers["User-Agent"] = UA
    for d in depts:
        did, url = d["dept_id"], d["roster_url"]
        try:
            local = Path(args.html_dir) / f"{did}.html" if args.html_dir else None
            if local and local.exists():
                html, source = local.read_text(encoding="utf-8", errors="ignore"), f"file:{local.name}"
            else:
                if not robots_ok(url, session):
                    print(f"{did}: disallowed by robots.txt, skipping", file=sys.stderr)
                    continue
                html, source = fetch(url, session), url
            rows = extract(html, url)
            path = write_rows(did, rows, source)
            print(f"{did}: {len(rows)} candidates -> {path}")
        except Exception as e:
            print(f"{did}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
