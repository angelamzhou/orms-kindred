#!/usr/bin/env python3
"""Look up advisor/student links on the Mathematics Genealogy Project for a list of names and
append them to data/coi/genealogy.csv (person_a, person_b, relation, source).

Usage: python scripts/fetch_genealogy.py "Jane Q. Researcher" ["Another Name" ...]
       python scripts/fetch_genealogy.py --from-roster data/rosters_raw/mit_orc.csv

MGP is name-keyed and covers mathematics/OR/statistics PhDs; matches are heuristic (first hit
with the same surname), so rows carry the MGP id for checking. The file is read by
`ormatch suggest --author ...`: a candidate linked to a manuscript author is flagged (not
dropped) with the relation as evidence. Network is used only by this script, never by suggest.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "coi" / "genealogy.csv"
BASE = "https://www.mathgenealogy.org"
UA = "ORMatch-genealogy/0.1 (academic; local reviewer matching)"


def search(session: requests.Session, name: str) -> list[tuple[str, str]]:
    toks = name.replace(",", " ").split()
    if not toks:
        return []
    given, family = " ".join(toks[:-1]), toks[-1]
    r = session.post(f"{BASE}/query-prep.php", data={"given_name": given, "family_name": family, "other_names": "",
                                                    "school": "", "year": "", "thesis": "", "country": ""}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    hits = []
    for a in soup.select("a[href*='id.php?id=']"):
        hits.append((a["href"].split("id=")[-1], a.get_text(" ", strip=True)))
    if not hits and "id.php?id=" in r.url:  # single match redirects straight to the page
        hits.append((r.url.split("id=")[-1], name))
    return hits


def person(session: requests.Session, mgp_id: str) -> dict:
    r = session.get(f"{BASE}/id.php", params={"id": mgp_id}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    name = soup.find("h2").get_text(" ", strip=True) if soup.find("h2") else mgp_id
    advisors = []
    for m in re.finditer(r"Advisor(?: \d)?:\s*(.+)", text):
        advisors.append(m.group(1).strip())
    students = []
    tbl = soup.find("table")
    if tbl:
        for row in tbl.find_all("tr")[1:]:
            cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
            if cells:
                students.append(cells[0])
    return {"name": name, "advisors": advisors, "students": students}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--from-roster", help="rosters_raw CSV with a 'name' column")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    names = list(args.names)
    if args.from_roster:
        names += [r["name"] for r in csv.DictReader(open(args.from_roster, encoding="utf-8"))]
    if not names:
        ap.error("give names or --from-roster")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT.exists()
    s = requests.Session(); s.headers["User-Agent"] = UA
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["person_a", "person_b", "relation", "source"])
        for n in names:
            try:
                hits = search(s, n)
                fam = n.replace(",", " ").split()[-1].lower()
                hit = next(((i, h) for i, h in hits if fam in h.lower()), None)
                if not hit:
                    print(f"{n}: no MGP match", file=sys.stderr); continue
                info = person(s, hit[0])
                for adv in info["advisors"]:
                    w.writerow([info["name"], adv, "advisor", f"mgp:{hit[0]}"])
                for stu in info["students"]:
                    w.writerow([info["name"], stu, "student", f"mgp:{hit[0]}"])
                f.flush()
                print(f"{n} -> {info['name']} (mgp {hit[0]}): {len(info['advisors'])} advisors, {len(info['students'])} students")
            except Exception as e:  # noqa: BLE001
                print(f"{n}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(args.sleep)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
