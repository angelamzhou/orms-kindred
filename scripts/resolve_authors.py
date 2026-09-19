#!/usr/bin/env python3
"""Resolve roster names to OpenAlex author IDs.

Usage:
    python scripts/resolve_authors.py <dept_id> [--max 40] [--sleep 0.2]

Reads data/rosters_raw/<dept_id>.csv and the department's openalex_institution_id
from data/departments.csv, queries
  https://api.openalex.org/authors?search=<name>&filter=affiliations.institution.id:<inst_id>
(falls back to an unfiltered search if no institution id is known), picks the top
hit whose affiliations include the institution, and writes
data/reviewers_<dept_id>.csv with a confidence score. Output is flushed after
every row so partial progress survives interruption.
"""
import argparse, csv, os, re, sys, time
from pathlib import Path

import requests
try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(ROOT / ".env")
API_KEY = os.environ.get("OPENALEX_API_KEY", "")
BASE = "https://api.openalex.org/authors"
FIELDS = ["dept_id", "name", "title", "profile_url", "openalex_author_id",
          "openalex_display_name", "matched_institution", "works_count",
          "cited_by_count", "confidence", "note"]


def norm(s):
    return re.sub(r"[^a-z ]", "", (s or "").lower()).split()


def name_similarity(a, b):
    ta, tb = norm(a), norm(b)
    if not ta or not tb:
        return 0.0
    if ta[-1] != tb[-1]:  # surname must agree
        return 0.0
    first_ok = ta[0] == tb[0] or ta[0][0] == tb[0][0]
    return 1.0 if ta[0] == tb[0] else (0.7 if first_ok else 0.4)


def query(name, inst_id, session):
    params = {"search": name, "per_page": 5}
    if inst_id:
        params["filter"] = f"affiliations.institution.id:{inst_id}"
    if API_KEY:
        params["api_key"] = API_KEY
    for attempt in range(3):
        try:
            r = session.get(BASE, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json().get("results", [])
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def pick(name, inst_id, results):
    best, best_conf, matched = None, 0.0, ""
    for res in results:
        insts = {a["institution"]["id"].rsplit("/", 1)[-1]: a["institution"]["display_name"]
                 for a in res.get("affiliations", []) if a.get("institution")}
        sim = name_similarity(name, res.get("display_name", ""))
        if sim == 0:
            continue
        inst_hit = inst_id in insts if inst_id else False
        conf = sim * (1.0 if inst_hit else 0.6)
        if res.get("works_count", 0) < 3:
            conf *= 0.7
        if conf > best_conf:
            best, best_conf, matched = res, conf, insts.get(inst_id, "")
    return best, round(best_conf, 2), matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dept_id")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    depts = {d["dept_id"]: d for d in csv.DictReader(open(ROOT / "data" / "departments.csv", encoding="utf-8"))}
    inst_id = depts[args.dept_id].get("openalex_institution_id", "")
    roster = list(csv.DictReader(open(ROOT / "data" / "rosters_raw" / f"{args.dept_id}.csv", encoding="utf-8")))
    out_path = ROOT / "data" / f"reviewers_{args.dept_id}.csv"
    session = requests.Session()
    session.headers["User-Agent"] = "ORMatch-resolver/0.1"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); f.flush()
        for i, row in enumerate(roster[: args.max]):
            name = re.sub(r"^(Dr|Prof|Professor)\.?\s+", "", row["name"]).strip()
            out = {k: row.get(k, "") for k in ("dept_id", "name", "title", "profile_url")}
            try:
                results = query(name, inst_id, session)
                if not results and inst_id:  # retry without institution filter
                    results = query(name, "", session)
                best, conf, matched = pick(name, inst_id, results)
                if best:
                    out.update(openalex_author_id=best["id"].rsplit("/", 1)[-1],
                               openalex_display_name=best.get("display_name", ""),
                               matched_institution=matched, works_count=best.get("works_count", ""),
                               cited_by_count=best.get("cited_by_count", ""), confidence=conf, note="")
                else:
                    out.update(confidence=0, note="no match")
            except Exception as e:
                out.update(confidence=0, note=f"error: {type(e).__name__}")
            w.writerow(out); f.flush()
            print(f"[{i+1}/{min(len(roster), args.max)}] {name} -> {out.get('openalex_author_id','')} ({out.get('confidence')})")
            time.sleep(args.sleep)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
