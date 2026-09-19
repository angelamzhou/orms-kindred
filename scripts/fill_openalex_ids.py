"""Fill openalex_institution_id in data/departments.csv (one search per distinct institution)."""
import csv, os, sys, time, requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
KEY = os.environ.get("OPENALEX_API_KEY", "")
PATH = ROOT / "data" / "departments.csv"

rows = list(csv.DictReader(open(PATH, newline="", encoding="utf-8")))
cache = {r["institution"]: r["openalex_institution_id"] for r in rows if r["openalex_institution_id"]}

def save():
    with open(PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

for i, r in enumerate(rows):
    inst = r["institution"]
    if inst in cache:
        r["openalex_institution_id"] = cache[inst]; continue
    try:
        params = {"search": inst, "per_page": 1}
        if KEY: params["api_key"] = KEY
        resp = requests.get("https://api.openalex.org/institutions", params=params, timeout=20)
        if resp.status_code == 429:
            print("429; stopping"); break
        resp.raise_for_status()
        res = resp.json().get("results", [])
        oid = res[0]["id"].rsplit("/", 1)[-1] if res else ""
        cache[inst] = oid; r["openalex_institution_id"] = oid
        print(inst, "->", oid, res[0]["display_name"] if res else "")
    except Exception as e:
        print(inst, "FAILED", type(e).__name__, file=sys.stderr)
    time.sleep(0.15)
    if i % 5 == 0: save()
save()
