#!/usr/bin/env python3
"""Package the public index for distribution: one tarball + its sha256, consumable by
`kindred fetch-index URL`.

Contents (all public OpenAlex-derived data, nothing about any manuscript):
  index/embeddings.npy, index/paper_ids.json, index/meta.json,
  papers.parquet, authorships.parquet, references.parquet, authors.parquet (optional),
  editors.csv (optional), coi/genealogy.csv (optional).
`fetch-index` unpacks into --index-dir (default data/index) and the CLI finds the tables in
that directory or its parent, so the layout below is unpacked with --index-dir data.

Usage: python scripts/package_index.py [--version v1] [--out dist/]
       python scripts/package_index.py --collection stats   # package data/index_stats/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=time.strftime("v%Y.%m.%d"))
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--collection", default=None, help="package an add-on collection dir data/index_<name> instead of the core")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.collection:
        src = ROOT / "data" / f"index_{args.collection}"
        name = f"kindred-index-{args.collection}-{args.version}"
        members = [(src / f, f) for f in ("embeddings.npy", "paper_ids.json", "meta.json", "papers.parquet",
                                            "authorships.parquet", "references.parquet") if (src / f).exists()]
    else:
        d = ROOT / "data"
        name = f"kindred-index-{args.version}"
        members = [(d / "index" / f, f"index/{f}") for f in ("embeddings.npy", "paper_ids.json", "meta.json")]
        members += [(d / f, f) for f in ("papers.parquet", "authorships.parquet", "references.parquet", "authors.parquet",
                                         "editors.csv") if (d / f).exists()]
        if (d / "coi" / "genealogy.csv").exists():
            members.append((d / "coi" / "genealogy.csv", "coi/genealogy.csv"))
    missing = [str(p) for p, _ in members if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    tar_path = out / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tf:
        for p, arc in members:
            tf.add(p, arcname=arc)
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (out / f"{name}.tar.gz.sha256").write_text(f"{digest}  {tar_path.name}\n")
    meta = json.loads((members[0][0].parent / "meta.json").read_text()) if not args.collection else json.loads((src / "meta.json").read_text())
    print(f"{tar_path}  {tar_path.stat().st_size/1e6:.0f} MB  sha256 {digest[:16]}...  ({meta.get('n')} papers, {meta.get('backend')})")
    print(f"install: kindred fetch-index <URL of {tar_path.name}> --index-dir data" + ("" if not args.collection else f"/index_{args.collection}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
