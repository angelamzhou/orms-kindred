#!/usr/bin/env bash
# Build a self-contained add-on index for one collection from src/kindred/sources.py:
#   scripts/build_collection.sh stats-ml [--backend specter2] [--from-year 2014]
# Produces data/index_<name>/ with embeddings.npy, paper_ids.json, meta.json AND its own
# papers/authorships/references.parquet, so it can be tarred and downloaded on its own and
# queried together with the core index: kindred suggest X.pdf --index-dir data/index --index-dir data/index_stats-ml
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?collection name}"; shift
BACKEND=specter2; FROM_YEAR=2014
while [ $# -gt 0 ]; do case "$1" in --backend) BACKEND="$2"; shift 2;; --from-year) FROM_YEAR="$2"; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
PY="${PYTHON:-python3}"; [ -x .venv/bin/python ] && PY=.venv/bin/python
RAW="data/collections/$NAME/raw"; OUT="data/index_$NAME"
mkdir -p "$RAW" "$OUT"
SOURCES=$($PY -c "import sys; sys.path.insert(0,'src'); from kindred.sources import collection_sources; print(' '.join(collection_sources('$NAME')))")
for S in $SOURCES; do
  for mode in "" "--non-primary"; do
    for attempt in 1 2 3; do
      $PY scripts/pull_journal_works.py --source "$S" --from-year "$FROM_YEAR" --sleep 0.12 $mode \
          --out "$RAW/works_${S}$([ -n "$mode" ] && echo _nonprimary).jsonl" 2>&1 | grep -v -E 'INFO page [0-9]+:' && break
      sleep 15
    done
  done
done
$PY scripts/normalize_works.py --raw-dir "$RAW" --out-dir "$OUT"
$PY scripts/build_index.py --papers "$OUT/papers.parquet" --out "$OUT" --backend "$BACKEND" --batch-size 64 --collection "$NAME"
echo "### COLLECTION $NAME DONE -> $OUT"
