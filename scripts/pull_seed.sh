#!/usr/bin/env bash
# Pull the initial seed journals (OR + M&SOM, 2014+) and normalize. Resumable: rerun freely.
set -euo pipefail
cd "$(dirname "$0")/.."
for S in S125775545 S81410195; do
  python3 scripts/pull_journal_works.py --source "$S" --from-year 2014 --out "data/raw/works_${S}.jsonl"
done
python3 scripts/normalize_works.py
