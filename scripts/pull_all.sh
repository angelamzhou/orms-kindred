#!/usr/bin/env bash
# Pull every journal in src/ormatch/sources.py from 2014 onward, one source at a time,
# retrying each up to 4 times. Resumable: pull_journal_works.py keeps a cursor per source,
# so rerunning this script only fetches what is missing. Then normalize.
set -uo pipefail
cd "$(dirname "$0")/.."
SOURCES=$(python3 -c "import sys; sys.path.insert(0,'src'); from ormatch.sources import SOURCES; print(' '.join(SOURCES))")
FROM_YEAR="${FROM_YEAR:-2014}"
FAILED=""
for S in $SOURCES; do
  ok=0
  for attempt in 1 2 3 4; do
    echo "### $S attempt $attempt $(date -u +%FT%TZ)"
    python3 scripts/pull_journal_works.py --source "$S" --from-year "$FROM_YEAR" --sleep 0.15 2>&1 | grep -v -E 'INFO page [0-9]+:'
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('data/raw/state_$S.json')).get('done') else 1)" 2>/dev/null; then
      ok=1; break
    fi
    sleep $((attempt * 10))
  done
  [ "$ok" = 1 ] || FAILED="$FAILED $S"
done
echo "### pull finished. failed:${FAILED:- none}"
python3 scripts/normalize_works.py
[ -z "$FAILED" ]
