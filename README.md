# orms-kindred

Abstract-based, privacy-preserving reviewer ranking for Operations Research / Management Science /
Operations Management papers.

Give it a manuscript PDF; it returns a ranked list of candidate reviewers drawn from the
authors of recent OR/MS/OM papers (indexed from OpenAlex), each with the papers that justify the
match. Conflicts can be excluded by institution or author.

## Privacy model: public index, private query

* **Public index.** Titles, abstracts, authorships and pre-computed embeddings of published
  OR/MS/OM papers are built offline (see `scripts/`) and distributed as a versioned tarball.
  Nothing in it is confidential.
* **Private query.** The manuscript is processed entirely on your machine: text extraction
  (`pypdfium2`), embedding (a local SPECTER2/SciNCL model or TF-IDF), nearest-neighbour search
  and reviewer aggregation. No text, embedding or metadata about the submission is sent anywhere.
  `kindred verify-offline` runs the whole pipeline with Python sockets disabled to prove it.
* The only network calls the tool ever makes are `kindred fetch-index` (download the public
  index) and, optionally, the first download of the embedding model weights from Hugging Face.

## Install

```bash
pip install -e .              # core: PDF parsing, TF-IDF backend, CLI
pip install -e ".[embed]"     # + torch/transformers/adapters for SPECTER2 / SciNCL
                              #   SPECTER2 needs torch>=2.6 (its weights are pickled .bin files that
                              #   transformers refuses to load on older torch); SciNCL works on any torch.
                              #   In an environment pinned to an older torch, use a venv:
                              #   python -m venv .venv && .venv/bin/pip install -e ".[embed]" "torch>=2.6"
pip install -e ".[ui]"        # + streamlit drag-and-drop UI
kindred fetch-index https://example.org/orms-kindred-index-v1.tar.gz   # downloads into data/index
```

## Commands

| Command | What it does |
|---|---|
| `kindred suggest PAPER.pdf [--n 20] [--exclude-institution I123]... [--exclude-author A456]... [--backend specter2\|scincl\|tfidf] [--index-dir data/index] [--json] [--cite-weight 0.1] [--lam 0.3] [--k 3] [--half-life YEARS]` | Rank reviewers for one PDF; prints a table (or JSON) with per-reviewer evidence papers and the weights used. |
| `kindred batch DIR [--out results.jsonl]` | Run `suggest` on every PDF in a directory, one JSON object per line. |
| `kindred verify-offline PAPER.pdf` | Same as `suggest`, but with `socket.socket` monkeypatched to raise; exits non-zero if anything tries to reach the network. |
| `kindred fetch-index URL [--sha256 HEX]` | Download the public index tarball, verify its SHA-256 (from `--sha256` or `URL.sha256`), unpack into `--index-dir`. |
| `kindred ui` | Launch the Streamlit app (`src/kindred/ui_app.py`): drop a PDF, get a reviewer table with expandable evidence. Sidebar sliders change the weights below and re-rank instantly (the PDF is embedded once and cached). |

### Ranking weights

`score(author) = (1 - lam) * max_i sim_i + lam * mean(top-k sim_i) + cite_weight * (1 - 0.5^n_cited)`

| weight | default | meaning | where it comes from |
|---|---|---|---|
| `lam` | 0.3 | 0 = an author's single most similar paper; 1 = mean of their top-`k`. | Max vs. mean pooling is the standard design choice (OpenReview `max_score`/`average_score`; TPMS concatenates or averages). Stelmakh et al. find max pooling best for SPECTER2; our leave-one-out agrees (lam 0 edges out 0.3). |
| `k` | 3 | papers per author in the mean | ACL's matcher uses the top 3 cosines weighted 1, 1/2, 1/3. |
| `cite_weight` | 0.1 | bonus for authors of papers the manuscript cites, saturating in `n_cited` | Our addition; the reference list as a reviewer signal goes back to Rodriguez & Bollen (2008). The saturating form is a design choice; 0.1 is where leave-one-out MRR plateaus (0.20 -> 0.32). |
| `half_life` | off | years until an old paper keeps half its advantage over an average paper | Product preference for active reviewers; costs accuracy in leave-one-out. |

Signals are combined linearly, as TPMS/CMT and OpenReview (SPECTER 0.8 / MFR 0.2) do; none of the published systems learn these weights without elicited reviewer scores, so they are exposed rather than fixed.

Further knobs (all off by default, all in the CLI and the UI sidebar; leave-one-out cost measured with the
citation boost at 0.1, baseline MRR 0.317 / recall@10 0.324):

| knob | what it does | needs | LOO cost |
|---|---|---|---|
| `--volume-correction 0..1` | subtracts the expected best-of-n cosine of n random papers, so prolific authors stop winning on volume ("usual suspects") | nothing | 0.5 -> MRR 0.263; 1.0 -> 0.220 |
| `--seniority-weight w` | penalty per log-unit of OpenAlex works_count above the median author | `data/authors.parquet` from `scripts/fetch_author_stats.py` | 0.01 -> MRR 0.292; 0.03 -> 0.242 |
| `--editor-weight w` | penalty per editorial role: 1 per current role, 0.5 per past role | `data/editors.csv` from `scripts/fetch_editorial_boards.py`, which reads yearly Internet Archive snapshots of 15 journals' board pages (publisher sites block scripts) | not measured |
| `--min-or-links n` | with add-on collections, drop authors with fewer links to the core OR literature (core papers + citations to/from core); default 1 | references.parquet | n/a |

The leave-one-out metric rewards finding a paper's *actual* authors, who are disproportionately prolific, so every knob that de-emphasises volume or seniority lowers it. That is expected: these are preferences about who should review, not accuracy tuning.

### Seeds and diversity

`--seed "Name"` (repeatable) names reviewers you already have in mind. Each candidate gets
`--seed-weight` x cosine between their profile (mean embedding of their indexed papers) and the closest seed's
profile, so the list tilts toward the *kind* of reviewer the seeds represent, which the abstract alone cannot
express; the seeds themselves are removed from the output. `--diversity d` (0..1) re-ranks the top pool by
maximal marginal relevance, `(1-d) * score - d * max cosine to seeds and already-picked reviewers`, so
consecutive picks cover different neighbourhoods instead of clones of the seeds. Both are sliders in the UI.

### Conflicts of interest

`kindred suggest PAPER.pdf --author "Jane Doe" --author A5012345678 ...` resolves the manuscript's authors against the
index (exact normalised name, then surname + initials; ambiguous names are reported) and flags, with the evidence in a
`COI?` column: their co-authors on papers from the last `--coi-years` (default 5), people at the same institution, and
advisor/student pairs from `data/coi/genealogy.csv` (fill it by hand or with `scripts/fetch_genealogy.py NAME ...`, which
queries the Mathematics Genealogy Project). Flags are hints for a human to follow up; `--coi exclude` drops them instead.
The manuscript authors themselves are always removed.

### Add-on collections

The core index covers 29 OR/MS venues (73,643 papers from 2014; abstract coverage 62.5% after the Semantic Scholar backfill, the gap being Elsevier titles). Only ~29% of the references in the original 17-venue sample pointed back into those venues. Add-on
collections (`src/kindred/sources.py: COLLECTIONS`: `applied-or`, `econ-finance`, `stats-ml`, `algorithms`) are built as
self-contained directories with `scripts/build_collection.sh NAME` -> `data/index_NAME/` (embeddings plus their own
parquet tables) and can be distributed and downloaded separately. Query several at once with repeated `--index-dir`
or `--all-collections`; the UI lists every collection it finds. Authors who appear only in add-ons must have at least
`--min-or-links` links to the core literature, so the pool widens to people connected to OR rather than to arbitrary outsiders.

### Departments

`data/departments.csv` lists 167 OR/IE/OM departments (US, Canada, UK, continental Europe, Israel, Turkey, China, Hong
Kong, Singapore, Korea, Japan, India, Australia, Latin America) with roster URLs and OpenAlex institution IDs
(`scripts/fill_openalex_ids.py`, `scripts/scrape_rosters.py`, `scripts/resolve_authors.py`). Rosters identify who holds
a faculty position where; many university sites block scripts, so expect failures and use `--html-dir` with saved pages.

Title/abstract extraction (`kindred.pdf`) is heuristic: the title is the leading lines before
"Abstract" (stopping at author-looking lines), the abstract is the text between "Abstract" and
"Introduction"/"1."/"Keywords" capped at 400 words, falling back to the first 300 words.

## How the index is built

Scripts live in `scripts/` and use the modules in `src/kindred/`:

1. `openalex.py` / `sources.py` pull works from OpenAlex for a curated list of OR/MS/OM venues
   (e.g. Operations Research, Management Science, M&SOM, MOR, POM, Transportation Science,
   Mathematical Programming) for a recent window, keeping title, abstract, venue, year,
   authorships and institutions. Set `OPENALEX_API_KEY` in `.env` for polite-pool rate limits.
2. `embed.py` embeds `title + abstract` with the chosen backend (`Embedder(backend)`).
3. `index.py` writes `data/index/` (embedding matrix + paper ids) beside `papers.parquet` and
   `authorships.parquet`.
4. `match.py` aggregates paper-level similarities into author scores (a diminishing-returns sum,
   parameter `lam`) and attaches the top-`k` evidence papers per author; `eval.py` measures
   recall against held-out citation/authorship signals.
   `suggest` also parses the manuscript's bibliography locally (`pdf.extract_references`),
   matches entries to indexed papers by DOI or title, and adds `--cite-weight * (1 - 0.5**n)`
   to authors with `n` cited papers. Leave-one-out on the 2014+ index (n=200, SPECTER2), using
   each held-out paper's OpenAlex references as the bibliography: MRR 0.20 -> 0.32,
   recall@10 0.26 -> 0.32, recall@20 0.30 -> 0.40 at the default weight 0.1 (17-venue index;
   on the 29-venue, 73,643-paper index: MRR 0.324, recall@10 0.327, recall@20 0.388).
5. `tar czf orms-kindred-index-vN.tar.gz -C data index && sha256sum ... > orms-kindred-index-vN.tar.gz.sha256`
   publishes a new index version.

## Licence

Code is released under the Apache License 2.0. The index is derived from OpenAlex metadata
(CC0); embedding model weights are subject to their own licences (SPECTER2, SciNCL: Apache-2.0).
