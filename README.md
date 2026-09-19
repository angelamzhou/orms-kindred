# ORMatch

Local, privacy-preserving reviewer matching for Operations Research / Management Science /
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
  `ormatch verify-offline` runs the whole pipeline with Python sockets disabled to prove it.
* The only network calls the tool ever makes are `ormatch fetch-index` (download the public
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
ormatch fetch-index https://example.org/ormatch-index-v1.tar.gz   # downloads into data/index
```

## Commands

| Command | What it does |
|---|---|
| `ormatch suggest PAPER.pdf [--n 20] [--exclude-institution I123]... [--exclude-author A456]... [--backend specter2\|scincl\|tfidf] [--index-dir data/index] [--json]` | Rank reviewers for one PDF; prints a table (or JSON) with per-reviewer evidence papers. |
| `ormatch batch DIR [--out results.jsonl]` | Run `suggest` on every PDF in a directory, one JSON object per line. |
| `ormatch verify-offline PAPER.pdf` | Same as `suggest`, but with `socket.socket` monkeypatched to raise; exits non-zero if anything tries to reach the network. |
| `ormatch fetch-index URL [--sha256 HEX]` | Download the public index tarball, verify its SHA-256 (from `--sha256` or `URL.sha256`), unpack into `--index-dir`. |
| `ormatch ui` | Launch the Streamlit app (`src/ormatch/ui_app.py`): drop a PDF, get a reviewer table with expandable evidence. |

Title/abstract extraction (`ormatch.pdf`) is heuristic: the title is the leading lines before
"Abstract" (stopping at author-looking lines), the abstract is the text between "Abstract" and
"Introduction"/"1."/"Keywords" capped at 400 words, falling back to the first 300 words.

## How the index is built

Scripts live in `scripts/` and use the modules in `src/ormatch/`:

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
   recall@10 0.26 -> 0.32, recall@20 0.30 -> 0.40 at the default weight 0.1.
5. `tar czf ormatch-index-vN.tar.gz -C data index && sha256sum ... > ormatch-index-vN.tar.gz.sha256`
   publishes a new index version.

## Licence

Code is released under the Apache License 2.0. The index is derived from OpenAlex metadata
(CC0); embedding model weights are subject to their own licences (SPECTER2, SciNCL: Apache-2.0).
