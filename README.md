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
| `ormatch suggest PAPER.pdf [--n 20] [--exclude-institution I123]... [--exclude-author A456]... [--backend specter2\|scincl\|tfidf] [--index-dir data/index] [--json] [--cite-weight 0.1] [--lam 0.3] [--k 3] [--half-life YEARS]` | Rank reviewers for one PDF; prints a table (or JSON) with per-reviewer evidence papers and the weights used. |
| `ormatch batch DIR [--out results.jsonl]` | Run `suggest` on every PDF in a directory, one JSON object per line. |
| `ormatch verify-offline PAPER.pdf` | Same as `suggest`, but with `socket.socket` monkeypatched to raise; exits non-zero if anything tries to reach the network. |
| `ormatch fetch-index URL [--sha256 HEX]` | Download the public index tarball, verify its SHA-256 (from `--sha256` or `URL.sha256`), unpack into `--index-dir`. |
| `ormatch ui` | Launch the Streamlit app (`src/ormatch/ui_app.py`): drop a PDF, get a reviewer table with expandable evidence. Sidebar sliders change the weights below and re-rank instantly (the PDF is embedded once and cached). |

### Ranking weights

`score(author) = (1 - lam) * max_i sim_i + lam * mean(top-k sim_i) + cite_weight * (1 - 0.5^n_cited)`

| weight | default | meaning | where it comes from |
|---|---|---|---|
| `lam` | 0.3 | 0 = an author's single most similar paper; 1 = mean of their top-`k`. | Max vs. mean pooling is the standard design choice (OpenReview `max_score`/`average_score`; TPMS concatenates or averages). Stelmakh et al. find max pooling best for SPECTER2; our leave-one-out agrees (lam 0 edges out 0.3). |
| `k` | 3 | papers per author in the mean | ACL's matcher uses the top 3 cosines weighted 1, 1/2, 1/3. |
| `cite_weight` | 0.1 | bonus for authors of papers the manuscript cites, saturating in `n_cited` | Our addition; the reference list as a reviewer signal goes back to Rodriguez & Bollen (2008). The saturating form is a design choice; 0.1 is where leave-one-out MRR plateaus (0.20 -> 0.32). |
| `half_life` | off | years until an old paper keeps half its advantage over an average paper | Product preference for active reviewers; costs accuracy in leave-one-out. |

Signals are combined linearly, as TPMS/CMT and OpenReview (SPECTER 0.8 / MFR 0.2) do; none of the published systems learn these weights without elicited reviewer scores, so they are exposed rather than fixed.

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
