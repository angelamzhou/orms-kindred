"""PDF text extraction and title/abstract heuristics (pypdfium2, fully offline)."""
from __future__ import annotations

import re
from pathlib import Path

_ABSTRACT_RE = re.compile(r"^\s*(abstract|summary)\b[\s.:—-]*", re.IGNORECASE)
_INTRO_RE = re.compile(
    r"^\s*(1\.?\s|1\s*\.|I\.\s|introduction\b|keywords?\b|key\s*words\b|index terms\b|"
    r"jel\b|msc\b|history\b|subject classifications?\b|area of review\b)",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"(arxiv|@|http|doi|university|department|school|institute|college|"
    r"\bcollege\b|received|accepted|copyright|©|\bpreprint\b|submitted to|"
    r"^\d{1,2}\s+\w+\s+\d{4}$|^\s*[\d,\s*†‡§¶]+\s*$)",
    re.IGNORECASE,
)


def extract_text(path: str | Path, max_pages: int = 3) -> str:
    """Return concatenated text of the first `max_pages` pages of a PDF."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        parts = []
        for i in range(min(len(pdf), max_pages)):
            page = pdf[i]
            tp = page.get_textpage()
            parts.append(tp.get_text_bounded() or "")
            tp.close()
            page.close()
        return "\n".join(parts)
    finally:
        pdf.close()


def _clean(s: str) -> str:
    s = s.replace("\r", "\n").replace("­", "")
    s = re.sub(r"-\n(?=[a-z])", "", s)  # de-hyphenate line breaks
    s = re.sub(r"[ \t]+", " ", s)
    return s


def _cap_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n])


_NAME_TOKEN = re.compile(r"^[A-Z][a-zA-Z'’-]+\.?[\d*∗†‡§¶,]*$|^[A-Z]\.$|^(and|&)$")


def _looks_like_authors(ln: str) -> bool:
    """'Jane Q. Researcher1', 'A. Smith, B. Jones and C. Lee' -> True; title lines -> False."""
    toks = ln.replace(",", " ").split()
    if not 2 <= len(toks) <= 12:
        return False
    if re.search(r"\d[\d*∗†‡§¶]*$", ln.split()[-1]) or re.search(r"[a-zA-Z]\d", ln):
        return True  # trailing affiliation superscript
    if re.search(r"[*∗†‡§¶]$", toks[-1]) and len(toks) <= 8 and all(_NAME_TOKEN.match(t) for t in toks):
        return True  # 'Laixi Shi∗' (arXiv-style footnote marker, U+2217 as well as ASCII *)
    lowered = [t for t in toks if t[0].islower() and t not in ("and", "&", "de", "van", "von", "der")]
    has_sep = "," in ln or any(t in ("and", "&") for t in toks) or any(re.match(r"^[A-Z]\.$", t) for t in toks)
    return has_sep and not lowered and all(_NAME_TOKEN.match(t) for t in toks) and len(toks) <= 8 and not ln.endswith(":")


def guess_title_abstract(text: str) -> tuple[str, str]:
    """Heuristic (title, abstract) from raw first-pages text.

    title    = first substantive lines before 'Abstract' (stops at author/affil-looking lines).
    abstract = text between 'Abstract' and 'Introduction' / '1.' / 'Keywords', capped at 400 words.
    fallback = first 300 words as abstract when no 'Abstract' marker is found.
    """
    text = _clean(text)
    lines = [ln.strip() for ln in text.split("\n")]
    nonempty = [(i, ln) for i, ln in enumerate(lines) if ln]

    abs_idx = next((i for i, ln in nonempty if _ABSTRACT_RE.match(ln)), None)

    # --- title -------------------------------------------------------------
    title_lines: list[str] = []
    head = [ln for i, ln in nonempty if abs_idx is None or i < abs_idx][:12]
    for ln in head:
        if _META_RE.search(ln) or len(ln) < 4:
            if title_lines:
                break
            continue
        if title_lines and (len(ln.split()) <= 4 and "," in ln):  # author-list heuristic
            break
        if title_lines and _looks_like_authors(ln):
            break
        title_lines.append(ln)
        if len(" ".join(title_lines).split()) >= 25:
            break
    title = " ".join(title_lines).strip()
    title = re.sub(r"[\*∗\d†‡§¶]+$", "", title).strip()

    # --- abstract ----------------------------------------------------------
    if abs_idx is not None:
        first = _ABSTRACT_RE.sub("", lines[abs_idx]).strip()
        body = [first] if first else []
        for ln in lines[abs_idx + 1 :]:
            if _INTRO_RE.match(ln) and body:
                break
            body.append(ln)
            if len(" ".join(body).split()) > 450:
                break
        abstract = _cap_words(" ".join(b for b in body if b), 400)
    else:
        start = len(" ".join(title_lines)) if title_lines else 0
        abstract = _cap_words(" ".join(text.split())[start:], 300)

    if not title:
        title = _cap_words(abstract, 12)
    return title, abstract


def pdf_to_query(path: str | Path) -> dict:
    """Convenience: {'title','abstract','text'} for a PDF path."""
    text = extract_text(path)
    title, abstract = guess_title_abstract(text)
    return {"title": title, "abstract": abstract, "text": f"{title}. {abstract}".strip()}


# --------------------------------------------------------------------------- references
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>)\]]+", re.IGNORECASE)
_REFS_HEAD_RE = re.compile(r"^\s*(references|bibliography|literature cited)\s*$", re.IGNORECASE)
_YEAR_RE = re.compile(r"\(?\b(19|20)\d{2}[a-z]?\b\)?")
_ENTRY_START_RE = re.compile(r"^\s*(\[\d+\]|\d{1,3}\.\s|[A-Z][A-Za-z'’\-]+,\s)")


def normalize_title(t: str) -> str:
    """Lower-case alphanumerics only; used to match bibliography titles to the index."""
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def extract_references(path: str | Path, max_pages: int = 60) -> list[dict]:
    """Parse the bibliography of a PDF into entries with a DOI and/or a title guess.

    Fully local: text from pypdfium2, the 'References' heading located from the end of the
    document, entries split on numbering / 'Surname, I.' starts or blank lines, DOIs pulled by
    regex, and for the rest a title guess = the sentence following the year.
    Returns [{'raw','doi','title','year'}]; runs in well under a second for a normal paper.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        n = len(pdf)
        pages = []
        for i in range(max(0, n - max_pages), n):
            page = pdf[i]
            tp = page.get_textpage()
            pages.append(tp.get_text_bounded() or "")
            tp.close()
            page.close()
    finally:
        pdf.close()
    return parse_reference_text("\n".join(pages), require_heading=True)


def parse_reference_text(text: str, require_heading: bool = False) -> list[dict]:
    """Parse bibliography entries from plain text (a pasted reference list, or the tail of a PDF).
    With require_heading, only text after the last 'References' heading is used."""
    text = _clean(text)
    lines = text.split("\n")
    start = None
    for i in range(len(lines) - 1, -1, -1):  # last 'References' heading wins
        if _REFS_HEAD_RE.match(lines[i]):
            start = i + 1
            break
    if start is None:
        if require_heading:
            return []
        start = 0
    body = lines[start:]
    # Stop at appendix-like headings that follow the bibliography.
    for j, ln in enumerate(body):
        if re.match(r"^\s*(appendix|supplementary|online appendix|e-companion)\b", ln, re.I) and j > 5:
            body = body[:j]
            break
    # pypdfium2 emits a blank line after every physical line, so blank lines carry no
    # information; an entry starts where a line looks like '[n]', 'n.' or 'Surname, I.' and
    # the text accumulated so far already ends a sentence (period, closing paren, page range).
    entries, cur = [], []
    for ln in body:
        ln = ln.strip()
        if not ln:
            continue
        if cur and _ENTRY_START_RE.match(ln) and re.search(r"[.)\]]$|\d$", cur[-1]):
            entries.append(" ".join(cur)); cur = []
        cur.append(ln)
    if cur:
        entries.append(" ".join(cur))
    out = []
    for raw in entries:
        raw = re.sub(r"\s+", " ", raw).strip()
        if len(raw) < 20:
            continue
        doi = _DOI_RE.search(raw)
        doi = doi.group(0).rstrip(".,;").lower() if doi else None
        ym = _YEAR_RE.search(raw)
        year = int(ym.group(0).strip("()")[:4]) if ym else None
        title = None
        if ym:
            after = raw[ym.end():].lstrip(" .):")
            # title = up to the first period followed by a space+capital, or a quote pair
            q = re.match(r"[“\"](.+?)[”\"]", after)
            if q:
                title = q.group(1)
            else:
                m = re.match(r"(.+?[a-z0-9\)\?])\.\s+(?=[A-Z])", after)
                title = m.group(1) if m else after[:200]
            title = title.strip(" .,")
        out.append({"raw": raw, "doi": doi, "title": title, "year": year})
    return out


def match_references(refs: list[dict], papers, min_title_chars: int = 25) -> list[str]:
    """Map parsed bibliography entries to index paper ids by DOI, then by normalized title
    (exact, then 'index title is a prefix of the entry's title guess', which absorbs trailing
    venue text). ``papers`` is the papers DataFrame (openalex_work_id, doi, title)."""
    by_doi = {d.lower(): w for d, w in zip(papers["doi"], papers["openalex_work_id"]) if isinstance(d, str)}
    norm = [(normalize_title(t), w) for t, w in zip(papers["title"], papers["openalex_work_id"]) if isinstance(t, str)]
    exact = {n: w for n, w in norm if len(n) >= min_title_chars}
    long_titles = [(n, w) for n, w in norm if len(n) >= min_title_chars]
    hits: list[str] = []
    for r in refs:
        w = None
        if r.get("doi"):
            w = by_doi.get(r["doi"].lower())
        if w is None and r.get("title"):
            n = normalize_title(r["title"])
            if len(n) >= min_title_chars:
                w = exact.get(n)
                if w is None:
                    w = next((wid for t, wid in long_titles if n.startswith(t)), None)
        if w:
            hits.append(w)
    return sorted(set(hits))
