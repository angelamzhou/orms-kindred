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
