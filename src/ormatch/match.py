"""Reviewer scoring: aggregate paper-level similarities per author, with conflict filtering."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .index import PaperIndex


@dataclass
class ReviewerCandidate:
    author_id: str
    author_name: str
    score: float
    n_papers: int
    evidence: List[Tuple[str, float]] = field(default_factory=list)  # top-3 (paper_id, sim)
    institutions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "author_id": self.author_id,
            "author_name": self.author_name,
            "score": round(self.score, 4),
            "n_papers": self.n_papers,
            "evidence": [{"paper_id": p, "sim": round(s, 4)} for p, s in self.evidence],
            "institutions": self.institutions,
        }


class ReviewerMatcher:
    """Rank authors as reviewers for a manuscript.

    score(author) = (1-lam) * max(sims) + lam * mean(top-k sims)
    where sims are (optionally recency-weighted) cosine similarities between the
    manuscript and each of the author's papers in the index.

    Parameters
    ----------
    index : PaperIndex
    authorships : DataFrame with columns work_id, author_id, author_name,
        institution_id, institution_name (position optional).
    papers : optional DataFrame with openalex_work_id, year (for recency weights).
    lam, k : aggregation parameters.
    recency_half_life : years; None disables recency weighting. Weight
        w = 0.5 ** ((ref_year - year) / half_life) shrinks a paper's similarity toward
        the query's mean similarity over the whole index: sim' = mean + w * (sim - mean).
        An 8-year-old paper therefore keeps 50% of its *advantage over an average paper*.
        (Multiplying the raw cosine instead is useless with dense encoders, whose cosines
        all sit in roughly [0.7, 1]: the weight then outweighs topic entirely.)
    min_papers : drop authors with fewer indexed papers.
    """

    def __init__(
        self,
        index: PaperIndex,
        authorships: pd.DataFrame,
        papers: Optional[pd.DataFrame] = None,
        lam: float = 0.3,
        k: int = 3,
        recency_half_life: Optional[float] = None,
        min_papers: int = 1,
        ref_year: Optional[int] = None,
    ):
        self.index = index
        self.lam = lam
        self.k = k
        self.recency_half_life = recency_half_life
        self.min_papers = min_papers

        a = authorships.copy()
        a["work_id"] = a["work_id"].astype(str)
        a["author_id"] = a["author_id"].astype(str)
        a = a[a["work_id"].isin(set(index.paper_ids))]
        self.authorships = a

        # author -> row positions in index
        self.author_rows: Dict[str, np.ndarray] = {}
        self.author_name: Dict[str, str] = {}
        self.author_inst: Dict[str, Set[str]] = defaultdict(set)
        self.author_inst_names: Dict[str, Set[str]] = defaultdict(set)
        self.coauthors: Dict[str, Set[str]] = defaultdict(set)
        self.paper_authors: Dict[str, Set[str]] = defaultdict(set)

        for r in a.itertuples(index=False):
            self.paper_authors[r.work_id].add(r.author_id)
            self.author_name.setdefault(r.author_id, getattr(r, "author_name", r.author_id))
            inst = getattr(r, "institution_id", None)
            if isinstance(inst, str) and inst:
                self.author_inst[r.author_id].add(inst)
                iname = getattr(r, "institution_name", None)
                if isinstance(iname, str) and iname:
                    self.author_inst_names[r.author_id].add(iname)
        rows = defaultdict(list)
        for wid, auths in self.paper_authors.items():
            pos = index.position(wid)
            for au in auths:
                rows[au].append(pos)
                self.coauthors[au].update(auths - {au})
        self.author_rows = {au: np.array(sorted(set(p)), dtype=np.int64) for au, p in rows.items()}

        # recency weights per index row
        self.row_weight = np.ones(len(index), dtype=np.float32)
        if recency_half_life and papers is not None and "year" in papers.columns:
            p = papers[["openalex_work_id", "year"]].dropna()
            years = dict(zip(p["openalex_work_id"].astype(str), p["year"].astype(float)))
            ref = float(ref_year) if ref_year else max(years.values(), default=0)
            for i, pid in enumerate(index.paper_ids):
                y = years.get(pid)
                if y is not None:
                    self.row_weight[i] = 0.5 ** (max(0.0, ref - y) / recency_half_life)

    # ------------------------------------------------------------------ API
    def rank(
        self,
        query_vec: np.ndarray,
        top_n: int = 20,
        exclude_papers: Optional[Iterable[str]] = None,
        manuscript_author_ids: Optional[Iterable[str]] = None,
        excluded_institution_ids: Optional[Iterable[str]] = None,
        exclude_coauthors: bool = True,
        n_evidence: int = 3,
    ) -> List[ReviewerCandidate]:
        sims = self.index.similarities(query_vec).astype(np.float32)
        if exclude_papers:
            for pid in exclude_papers:
                pos = self.index.position(pid)
                if pos is not None:
                    sims[pos] = np.nan
        if self.recency_half_life:
            base = float(np.nanmean(sims))
            wsims = base + (sims - base) * self.row_weight
        else:
            wsims = sims

        ms_authors = {str(x) for x in (manuscript_author_ids or [])}
        bad_inst = {str(x) for x in (excluded_institution_ids or [])}
        conflicted: Set[str] = set(ms_authors)
        if exclude_coauthors:
            for au in ms_authors:
                conflicted |= self.coauthors.get(au, set())

        out: List[ReviewerCandidate] = []
        for au, rows in self.author_rows.items():
            if au in conflicted:
                continue
            if bad_inst and (self.author_inst.get(au, set()) & bad_inst):
                continue
            s = wsims[rows]
            ok = ~np.isnan(s)
            if ok.sum() < self.min_papers or ok.sum() == 0:
                continue
            s, r = s[ok], rows[ok]
            order = np.argsort(-s)
            topk = s[order[: self.k]]
            score = (1 - self.lam) * float(s[order[0]]) + self.lam * float(topk.mean())
            ev = [(self.index.paper_ids[r[i]], float(sims[r[i]])) for i in order[:n_evidence]]
            out.append(
                ReviewerCandidate(
                    au, self.author_name.get(au, au), score, int(len(rows)), ev,
                    sorted(self.author_inst_names.get(au, set())),
                )
            )
        out.sort(key=lambda c: -c.score)
        return out[:top_n] if top_n else out

    def rank_frame(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.rank(*args, **kwargs)])
