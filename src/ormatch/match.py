"""Reviewer scoring: aggregate paper-level similarities per author, with conflict filtering."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import re
import unicodedata

import numpy as np
import pandas as pd

from .index import PaperIndex


def normalize_name(name: str) -> str:
    """'J. Q. Public-Smith' / 'Public, Jane' -> 'jane public smith'-ish key: ascii, lowercase,
    initials kept as single letters, no punctuation, tokens sorted so order does not matter."""
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-zA-Z ]", " ", n.replace(",", " ")).lower()
    toks = [t for t in n.split() if t]
    return " ".join(sorted(toks))


def name_key_loose(name: str) -> str:
    """surname + first initial, for matching 'J. Smith' against 'Jane Smith'."""
    toks = normalize_name(name).split()
    if not toks:
        return ""
    # after sorting we cannot tell the surname; use the longest token as surname proxy plus
    # the initials of the others
    surname = max(toks, key=len)
    initials = "".join(sorted(t[0] for t in toks if t != surname))
    return f"{surname}:{initials}"


@dataclass
class ReviewerCandidate:
    author_id: str
    author_name: str
    score: float
    n_papers: int
    evidence: List[Tuple[str, float]] = field(default_factory=list)  # top-3 (paper_id, sim)
    institutions: List[str] = field(default_factory=list)
    n_cited: int = 0  # of this author's indexed papers, how many the manuscript cites
    n_editor_roles: int = 0  # current editorial positions known from data/editors.csv
    seniority: Optional[float] = None  # works_count from data/authors.parquet, if loaded
    or_links: Optional[int] = None  # core-venue papers + citation links to core (add-on authors)
    components: Dict[str, float] = field(default_factory=dict)  # score = sum of these terms

    def to_dict(self) -> Dict:
        return {
            "author_id": self.author_id,
            "author_name": self.author_name,
            "score": round(self.score, 4),
            "n_papers": self.n_papers,
            "evidence": [{"paper_id": p, "sim": round(s, 4)} for p, s in self.evidence],
            "institutions": self.institutions,
            "n_cited": self.n_cited,
            "n_editor_roles": self.n_editor_roles,
            "seniority": self.seniority,
            "or_links": self.or_links,
            "components": {k: round(v, 4) for k, v in self.components.items()},
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

        self.paper_year: Dict[str, float] = {}
        if papers is not None and "year" in papers.columns:
            py = papers[["openalex_work_id", "year"]].dropna()
            self.paper_year = dict(zip(py["openalex_work_id"].astype(str), py["year"].astype(float)))
        self.name_index: Dict[str, Set[str]] = defaultdict(set)   # normalized full name -> ids
        self.loose_index: Dict[str, Set[str]] = defaultdict(set)  # surname:initials -> ids
        self.editor_roles: Dict[str, int] = {}
        self.author_stats: Dict[str, float] = {}
        self.or_links: Dict[str, int] = {}
        for r in a.itertuples(index=False):
            self.paper_authors[r.work_id].add(r.author_id)
            if r.author_id not in self.author_name:
                nm = getattr(r, "author_name", None)
                nm = nm if isinstance(nm, str) and nm else r.author_id
                self.author_name[r.author_id] = nm
                self.name_index[normalize_name(nm)].add(r.author_id)
                self.loose_index[name_key_loose(nm)].add(r.author_id)
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
        # flat layout for vectorised per-author max: rows concatenated, with segment starts
        self._authors_flat = list(self.author_rows)
        lens = np.array([len(self.author_rows[a]) for a in self._authors_flat], dtype=np.int64)
        self._flat_rows = np.concatenate([self.author_rows[a] for a in self._authors_flat]) if lens.size else np.zeros(0, np.int64)
        self._flat_starts = np.concatenate([[0], np.cumsum(lens)[:-1]]) if lens.size else np.zeros(0, np.int64)

        self.ref_year = ref_year
        self.set_recency(recency_half_life, papers)

    def set_recency(self, half_life: Optional[float], papers: Optional[pd.DataFrame]) -> None:
        """(Re)compute per-row recency weights; cheap, so the UI can flip it live."""
        self.recency_half_life = half_life
        self.row_weight = np.ones(len(self.index), dtype=np.float32)
        if half_life and papers is not None and "year" in papers.columns:
            p = papers[["openalex_work_id", "year"]].dropna()
            years = dict(zip(p["openalex_work_id"].astype(str), p["year"].astype(float)))
            ref = float(self.ref_year) if self.ref_year else max(years.values(), default=0)
            for i, pid in enumerate(self.index.paper_ids):
                y = years.get(pid)
                if y is not None:
                    self.row_weight[i] = 0.5 ** (max(0.0, ref - y) / half_life)

    # ------------------------------------------------------------------ COI helpers
    def resolve_authors(self, names_or_ids: Iterable[str]) -> Dict[str, List[str]]:
        """Map manuscript author names (or OpenAlex ids) to author ids in the index.
        Exact normalized-name match first; falls back to surname + initials."""
        out: Dict[str, List[str]] = {}
        for q in names_or_ids:
            q = (q or "").strip()
            if not q:
                continue
            if re.fullmatch(r"A\d{4,}", q):
                out[q] = [q]
                continue
            ids = sorted(self.name_index.get(normalize_name(q), set()))
            if not ids:
                ids = sorted(self.loose_index.get(name_key_loose(q), set()))
            out[q] = ids
        return out

    def conflicts_for(self, author_ids: Iterable[str], coauthor_years: Optional[float] = None,
                      same_institution: bool = True, extra_pairs: Optional[Iterable[Tuple[str, str]]] = None,
                      ref_year: Optional[float] = None) -> Dict[str, str]:
        """Conflicted reviewer ids -> reason. Co-authors (optionally only on papers from the last
        `coauthor_years` years), same current institution, and any extra name pairs
        (e.g. advisor/advisee from data/coi/genealogy.csv) whose either side is a manuscript author."""
        ids = {str(x) for x in author_ids}
        out: Dict[str, str] = {}
        ref = ref_year or (max(self.paper_year.values()) if self.paper_year else None)
        for au in ids:
            out[au] = "manuscript author"
            for pid in (self.index.paper_ids[i] for i in self.author_rows.get(au, [])):
                if coauthor_years and ref is not None:
                    y = self.paper_year.get(pid)
                    if y is not None and ref - y > coauthor_years:
                        continue
                for co in self.paper_authors.get(pid, ()):
                    if co not in ids:
                        out.setdefault(co, f"co-author of {self.author_name.get(au, au)}")
            if same_institution:
                insts = self.author_inst.get(au, set())
                if insts:
                    for other, oi in self.author_inst.items():
                        if other not in ids and oi & insts:
                            out.setdefault(other, f"same institution as {self.author_name.get(au, au)}")
        if extra_pairs:
            names = {normalize_name(self.author_name.get(au, "")) for au in ids}
            for a_name, b_name in extra_pairs:
                na, nb = normalize_name(a_name), normalize_name(b_name)
                if na in names or nb in names:
                    other = b_name if na in names else a_name
                    for oid in self.resolve_authors([other]).get(other, []):
                        if oid not in ids:
                            out.setdefault(oid, f"genealogy link with {a_name if nb in names else b_name}")
        return out

    def set_editor_roles(self, roles: Dict[str, int]) -> None:
        self.editor_roles = dict(roles)

    def set_author_stats(self, stats: Dict[str, float]) -> None:
        """author_id -> works_count (or any seniority proxy); larger = more senior."""
        self.author_stats = dict(stats)

    def compute_or_links(self, core_paper_ids: Set[str], references: Optional[pd.DataFrame]) -> None:
        """For every author: number of their papers in the core collection, plus their papers
        citing core papers, plus core papers citing theirs. Authors with 0 are outsiders."""
        cited_core: Dict[str, int] = defaultdict(int)   # paper -> n core papers it cites
        cited_by_core: Dict[str, int] = defaultdict(int)  # paper -> n core papers citing it
        if references is not None and len(references):
            r = references
            r1 = r[r["referenced_work_id"].isin(core_paper_ids)]
            for w, c in r1.groupby("work_id").size().items():
                cited_core[str(w)] = int(c)
            r2 = r[r["work_id"].isin(core_paper_ids)]
            for w, c in r2.groupby("referenced_work_id").size().items():
                cited_by_core[str(w)] = int(c)
        self.or_links = {}
        for au, rows in self.author_rows.items():
            n = 0
            for i in rows:
                pid = self.index.paper_ids[i]
                n += (pid in core_paper_ids) + min(cited_core.get(pid, 0), 3) + min(cited_by_core.get(pid, 0), 3)
            self.or_links[au] = int(n)

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
        cited_papers: Optional[Iterable[str]] = None,
        cite_weight: float = 0.0,
        exclude_ids: Optional[Iterable[str]] = None,
        editor_weight: float = 0.0,
        seniority_weight: float = 0.0,
        min_or_links: int = 0,
        volume_correction: float = 0.0,
        early_career_weight: float = 0.0,
        early_career_max_works: int = 15,
        seed_author_ids: Optional[Iterable[str]] = None,
        seed_weight: float = 0.0,
        diversity: float = 0.0,
        pool_factor: int = 5,
    ) -> List[ReviewerCandidate]:
        """Rank authors for one query vector.

        cited_papers : index paper ids that the manuscript cites (from its bibliography).
            Each author gets + cite_weight * (1 - 0.5 ** n_cited) added to the similarity
            score, i.e. half the weight for one cited paper, three quarters for two, ...
            This is how editors find reviewers by hand; it needs no full text on the index side.
        """
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
        conflicted: Set[str] = set(ms_authors) | {str(x) for x in (exclude_ids or [])}
        if exclude_coauthors:
            for au in ms_authors:
                conflicted |= self.coauthors.get(au, set())
        # "usual suspects" correction: an author with n indexed papers gets n draws at a high
        # cosine, so the expected best-of-n for *random* papers rises with n. Subtract
        # volume_correction * (E[max of n] - mean), estimated from this query's own similarity
        # distribution (normal approximation), so a 2-paper author with one very close paper can
        # outrank a 40-paper author whose best paper is only moderately close.
        exp_gain = None
        if volume_correction:
            finite = sims[np.isfinite(sims)]
            mu, sd = float(np.mean(finite)), float(np.std(finite)) + 1e-9
            from scipy.stats import norm as _norm
            ns = np.arange(1, 2001)
            exp_gain = sd * _norm.ppf(ns / (ns + 1.0))  # E[max of n] - mu, n = 1..2000
            exp_gain -= exp_gain[0]  # a single paper is the reference point (no penalty)
        # seniority penalty: log works_count relative to the median author, in cosine units
        sen_med = None
        if seniority_weight and self.author_stats:
            sen_med = float(np.median(np.log1p(list(self.author_stats.values()))))

        cited_count: Dict[str, int] = defaultdict(int)
        if cited_papers and cite_weight:
            for pid in set(map(str, cited_papers)):
                for au in self.paper_authors.get(pid, ()):
                    cited_count[au] += 1

        # Candidate pool: the exact per-author score needs a Python loop, so restrict it to the
        # authors whose best (weighted) paper is in the top `pool` by a vectorised max, plus
        # anyone the manuscript cites or a seed resembles (bonuses can lift them in). Penalties
        # only lower scores, so nobody outside the pool could have ranked above it.
        flat = wsims[self._flat_rows]
        flat_nan = np.where(np.isnan(flat), -np.inf, flat)
        best = np.maximum.reduceat(flat_nan, self._flat_starts) if len(self._flat_starts) else np.zeros(0)
        pool = max(2000, 100 * (top_n or 20))
        if len(best) > pool:
            idx = np.argpartition(-best, pool - 1)[:pool]
        else:
            idx = np.arange(len(best))
        candidates = {self._authors_flat[i] for i in idx if np.isfinite(best[i])}
        candidates |= set(cited_count)
        if seed_author_ids and seed_weight:
            candidates |= set(self.author_rows)  # seed bonus applies to everyone; keep exact
        out: List[ReviewerCandidate] = []
        for au in candidates:
            rows = self.author_rows[au]
            if au in conflicted:
                continue
            if bad_inst and (self.author_inst.get(au, set()) & bad_inst):
                continue
            if min_or_links and self.or_links and self.or_links.get(au, 0) < min_or_links:
                continue
            s = wsims[rows]
            ok = ~np.isnan(s)
            if ok.sum() < self.min_papers or ok.sum() == 0:
                continue
            s, r = s[ok], rows[ok]
            order = np.argsort(-s)
            topk = s[order[: self.k]]
            comp: Dict[str, float] = {
                "best paper (1-lam)*max": (1 - self.lam) * float(s[order[0]]),
                "top-k mean lam*mean": self.lam * float(topk.mean()),
            }
            nc = cited_count.get(au, 0)
            if nc:
                comp["cited bonus"] = cite_weight * (1 - 0.5 ** nc)
            if exp_gain is not None:
                comp["volume correction"] = -volume_correction * float(exp_gain[min(int(ok.sum()), len(exp_gain)) - 1])
            ne = self.editor_roles.get(au, 0)
            if ne and editor_weight:
                comp["editor penalty"] = -editor_weight * ne
            sen = self.author_stats.get(au)
            if sen is not None and seniority_weight and sen_med is not None:
                comp["seniority penalty"] = -seniority_weight * max(0.0, float(np.log1p(sen)) - sen_med)
            if sen is not None and early_career_weight:
                # bonus that fades linearly in log(works): full at 1 work, zero at early_career_max_works
                frac = 1.0 - float(np.log1p(sen)) / float(np.log1p(early_career_max_works))
                if frac > 0:
                    comp["early-career bonus"] = early_career_weight * frac
            score = float(sum(comp.values()))
            ev = [(self.index.paper_ids[r[i]], float(sims[r[i]])) for i in order[:n_evidence]]
            out.append(
                ReviewerCandidate(
                    au, self.author_name.get(au, au), score, int(len(rows)), ev,
                    sorted(self.author_inst_names.get(au, set())), nc, ne,
                    self.author_stats.get(au), self.or_links.get(au) if self.or_links else None, comp,
                )
            )
        out.sort(key=lambda c: -c.score)

        seeds = [str(x) for x in (seed_author_ids or []) if str(x) in self.author_rows]
        if not (seeds and seed_weight) and not diversity:
            return out[:top_n] if top_n else out

        # --- seeds and diversity work on author profiles (mean of their paper embeddings) ---
        pool = out[: max(top_n * pool_factor, 50)] if top_n else out
        def profile(au: str) -> np.ndarray:
            v = self.index.embeddings[self.author_rows[au]].mean(axis=0)
            n = float(np.linalg.norm(v)) or 1.0
            return v / n
        P = np.stack([profile(c.author_id) for c in pool]) if pool else np.zeros((0, self.index.embeddings.shape[1]))
        if seeds and seed_weight:
            # bonus = seed_weight * best cosine to any seed profile: "people like these"
            Sd = np.stack([profile(a) for a in seeds])
            seed_sim = (P @ Sd.T).max(axis=1)
            # profile cosines cluster near 1 for everyone in the pool, so use the deviation from
            # the pool median: candidates *unusually* like a seed gain, unlike ones lose
            seed_sim = seed_sim - float(np.median(seed_sim))
            for c, ss in zip(pool, seed_sim):
                c.components["seed bonus (vs pool median)"] = seed_weight * float(ss)
                c.score += seed_weight * float(ss)
            pool.sort(key=lambda c: -c.score)
            P = np.stack([profile(c.author_id) for c in pool])
        if diversity and pool:
            # maximal marginal relevance: greedily pick the candidate maximising
            # (1 - d) * score - d * max cosine to (seeds + already picked), so the list covers
            # different neighbourhoods instead of the seeds' clones
            d = float(min(max(diversity, 0.0), 1.0))
            chosen: List[int] = []
            taken = [profile(a) for a in seeds]
            scores = np.array([c.score for c in pool], dtype=np.float32)
            remaining = list(range(len(pool)))
            while remaining and len(chosen) < (top_n or len(pool)):
                if taken:
                    T = np.stack(taken)
                    red = (P[remaining] @ T.T).max(axis=1)
                else:
                    red = np.zeros(len(remaining), dtype=np.float32)
                mmr = (1 - d) * scores[remaining] - d * red
                j = remaining[int(np.argmax(mmr))]
                chosen.append(j); remaining.remove(j); taken.append(P[j])
            pool = [pool[j] for j in chosen]
        return pool[:top_n] if top_n else pool

    def rank_frame(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.rank(*args, **kwargs)])
