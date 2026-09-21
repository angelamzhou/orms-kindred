# Reviewer matching: how existing systems score and weight, and where Kindred's weights come from

*Compiled 2026-09-18 while building Kindred. Numbers quoted from the sources are as reported
there; Kindred numbers are leave-one-out on the 2014+ OR/MS index (31,649 papers, n=200 sample).*

## 1. The question

Kindred scores an author for a manuscript as

    score = (1 - lam) * max_i sim_i + lam * mean(top-k sim_i) + cite_weight * (1 - 0.5^n_cited)

where `sim_i` is the cosine between the manuscript's title+abstract embedding and the author's
i-th indexed paper, and `n_cited` is how many of the author's indexed papers the manuscript's
bibliography cites. This note records where each piece comes from and how deployed systems make
the same choices.

## 2. Systems and what they actually do

### TPMS, the Toronto Paper Matching System (Charlin & Zemel, 2013)
- **Reviewer profile.** A reviewer's papers ("archive") are *concatenated into a single document*;
  the archive and the submission are both mapped to features `f(.)`: word counts, or LDA topic
  proportions (30 topics for ICML; per-paper topics averaged in topic space).
- **Unsupervised score.** Dot product `s_rp = f(A_r) f(P_p)'` (their eq. 1); KL-divergence is
  mentioned as an alternative.
- **Supervised score.** Once reviewers have rated some papers, a per-reviewer linear regression on
  submission word counts (eq. 4), and for ICML-12 a shared+individual model
  `s_rp = b + b_r + (theta + theta_p) f(P_p)' + (omega + omega_r) g(A_r)'` (eq. 5) fit with squared
  loss and L2 regularisation; also probabilistic matrix factorisation. Nonlinear/ordinal variants
  "did not offer any significant performance gains".
- **Combination with other signals.** TPMS returns scores; CMT's matcher combines them linearly
  with reviewer bids and subject-area scores. No citation signal; the paper cites Rodriguez &
  Bollen (2008) as prior work that uses references.
- **Takeaway for Kindred.** Weights are either fixed linear mixes or learned *only* from elicited
  reviewer scores, which a journal-side tool does not have.

### OpenReview expertise model (SPECTER + MFR)
- **Pooling.** Configurable `model_params.max_score` (default true) or `average_score`: the
  reviewer-paper affinity is the max (or mean) cosine between the submission and the reviewer's
  publication embeddings. Exactly one must be on.
- **Ensemble.** `merge_alpha = 0.8`: affinity = 0.8 * SPECTER + 0.2 * MFR, a fixed linear mix.
- **Inputs.** Titles and abstracts only. Used at NeurIPS/ICLR/ICML scale.
- **Takeaway.** Max pooling as the default and a hand-set linear ensemble weight.

### ACL / ACL Rolling Review matcher
- Contrastive encoder trained on ACL Anthology abstracts; reviewer score = weighted sum of the
  **top-3** cosines with weights 1, 1/2, 1/3 (as described in Stelmakh et al.). This is the same
  diminishing-returns idea as Kindred's `lam`/`k` mixture and its saturating citation bonus.

### Rodriguez & Bollen (2008), "An algorithm to determine peer-reviewers"
- Builds a co-authorship network starting from the **authors of the manuscript's references**,
  expands to co-authors of co-authors, and runs a relative-rank particle-swarm walk to rank experts.
  Not limited to a preselected pool; can surface conflicts of interest.
- **Takeaway.** The bibliography as the primary reviewer signal has a 2008 pedigree; Kindred's
  `cite_weight` is a one-hop, linear version of this idea combined with text similarity.

### Stelmakh, Wieting, Xing & Shah, "A Gold Standard Dataset for the Reviewer Assignment Problem"
(arXiv 2303.16750; TMLR 2025 version)
- 477 self-reported expertise scores from 58 researchers; algorithms judged by how often they
  mis-order pairs ("easy" and "hard" triples).
- Reported accuracy (easy / hard): TPMS title+abstract 80/62, TPMS full text 84/64; ELMo 70/57;
  SPECTER 85/57; SPECTER+MFR 88/60; SPECTER2 with max pooling 89/61; ACL 78/62; off-the-shelf
  LLMs 52-82 / 41-57.
- **Pooling matters:** SPECTER2 loss 0.22 with max pooling vs 0.25 with mean pooling.
- **Full text helps modestly:** TPMS loss 0.27 (title+abstract) -> 0.24 (full text), close to but
  not better than SPECTER2 at 0.22. Classical TF-IDF with full text matches SPECTER2.
- All methods err on 12-30% of easy and 36-43% of hard triples.
- **Takeaway.** Abstract-based dense encoders with max pooling are the state of the art; full
  text on the index side buys a few points at most; no system combines a citation signal.

### Bibliographic coupling and citation-based recommendation
- Kessler's bibliographic coupling (shared references) and co-citation are the classical
  citation similarities; recent work (section-based and passage-based coupling, citation
  proximity) reports gains over text-only similarity for paper recommendation; ReviewerNet
  visualises citation and authorship relations for reviewer finding.
- **Takeaway.** With `referenced_works` now in the index (125k in-corpus pairs), bibliographic
  coupling between the manuscript and index papers is a natural next signal beyond the one-hop
  cited-author bonus.

## 3. What this implies for Kindred's weights

| weight | default | provenance |
|---|---|---|
| `lam` (max vs top-k mean) | 0.3 | Max/mean pooling is the universal design knob (OpenReview, TPMS, Stelmakh). Our LOO: lam 0 (pure max) MRR 0.214 vs 0.193 at 0.3; lam 0.5 and 0.7 worse. Kept at 0.3 pending a larger eval. |
| `k` | 3 | ACL's top-3 (1, 1/2, 1/3). k=5 did not help in our grid. |
| `cite_weight` | 0.1 | Our addition (Rodriguez & Bollen lineage). Saturating form `1 - 0.5^n` is a design choice. LOO with the held-out paper's OpenAlex references as bibliography: MRR 0.196 -> 0.295 (0.05) -> 0.317 (0.1) -> 0.320 (0.2); recall@10 0.259 -> 0.324; recall@20 0.303 -> 0.396. Plateau from 0.1. |
| `half_life` | off | Not in any of the systems above; a product preference for active reviewers. Implemented as shrinkage toward the query's mean similarity (multiplying raw cosines is meaningless for dense encoders whose cosines sit in [0.7, 1]). Costs LOO accuracy (MRR 0.070 at 8 years). |
| ensemble | linear | Every deployed system mixes signals linearly with hand-set weights; weights are learned only where elicited reviewer ratings exist. Hence Kindred exposes them (CLI flags, Streamlit sliders) instead of fixing them. |

## 4. Open questions
1. A human-judged evaluation on OR manuscripts (editor-chosen reviewers or self-rated expertise,
   as in Stelmakh et al.) is the only way to compare with TPMS-class systems; leave-one-out
   authorship is a proxy that rewards finding the exact author.
2. Bibliographic coupling (shared references between manuscript and index papers) as a third
   signal, now that in-corpus references are indexed.
3. Dense + BM25 fusion on title/abstract; author-profile (mean embedding) pooling.
4. Whether the LOO gain from citations transfers to real PDFs: the eval bibliography is
   OpenAlex's clean reference list; the local parser matched 13 of 120 entries on the sample paper.

## Sources
- Charlin, L., Zemel, R. (2013). The Toronto Paper Matching System. https://www.cs.toronto.edu/~lcharlin/papers/tpms.pdf
- OpenReview expertise model README. https://github.com/openreview/openreview-expertise/blob/master/README.md
- Stelmakh, I., Wieting, J., Xing, E., Shah, N. A Gold Standard Dataset for the Reviewer Assignment Problem. https://arxiv.org/abs/2303.16750
- Rodriguez, M. A., Bollen, J. (2008). An algorithm to determine peer-reviewers. CIKM. https://dl.acm.org/doi/10.1145/1458082.1458127
- Leyton-Brown, K. et al. (2022). Matching Papers and Reviewers at Large Conferences. https://arxiv.org/abs/2202.12273
- ReviewerNet: Visualizing Citation and Authorship Relations for Finding Reviewers. https://arxiv.org/abs/1903.08004
- Sections-based bibliographic coupling for research paper recommendation. Scientometrics 2019. https://link.springer.com/article/10.1007/s11192-019-03053-8
