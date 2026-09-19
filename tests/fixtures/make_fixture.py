"""Generate a small synthetic OR-flavoured corpus: ~200 papers, ~50 authors.

Authors are assigned to 1-2 topics; papers are built from their topic's vocabulary so
that leave-one-out eval has real signal. Writes papers.parquet + authorships.parquet here.
"""
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

TOPICS = {
    "inventory": ["inventory control", "base-stock policy", "lost sales", "newsvendor", "lead time",
                  "multi-echelon", "safety stock", "demand forecasting", "perishable goods", "(s,S) policy"],
    "queueing": ["queueing network", "M/M/1", "heavy traffic", "fluid limit", "service system", "call center",
                 "abandonment", "staffing", "many-server queue", "diffusion approximation"],
    "integer": ["mixed-integer programming", "branch-and-cut", "valid inequalities", "Benders decomposition",
                "cutting planes", "facility location", "lot sizing", "polyhedral study", "column generation", "MIP heuristics"],
    "robust": ["robust optimization", "distributionally robust", "Wasserstein ambiguity", "uncertainty set",
               "adjustable robust", "worst-case", "moment-based", "chance constraint", "tractable reformulation", "conic duality"],
    "revenue": ["revenue management", "dynamic pricing", "choice model", "assortment optimization", "MNL",
                "network revenue", "bid price", "customer choice", "markdown", "capacity control"],
    "healthcare": ["healthcare operations", "appointment scheduling", "operating room", "patient flow",
                   "emergency department", "kidney exchange", "hospital capacity", "no-show", "wait time", "triage"],
    "platform": ["two-sided platform", "ride-hailing", "matching market", "surge pricing", "gig economy",
                 "online marketplace", "network effects", "spatial matching", "driver supply", "dispatch policy"],
    "learning": ["multi-armed bandit", "online learning", "regret bound", "Thompson sampling", "reinforcement learning",
                 "Markov decision process", "approximate dynamic programming", "policy gradient", "contextual bandit", "exploration"],
}
VERBS = ["Optimal", "Dynamic", "Robust", "Data-Driven", "Approximate", "Near-Optimal", "Asymptotically Optimal", "Stochastic"]
TAIL = ["with Applications to Supply Chains", "under Demand Uncertainty", "in Large-Scale Systems",
        "via Lagrangian Relaxation", "with Learning", "for Service Operations", "with Fairness Constraints", ""]
VENUES = ["Operations Research", "Management Science", "M&SOM", "Mathematics of Operations Research", "POM"]
INSTS = [("I1", "Cornell"), ("I2", "MIT"), ("I3", "Stanford"), ("I4", "Columbia"), ("I5", "Georgia Tech"),
         ("I6", "Wharton"), ("I7", "Northwestern"), ("I8", "Berkeley"), ("I9", "Michigan"), ("I10", "CMU")]


def main(n_papers=200, n_authors=50, seed=0):
    rng = random.Random(seed)
    tnames = list(TOPICS)
    authors = []
    for i in range(n_authors):
        prim = rng.choice(tnames)
        sec = rng.choice(tnames) if rng.random() < 0.4 else prim
        inst = rng.choice(INSTS)
        authors.append({"id": f"A{i:03d}", "name": f"Author {i:03d}", "topics": [prim, sec], "inst": inst})
    by_topic = {t: [a for a in authors if t in a["topics"]] for t in tnames}

    papers, auths = [], []
    for j in range(n_papers):
        t = rng.choice(tnames)
        vocab = TOPICS[t]
        terms = rng.sample(vocab, 4)
        title = f"{rng.choice(VERBS)} {terms[0].title()} for {terms[1].title()} {rng.choice(TAIL)}".strip()
        abstract = (f"We study {terms[0]} in the context of {terms[1]}. Motivated by {terms[2]}, we propose a "
                    f"{rng.choice(VERBS).lower()} policy and analyse it using {terms[3]}. "
                    f"Numerical experiments on {rng.choice(vocab)} instances show {rng.randint(3, 30)}% improvement "
                    f"over benchmarks. Extensions to {rng.choice(vocab)} are discussed.")
        pid = f"W{j:04d}"
        year = rng.randint(2008, 2025)
        papers.append({"openalex_work_id": pid, "doi": f"10.1000/{pid}", "title": title, "abstract": abstract,
                       "year": year, "source_id": "S1", "venue": rng.choice(VENUES), "oa_pdf_url": None,
                       "cited_by_count": rng.randint(0, 200)})
        pool = by_topic[t]
        k = min(len(pool), rng.choice([1, 2, 2, 3, 3, 4]))
        for pos, a in enumerate(rng.sample(pool, k)):
            auths.append({"work_id": pid, "author_id": a["id"], "author_name": a["name"], "position": pos,
                          "institution_id": a["inst"][0], "institution_name": a["inst"][1]})
    pd.DataFrame(papers).to_parquet(os.path.join(HERE, "papers.parquet"), index=False)
    pd.DataFrame(auths).to_parquet(os.path.join(HERE, "authorships.parquet"), index=False)
    print(f"wrote {len(papers)} papers, {len(auths)} authorships, {n_authors} authors to {HERE}")


if __name__ == "__main__":
    main()
