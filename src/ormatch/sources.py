"""Seed journal -> OpenAlex source ID table."""

SOURCES: dict[str, str] = {
    "S125775545": "Operations Research",
    "S33323087": "Management Science",
    "S81410195": "Manufacturing & Service Operations Management",
    "S193920097": "Mathematical Programming",
    "S55826652": "Mathematics of Operations Research",
    "S149070780": "Production and Operations Management",
    "S165318533": "INFORMS Journal on Computing",
    "S130252234": "Transportation Science",
    "S4210232382": "Stochastic Systems",
    "S928796702": "SIAM Journal on Optimization",
    "S182666042": "Naval Research Logistics",
    "S4210225672": "IISE Transactions",
    "S63835026": "IIE Transactions",
    "S103321696": "European Journal of Operational Research",
    "S142306484": "Journal of Operations Management",
    "S177792750": "Decision Sciences",
    "S27769002": "Operations Research Letters",
    # applied / computational OR venues (added 2026-09-18; they are core OR, not an add-on)
    "S169988927": "Journal of the Operational Research Society",
    "S173256270": "Computers & Operations Research",
    "S57667410": "Annals of Operations Research",
    "S4210190151": "Omega",
    "S184816971": "International Journal of Production Economics",
    "S65690446": "International Journal of Production Research",
    "S196821226": "Computers & Industrial Engineering",
    "S96305778": "Transportation Research Part B: Methodological",
    "S173966628": "Transportation Research Part E: Logistics and Transportation Review",
    "S52430896": "Journal of Optimization Theory and Applications",
    "S897311980": "SIAM Journal on Control and Optimization",
    "S191798613": "Networks",
}

# Short aliases for CLI convenience.
ALIASES: dict[str, str] = {
    "OR": "S125775545",
    "MS": "S33323087",
    "MSOM": "S81410195",
    "MP": "S193920097",
    "MOR": "S55826652",
    "POM": "S149070780",
    "IJOC": "S165318533",
    "TS": "S130252234",
    "SS": "S4210232382",
    "SIOPT": "S928796702",
    "NRL": "S182666042",
    "IISE": "S4210225672",
    "IIE": "S63835026",
    "EJOR": "S103321696",
    "JOM": "S142306484",
    "DS": "S177792750",
    "ORL": "S27769002",
    "NETWORKS": "S191798613",
    "SICON": "S897311980",
    "JOTA": "S52430896",
    "TRE": "S173966628",
    "TRB": "S96305778",
    "CAIE": "S196821226",
    "IJPR": "S65690446",
    "IJPE": "S184816971",
    "OMEGA": "S4210190151",
    "AOR": "S57667410",
    "COR": "S173256270",
    "JORS": "S169988927",
}


# Add-on collections. Each becomes its own self-contained index directory that users can
# download in addition to the core OR/MS index (see docs and `ormatch suggest --index-dir`).
# Source IDs are OpenAlex sources. Chosen from what indexed OR papers cite most outside the
# core venues (300-paper sample, 2026-09): econ/finance ~6% of references, statistics/ML ~2%.
# Applied-OR venues (another ~6%) were folded into the core list above.
COLLECTIONS: dict[str, dict[str, str]] = {
    "core": SOURCES,
    "econ-finance": {
        "S5353659": "The Journal of Finance",
        "S149240962": "Journal of Financial Economics",
        "S170137484": "Review of Financial Studies",
        "S95464858": "Econometrica",
        "S23254222": "American Economic Review",
        "S203860005": "The Quarterly Journal of Economics",
        "S95323914": "Journal of Political Economy",
        "S163534328": "Marketing Science",
        "S119950638": "Journal of Marketing Research",
    },
    "stats-ml": {
        "S118988714": "Journal of Machine Learning Research",
        "S4394736638": "Journal of the American Statistical Association",
        "S119757635": "The Annals of Statistics",
        "S145009937": "Journal of the Royal Statistical Society Series B",
        "S172180718": "Biometrika",
        "S9093621": "The Annals of Applied Probability",
    },
    "algorithms": {
        "S118992489": "Journal of the ACM",
        "S153560523": "SIAM Journal on Computing",
        "S102439543": "Mathematics of Computation",
    },
}

ALL_SOURCES: dict[str, str] = {sid: name for coll in COLLECTIONS.values() for sid, name in coll.items()}


def collection_sources(name: str) -> dict[str, str]:
    if name not in COLLECTIONS:
        raise KeyError(f"unknown collection {name!r}; known: {sorted(COLLECTIONS)}")
    return COLLECTIONS[name]


def resolve_source(s: str) -> str:
    """Accept an OpenAlex source ID (with or without URL prefix) or an alias."""
    s = s.strip()
    if s.upper() in ALIASES:
        return ALIASES[s.upper()]
    if s.startswith("https://openalex.org/"):
        s = s.rsplit("/", 1)[-1]
    return s.upper()


def source_name(source_id: str) -> str:
    return ALL_SOURCES.get(source_id, source_id)
