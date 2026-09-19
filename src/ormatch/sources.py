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
}


def resolve_source(s: str) -> str:
    """Accept an OpenAlex source ID (with or without URL prefix) or an alias."""
    s = s.strip()
    if s.upper() in ALIASES:
        return ALIASES[s.upper()]
    if s.startswith("https://openalex.org/"):
        s = s.rsplit("/", 1)[-1]
    return s.upper()


def source_name(source_id: str) -> str:
    return SOURCES.get(source_id, source_id)
