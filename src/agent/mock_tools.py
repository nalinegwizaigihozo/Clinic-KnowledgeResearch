import re

# (disease)-[:ASSOCIATED_WITH]->(gene) and (drug)-[:TARGETS]->(gene). Sample data only.
ASSOCIATED_WITH = [
    ("Type 2 Diabetes", "TCF7L2"), ("Type 2 Diabetes", "PPARG"),
    ("Alzheimer's Disease", "APOE"),
]
TARGETS = [
    ("Pioglitazone", "PPARG"), ("Rosiglitazone", "PPARG"), ("Metformin", "PRKAA1"),
]

def mock_query_knowledge_graph(cypher):
    names = re.findall(r'name:\s*"([^"]+)"', cypher)
    name = names[0] if names else None
    if name is None:
        return {"ok": False, "results": [], "error": "mock graph: no {name: \"...\"} filter found in Cypher"}
    c = cypher.lower()
    rows = []
    if "(d:disease" in c and "(dr:drug" in c:
        for dis, gene in ASSOCIATED_WITH:
            if dis.lower() == name.lower():
                for drug, g2 in TARGETS:
                    if g2 == gene:
                        rows.append({"disease": dis, "gene": gene, "drug": drug})
    elif "(d:disease" in c:
        rows = [{"disease": d, "gene": g} for d, g in ASSOCIATED_WITH if d.lower() == name.lower()]
    elif "(dr:drug" in c:
        rows = [{"drug": d, "gene": g} for d, g in TARGETS if d.lower() == name.lower()]
    return {"ok": True, "results": rows, "error": None}

CORPUS = [
    {"id": "MOCK-001", "title": "Metformin and glycaemic control in type 2 diabetes (synthetic sample)",
     "snippet": "Sample abstract: metformin is widely described as a first-line option for type 2 diabetes, with trials reporting HbA1c reductions.",
     "source": "synthetic sample data", "url": None},
    {"id": "MOCK-002", "title": "PPARG agonists, pioglitazone and insulin sensitivity (synthetic sample)",
     "snippet": "Sample abstract: PPARG agonists such as pioglitazone improve insulin sensitivity; fluid retention is a noted adverse effect.",
     "source": "synthetic sample data", "url": None},
    {"id": "MOCK-003", "title": "TCF7L2 variants and type 2 diabetes risk (synthetic sample)",
     "snippet": "Sample abstract: common TCF7L2 variants are among the strongest genetic associations with type 2 diabetes risk.",
     "source": "synthetic sample data", "url": None},
    {"id": "MOCK-004", "title": "APOE e4 allele and Alzheimer's disease risk (synthetic sample)",
     "snippet": "Sample abstract: the APOE e4 allele is associated with increased Alzheimer's disease risk in cohort studies.",
     "source": "synthetic sample data", "url": None},
]

_STOP = {"with", "that", "this", "what", "which", "does", "about", "them", "from", "have", "were",
         "associated", "literature", "drugs", "genes", "target", "targets", "says", "trials", "clinical",
         "sample", "abstract", "common", "widely"}

def _tokens(s):
    return {t for t in re.findall(r"[a-z0-9']+", s.lower()) if len(t) > 3 and t not in _STOP}

def mock_search_medical_literature(query, k=4):
    q = _tokens(query)
    scored = []
    for doc in CORPUS:
        score = len(q & _tokens(doc["title"] + " " + doc["snippet"]))
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return {"ok": True, "results": [d for _, d in scored[:k]], "error": None}
