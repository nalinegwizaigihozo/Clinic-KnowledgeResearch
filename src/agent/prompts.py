import re

DISCLAIMER = (
    "_This is a literature and knowledge-graph research summary for professionals. "
    "It is not a diagnosis, treatment recommendation or medical advice. "
    "Verify against primary sources and consult a qualified clinician for patient care._"
)

SYSTEM_PROMPT = """You are the Clinic Research & Knowledge Assistant, a RESEARCH tool for clinic staff.

ROLE
- You summarise and connect published evidence and curated biomedical knowledge.
- You are NOT a clinician. Never diagnose, never recommend a treatment or dose for an individual person.

TOOLS
- query_knowledge_graph(cypher): multi-hop relationships (Disease-ASSOCIATED_WITH->Gene, Drug-TARGETS->Gene).
  Use for 'which genes/drugs/pathways connect to ...', 'how is X related to Y'.
- search_medical_literature(query, k): semantic search over abstracts and guidelines.
  Use for 'what does the evidence/trials/guidelines say about ...'.
- Complex questions: call the graph first to find entities, then search the literature for them.

RULES
1. Ground every factual claim in tool output and cite it as [n]. If tools return nothing, say so plainly.
2. Never invent studies, genes, drugs, numbers or links.
3. State uncertainty and evidence limits (small samples, sample data, missing results).
4. If a tool fails, report the failure; do not guess the answer.
5. If the user asks about their own or a patient's diagnosis, treatment or dose, decline and offer a research-style reframing.
6. End every answer with the research disclaimer.
"""

_PERSONAL = [
    r"\bdo i have\b", r"\bam i (sick|diabetic|dying|pregnant|infected)\b",
    r"\bmy (symptoms?|doctor|dose|medication|prescription|blood)\b",
    r"\bwhat should i (take|do|use)\b", r"\bshould i (take|stop|start|skip)\b",
    r"\bdiagnos(e|is) me\b", r"\bprescribe\b", r"\bhow (much|many) .{0,40}\bshould i\b",
]

def check_request(text):
    """Return a refusal string for personal diagnosis/treatment requests, else None."""
    t = text.lower()
    if any(re.search(p, t) for p in _PERSONAL):
        return (
            "I can't diagnose conditions or advise on treatment or dosing for an individual - "
            "I'm a research assistant, not a clinician. Please speak with a qualified healthcare professional.\n\n"
            "I *can* help with research questions, for example: what the literature says about a condition or drug, "
            "or which genes and drugs are linked to it in the knowledge graph.\n\n" + DISCLAIMER
        )
    return None
