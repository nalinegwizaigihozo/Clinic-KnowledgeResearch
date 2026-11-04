import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.prompts import DISCLAIMER
from src.agent.tools import GRAPH, VECTOR

GRAPH_HINTS = ["target", "associated", "association", "pathway", "gene", "genes", "connect",
               "link between", "linked to", "related to", "interact", "which drugs", "mechanism of"]
TEXT_HINTS = ["study", "studies", "trial", "trials", "guideline", "evidence", "literature",
              "paper", "abstract", "efficacy", "outcome", "review", "say about", "published"]

DISEASES = {"type 2 diabetes": "Type 2 Diabetes", "diabetes": "Type 2 Diabetes",
            "alzheimer": "Alzheimer's Disease"}
DRUGS = {"metformin": "Metformin", "pioglitazone": "Pioglitazone", "rosiglitazone": "Rosiglitazone"}

@dataclass
class Decision:
    thought: str
    final: bool = False
    action: Optional[str] = None
    action_input: Dict[str, Any] = field(default_factory=dict)

class OfflineReActLLM:
    """Rule-based stand-in for an LLM. Same interface a real LLM adapter would expose."""

    # ---- helpers
    def _plan(self, q):
        ql = q.lower()
        g = any(h in ql for h in GRAPH_HINTS)
        t = any(h in ql for h in TEXT_HINTS)
        if g and t:
            return [GRAPH, VECTOR]
        if g:
            return [GRAPH]
        return [VECTOR]

    def _entities(self, q):
        ql = q.lower()
        drug = next((v for k, v in DRUGS.items() if k in ql), None)
        disease = next((v for k, v in DISEASES.items() if k in ql), None)
        if not disease and not drug:
            m = re.search(r"(?:associated with|linked to|related to)\s+([\w' -]+?)\s*(?:\?|,|\.|$|\band\b)", q, re.I)
            if m:
                disease = m.group(1).strip().title()
        return disease, drug

    def _cypher(self, q):
        disease, drug = self._entities(q)
        ql = q.lower()
        if disease:
            disease = disease.replace('"', "")
            if "drug" in ql:
                return (f'MATCH (d:Disease {{name: "{disease}"}})-[:ASSOCIATED_WITH]->(g:Gene)<-[:TARGETS]-(dr:Drug) '
                        f'RETURN d.name AS disease, g.name AS gene, dr.name AS drug LIMIT 25')
            return (f'MATCH (d:Disease {{name: "{disease}"}})-[:ASSOCIATED_WITH]->(g:Gene) '
                    f'RETURN d.name AS disease, g.name AS gene LIMIT 25')
        if drug:
            return (f'MATCH (dr:Drug {{name: "{drug}"}})-[:TARGETS]->(g:Gene) '
                    f'RETURN dr.name AS drug, g.name AS gene LIMIT 25')
        return None

    def _literature_query(self, q, steps):
        extra = []
        for s in steps:
            if s.action == GRAPH and s.observation.get("ok"):
                for row in s.observation["results"]:
                    for key in ("drug", "gene"):
                        if row.get(key) and row[key] not in extra:
                            extra.append(row[key])
        return (q + " " + " ".join(extra[:4])).strip()

    # ---- public interface
    def decide(self, question, steps, tool_names):
        used = [s.action for s in steps]
        for tool in self._plan(question):
            if tool in used or tool not in tool_names:
                continue
            if tool == GRAPH:
                cy = self._cypher(question)
                if cy is None:
                    continue  # cannot build a graph query -> fall through to literature
                return Decision("Relationship question: query the knowledge graph first.", action=GRAPH,
                                action_input={"cypher": cy})
            return Decision("Evidence needed: search the literature" +
                            (" using entities found in the graph." if GRAPH in used else "."),
                            action=VECTOR, action_input={"query": self._literature_query(question, steps), "k": 4})
        g_step = next((s for s in steps if s.action == GRAPH), None)
        if g_step and not g_step.observation.get("results") and VECTOR not in used and VECTOR in tool_names:
            return Decision("Graph returned nothing: fall back to literature search.", action=VECTOR,
                            action_input={"query": question, "k": 4})
        return Decision("Enough information gathered: write the answer.", final=True)

    def synthesize(self, question, steps, stopped_reason="final"):
        lines = ["## Research summary", f"**Question:** {question}", ""]
        refs, limits, found_any = [], [], False

        for s in steps:
            obs = s.observation
            label = "knowledge graph" if s.action == GRAPH else "literature search"
            if not obs.get("ok"):
                limits.append(f"The {label} step failed ({obs.get('error')}); results below are incomplete.")
                continue
            if not obs["results"]:
                limits.append(f"The {label} returned no results.")
                continue
            found_any = True
            if s.action == GRAPH:
                lines.append("### Knowledge-graph findings")
                for row in obs["results"]:
                    lines.append("- " + "; ".join(f"{k}: {v}" for k, v in row.items()))
                lines.append("")
            else:
                lines.append("### Literature evidence")
                for doc in obs["results"]:
                    refs.append(doc)
                    lines.append(f"- {doc.get('snippet', '')} [{len(refs)}]")
                lines.append("")

        if not found_any:
            lines.append("No supporting evidence was retrieved, so I can't answer this from the available sources.")
            lines.append("")
        if refs:
            lines.append("### References")
            for i, d in enumerate(refs, 1):
                link = d.get("url") or d.get("source") or "no link available"
                lines.append(f"[{i}] {d.get('title', 'Untitled')} - {link}")
            lines.append("")
        if stopped_reason == "max_steps":
            limits.append("Stopped at the step limit before finishing; the answer may be partial.")
        if any("synthetic" in str(d.get("source", "")) for d in refs):
            limits.append("Literature results are synthetic sample data, not real publications.")
        if limits:
            lines.append("### Limitations")
            lines.extend(f"- {x}" for x in limits)
            lines.append("")
        lines.append(DISCLAIMER)
        return "\n".join(lines)
