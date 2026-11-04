import importlib
from dataclasses import dataclass
from typing import Any, Callable

from src.agent.mock_tools import mock_query_knowledge_graph, mock_search_medical_literature

GRAPH = "query_knowledge_graph"
VECTOR = "search_medical_literature"

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]

    def run(self, **kwargs):
        """Always returns {ok, results, error}; never raises (tool failures become observations)."""
        try:
            out = self.func(**kwargs)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "results": [], "error": f"{type(e).__name__}: {e}"}
        if not isinstance(out, dict) or "results" not in out:
            out = {"results": out if isinstance(out, list) else [out]}
        out.setdefault("ok", True)
        out.setdefault("error", None)
        return out

def _load(module_path, func_name, fallback):
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name), "REAL"
    except (ImportError, AttributeError):
        return fallback, "MOCK"

def build_default_tools(verbose=True):
    g, g_src = _load("src.graph.tools", GRAPH, mock_query_knowledge_graph)
    v, v_src = _load("src.vector.tools", VECTOR, mock_search_medical_literature)
    if verbose:
        print(f"[tools] {GRAPH}: {g_src} | {VECTOR}: {v_src}")
    return [
        Tool(GRAPH, "Run a Cypher query for multi-hop relationships (genes, drugs, diseases).", g),
        Tool(VECTOR, "Semantic search over medical abstracts and guidelines.", v),
    ]
