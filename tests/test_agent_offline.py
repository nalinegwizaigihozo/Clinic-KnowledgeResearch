from src.agent.llm import OfflineReActLLM
from src.agent.orchestrator import ClinicalResearchAgent
from src.agent.tools import Tool, build_default_tools, GRAPH, VECTOR

def make(**kw):
    return ClinicalResearchAgent(OfflineReActLLM(), **kw)

def test_multi_step_graph_then_literature():
    r = make().run("Which drugs target genes associated with type 2 diabetes, and what does the literature say about them?")
    assert r.tools_used == [GRAPH, VECTOR]
    assert "PPARG" in r.answer and "[1]" in r.answer
    assert "disclaimer" not in r.answer.lower() or "not a diagnosis" in r.answer.lower()

def test_literature_only():
    r = make().run("What do clinical trials say about metformin?")
    assert r.tools_used == [VECTOR]
    assert "metformin" in r.answer.lower()

def test_graph_only():
    r = make().run("Which genes are associated with Alzheimer's disease?")
    assert r.tools_used == [GRAPH]
    assert "APOE" in r.answer

def test_drug_to_gene():
    r = make().run("What genes does metformin target?")
    assert r.tools_used == [GRAPH] and "PRKAA1" in r.answer

def test_guardrail_blocks_personal_diagnosis():
    r = make().run("Do I have diabetes? I am thirsty a lot, what should I take?")
    assert r.blocked and r.steps == []
    assert "not a clinician" in r.answer

def test_empty_graph_falls_back_to_literature():
    r = make().run("Which drugs target genes associated with fakeitis?")
    assert r.tools_used == [GRAPH, VECTOR]
    assert "returned no results" in r.answer

def test_tool_failure_is_survivable():
    tools = build_default_tools(verbose=False)
    tools[0] = Tool(GRAPH, "broken", lambda cypher: 1 / 0)
    r = make(tools=tools).run("Which genes are associated with Alzheimer's disease?")
    assert r.steps[0].observation["ok"] is False
    assert "failed" in r.answer and "ZeroDivisionError" in r.answer

def test_max_steps_limit():
    r = make(max_steps=1).run("Which drugs target genes associated with type 2 diabetes, and what does the literature say about them?")
    assert r.stopped_reason == "max_steps" and len(r.steps) == 1
    assert "step limit" in r.answer
