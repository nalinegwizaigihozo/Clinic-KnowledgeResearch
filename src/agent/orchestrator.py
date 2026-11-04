import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.agent.prompts import check_request
from src.agent.tools import build_default_tools

log = logging.getLogger("agent")

@dataclass
class Step:
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Dict[str, Any]
    seconds: float = 0.0

@dataclass
class AgentResult:
    answer: str
    steps: List[Step] = field(default_factory=list)
    blocked: bool = False
    stopped_reason: str = "final"

    @property
    def tools_used(self):
        return [s.action for s in self.steps]

class ClinicalResearchAgent:
    def __init__(self, llm, tools=None, max_steps=6):
        self.llm = llm
        self.tools = {t.name: t for t in (tools if tools is not None else build_default_tools())}
        self.max_steps = max_steps

    def run(self, question):
        refusal = check_request(question)
        if refusal:
            log.info("guardrail blocked request")
            return AgentResult(answer=refusal, blocked=True, stopped_reason="guardrail")

        steps, reason = [], "final"
        for i in range(self.max_steps):
            d = self.llm.decide(question, steps, list(self.tools))
            if d.final:
                break
            t0 = time.perf_counter()
            tool = self.tools.get(d.action)
            if tool is None:
                obs = {"ok": False, "results": [], "error": f"unknown tool '{d.action}'"}
            else:
                obs = tool.run(**d.action_input)
            dt = time.perf_counter() - t0
            steps.append(Step(d.thought, d.action, d.action_input, obs, dt))
            log.info("step %d | %s | ok=%s | rows=%d | %.4fs", i + 1, d.action, obs.get("ok"),
                     len(obs.get("results", [])), dt)
        else:
            reason = "max_steps"

        answer = self.llm.synthesize(question, steps, stopped_reason=reason)
        return AgentResult(answer=answer, steps=steps, stopped_reason=reason)

def print_trace(result):
    for i, s in enumerate(result.steps, 1):
        print(f"Step {i}: {s.thought}")
        print(f"   action: {s.action}({s.action_input})")
        print(f"   observation: ok={s.observation.get('ok')} rows={len(s.observation.get('results', []))} "
              f"error={s.observation.get('error')}  [{s.seconds:.4f}s]")
    print(f"stopped_reason={result.stopped_reason} blocked={result.blocked}")
