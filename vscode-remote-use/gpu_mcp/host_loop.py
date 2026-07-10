"""host_loop.py — the +æ^glocal Python host loop primitive (bounded, deterministic, offline).

SELF-CONTAINED COPY bundled inside the Hermes Agent VS Code extension.
Canonical source of truth: environments/host_loop.py (hermes-fork repo).
This copy ships with the plugin so the GPU-MCP server runs standalone (no repo needed).

Contract (per the sovereign local-loop pattern):
  planner  : a local brain that reasons + emits tool calls (default: ollama)
  executor : dispatches tool calls to the host (CUDA bridge, terminal, files)
  auditor  : verifies each step / the final result
  manifest : a run record (JSON) proving what happened — the loop's audit trail

Fully air-gappable: brain is a local model socket, hands are local processes.
"""
from __future__ import annotations
import json, re, subprocess, time, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class LoopStep:
    turn: int
    role: str
    action: str
    detail: str
    ok: Optional[bool] = None


class LocalHostLoop:
    """Bounded local agent loop: planner -> executor -> auditor -> manifest."""

    SYSTEM = ("You are a sovereign local agent. Emit ONE tool call per turn as "
              "`TOOL: <name> [arg]`, or `FINAL: <answer>` when done. Be terse.")

    def __init__(
        self,
        planner: Optional[Callable[[str], str]] = None,
        tools: Optional[Dict[str, Callable[[str], Any]]] = None,
        max_steps: int = 6,
        manifest_path: Optional[str] = None,
        name: str = "host-loop",
    ):
        self.planner = planner or self._default_planner
        self.tools = tools or {}
        self.max_steps = max_steps
        self.name = name
        tmp = Path(__import__("tempfile").gettempdir())
        self.manifest_path = Path(manifest_path or tmp / f"{name}.manifest.json")
        self.history: list[dict] = [{"role": "system", "content": self.SYSTEM}]
        self.steps: list[LoopStep] = []
        self.run_id = uuid.uuid4().hex[:12]

    def _default_planner(self, prompt: str) -> str:
        try:
            out = subprocess.run(
                ["curl", "-s", "-m", "180", "-d",
                 json.dumps({"model": "qwen2.5-coder:3b", "prompt": prompt, "stream": False}),
                 "http://localhost:11434/api/generate"],
                capture_output=True, text=True, timeout=200,
            )
            return json.loads(out.stdout).get("response", "").strip()
        except Exception as e:
            return f"FINAL: brain error ({e})"

    def execute(self, tool: str, arg: Optional[str]) -> Any:
        if tool not in self.tools:
            return {"error": f"unknown tool '{tool}'", "available": sorted(self.tools)}
        return self.tools[tool](arg)

    def audit(self, result: Any) -> bool:
        if isinstance(result, dict):
            if result.get("error"):
                return False
            if "ok" in result:
                return bool(result["ok"])
        return True

    def run(self, task: str, auditor: Optional[Callable[[Any], bool]] = None) -> Dict[str, Any]:
        audit = auditor or self.audit
        self.history.append({"role": "user", "content": task})
        self.steps.append(LoopStep(0, "planner", "task", task))
        answer = None
        for step in range(self.max_steps):
            out = self.planner(self._prompt())
            self.history.append({"role": "assistant", "content": out})
            self.steps.append(LoopStep(step, "planner", "think", out[:200]))
            m = re.search(r"TOOL:\s*(\w+)(?:\s+(\S+))?", out)
            if m and m.group(1) in self.tools:
                res = self.execute(m.group(1), m.group(2))
                ok = audit(res)
                self.steps.append(LoopStep(step, "executor", m.group(1), str(res)[:200], ok))
                self.history.append({"role": "user",
                                     "content": f"TOOL_RESULT: {json.dumps(res, default=str)[:400]}"})
                continue
            fm = re.search(r"FINAL:\s*([\s\S]+)", out)
            if fm:
                answer = fm.group(1).strip()
                self.steps.append(LoopStep(step, "planner", "final", answer[:200], True))
                break
        if answer is None:
            answer = "[incomplete: step budget]"
            self.steps.append(LoopStep(self.max_steps, "auditor", "incomplete", answer))
        self._write_manifest(task, answer)
        return {"answer": answer, "steps": len(self.steps), "run_id": self.run_id}

    def _prompt(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.history)

    def _write_manifest(self, task: str, answer: str) -> None:
        manifest = {
            "loop": self.name,
            "run_id": self.run_id,
            "task": task,
            "answer": answer,
            "steps": [s.__dict__ for s in self.steps],
            "ts": time.time(),
        }
        try:
            self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
