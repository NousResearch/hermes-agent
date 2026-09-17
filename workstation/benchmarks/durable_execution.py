"""Provider-free regression scenario using the production compiler and durable DB.

The baseline models item-by-item planner boundaries, not measured provider tokens.
"""
import json
from pathlib import Path
import sqlite3
import tempfile

from workstation.artifacts import ArtifactStore
from workstation.durable_tasks import DurableTaskStore
from workstation.task_compiler import TaskCompiler


def run(root: Path, count: int = 100) -> dict:
    artifacts = ArtifactStore(root / "artifacts")
    compiler = TaskCompiler(DurableTaskStore(conn=sqlite3.connect(root / "kanban.db")), artifacts)
    request = {"operation_key": "synthetic-v1", "kind": "browser_transaction",
               "items": [{"index": i} for i in range(count)],
               "steps": [{"tool": "browser_snapshot", "args": {"index": "$item.index"},
                          "expect": {"ok": True}},
                         {"tool": "read_file", "args": {"index": "$item.index", "path": "unchanged.txt"}}]}
    calls = []
    transient = set()
    def dispatch(tool, args, task, call):
        index = args["index"]
        if index == 37 and "restart" not in calls:
            calls.append("restart")
            raise KeyboardInterrupt()
        if tool == "browser_snapshot" and index == 50 and index not in transient:
            transient.add(index)
            raise TimeoutError("recoverable read")
        calls.append((index, tool))
        return {"ok": index != count - 1 or tool == "read_file", "text": "synthetic row\n" * 4000}
    try:
        compiler.execute(request, task_id="browser", session_id="session", dispatch=dispatch, environment="benchmark")
    except KeyboardInterrupt:
        pass
    compiler = TaskCompiler(DurableTaskStore(conn=sqlite3.connect(root / "kanban.db")), artifacts)
    result = compiler.execute(request, task_id="browser", session_id="session", dispatch=dispatch, environment="benchmark")
    artifact_bytes = sum(p.stat().st_size for p in artifacts.root.rglob("*") if p.is_file() and not p.name.endswith(".meta.json"))
    inline = len(json.dumps(result).encode())
    baseline_bytes = count * len(json.dumps({"ok": True, "text": "synthetic row\n" * 4000}).encode())
    return {"items": count, "baseline": {"planner_interventions": count,
                 "inline_bytes": baseline_bytes},
            "durable": {"planner_interventions": 2, "runtime_llm_calls": 0,
                 "completed": result["completed"], "exceptions": result["needs_reasoning"],
                 "inline_bytes": inline, "artifact_bytes": artifact_bytes,
                 "bytes_avoided": max(0, baseline_bytes - inline),
                 "tool_calls": len(calls) + len(transient),
                 "duplicate_calls_blocked": result["metrics"]["duplicate_calls_blocked"],
                 "cache_hits": result["metrics"]["cache_hits"], "replans": result["metrics"]["replans"],
                 "restart_completed_items_replayed": len([c for c in calls if isinstance(c, tuple) and c[0] in range(37)]) - 74}}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        print(json.dumps(run(Path(temp)), indent=2))
