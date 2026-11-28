"""MCP Tasks through the real client: a task-handle answer resolves to its final result.

Real stdio peer (``fixtures/mcp_tasks_server.py``), real ``register_mcp_servers`` and registry
dispatch; the wire log proves what Hermes sent.
"""

import json
import sys
from pathlib import Path

from tools.mcp_tool_discovery import register_mcp_servers
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
from tools.registry import registry

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_tasks_server.py"


def _register(tmp_path, **extra):
    log = tmp_path / "wire.jsonl"
    register_mcp_servers({"tasks_peer": {"command": sys.executable, "args": [str(FIXTURE), str(log)],
                                         "protocol": "stateless", "keepalive_interval": 0, **extra}})
    return lambda: [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()]


def test_task_handle_is_polled_to_its_result_and_failures_surface_as_errors(tmp_path):
    messages = _register(tmp_path)
    try:
        done = registry.dispatch("mcp__tasks_peer__slow_job", {})
        assert "ORCHID-5521" in done and "error" not in json.loads(done)
        failed = json.loads(registry.dispatch("mcp__tasks_peer__broken_job", {}))
        assert "upstream build server rejected the job" in failed["error"]
        sent = messages()
        call = next(m for m in sent if m.get("method") == "tools/call")
        caps = call["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"]
        assert "io.modelcontextprotocol/tasks" in caps["extensions"]
        gets = [m for m in sent if m.get("method") == "tasks/get"]
        assert len(gets) == 6 and all(len(m["params"]["taskId"]) > 128 for m in gets)

        asked = json.loads(registry.dispatch("mcp__tasks_peer__ask_job", {}))
        assert "asked for input mid-task" in asked["error"]
        lost = json.loads(registry.dispatch("mcp__tasks_peer__lost_job", {}))
        assert "do not start it again" in lost["error"]
        sent = messages()
        cancels = [m["params"]["taskId"] for m in sent if m.get("method") == "tasks/cancel"]
        assert len(cancels) == 1 and cancels[0].startswith("ask-")
        calls = [m["params"]["name"] for m in sent if m.get("method") == "tools/call"]
        assert calls == ["slow_job", "broken_job", "ask_job", "lost_job"]  # a started task is never replayed
    finally:
        shutdown_mcp_servers()


def test_tasks_false_keeps_the_extension_off_the_wire(tmp_path):
    messages = _register(tmp_path, tasks=False)
    try:
        result = registry.dispatch("mcp__tasks_peer__slow_job", {})
        assert "sync fallback" in result
        call = next(m for m in messages() if m.get("method") == "tools/call")
        caps = call["params"]["_meta"].get("io.modelcontextprotocol/clientCapabilities") or {}
        assert "io.modelcontextprotocol/tasks" not in (caps.get("extensions") or {})
    finally:
        shutdown_mcp_servers()
