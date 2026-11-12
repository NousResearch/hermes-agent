"""Argument-name checks in front of built-in tools, driven through the real dispatch."""

import json

from hermes_agent.model_tools import handle_function_call
from hermes_agent.tools.registry import registry

_SKILL = "---\nname: flat-shape-skill\ndescription: Use when probing. Does nothing.\n---\n# Probe\nbody\n"


def test_missing_required_argument_is_refused_before_the_handler_runs(monkeypatch, tmp_path):
    searched = []
    monkeypatch.setattr("hermes_agent.tools.file_tools.search_tool", lambda *a, **kw: searched.append(kw) or "{}")

    refused = json.loads(handle_function_call("search_files", {"path": str(tmp_path)}, task_id="t"))

    assert "pattern" in refused["error"] and "NOT invoked" in refused["error"]
    assert searched == []
    # A files listing takes no pattern: a required key the handler does without stays its call.
    handle_function_call("search_files", {"target": "files", "path": str(tmp_path)}, task_id="t")
    handle_function_call("search_files", {"pattern": "\t", "path": str(tmp_path)}, task_id="t")  # a tab is a pattern
    assert [(kw["target"], kw["pattern"]) for kw in searched] == [("files", ""), ("content", "\t")]
    # Every other built-in is refused by the generic check, before its handler.
    monkeypatch.setattr("hermes_agent.tools.file_tools.read_file_tool", lambda *a, **kw: searched.append("read") or "{}")
    refused = json.loads(handle_function_call("read_file", {"offset": 3}, task_id="t"))
    assert "path" in refused["error"] and "NOT invoked" in refused["error"] and "read" not in searched
    # The flat single-op shape old transcripts replay omits the required ``operations``.
    created = json.loads(handle_function_call(
        "skill_manage", {"action": "create", "name": "flat-shape-skill", "content": _SKILL}, task_id="t"))
    assert created["success"] is True


def test_unknown_argument_reaches_the_handler_and_is_flagged(monkeypatch):
    received = {}
    monkeypatch.setattr("hermes_agent.tools.process_registry._handoff_process",
                        lambda session_id, args, task_id: received.update(args) or {"status": "ok"})
    call = {"action": "handoff", "session_id": "proc_1", "sesion_id": "proc_2"}

    result = json.loads(handle_function_call("process_manage", dict(call), task_id="t"))

    assert received == call
    assert result["status"] == "ok" and "sesion_id" in result["_warning"]

    # A third-party schema's property list is no promise about what its handler accepts. The
    # handler is a ``tools`` one, as an MCP tool's is, so only the ``mcp-*`` toolset exempts it.
    registry.register(
        name="mcp_probe_handoff", toolset="mcp-probe", handler=registry.get_entry("process_manage").handler,
        schema={"name": "mcp_probe_handoff", "description": "probe",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}})
    try:
        assert json.loads(handle_function_call("mcp_probe_handoff", dict(call), task_id="t")) == {"status": "ok"}
    finally:
        registry.deregister("mcp_probe_handoff")
