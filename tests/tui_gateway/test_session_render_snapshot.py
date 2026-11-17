"""``session.render_snapshot``: what the agent saw, read from the stored prompt hash and tools pin.

The inspector must never rebuild the system prompt (that would re-render it and could move the
cached prefix), and must diff a compression lineage segment by segment.
"""

import threading
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from hermes_state import SessionDB


def _tool(name: str, description: str = "") -> dict:
    return {"type": "function", "function": {"name": name, "description": description or f"{name} tool"}}


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


def _live(monkeypatch, key: str, agent=None) -> str:
    sid = "rt-inspect"
    monkeypatch.setitem(srv._sessions, sid, {"session_key": key, "agent": agent, "history": [],
                                            "history_lock": threading.Lock(), "running": False})
    return sid


def test_render_snapshot_reads_the_stored_prompt_and_diffs_the_lineage(db, monkeypatch):
    db.create_session("seg-1", source="desktop", model="model-a",
                      system_prompt="You are Hermes. token sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789")
    db.update_session_tool_names("seg-1", {"version": "v", "tools": [_tool("terminal"), _tool("read_file")]})
    db.end_session("seg-1", "compression")
    db.create_session("seg-2", source="desktop", model="model-b", parent_session_id="seg-1",
                      system_prompt="You are Hermes, compacted.")
    db.update_session_tool_names("seg-2", {"version": "v", "tools": [_tool("terminal"), _tool("mcp__gh__issues")]})

    def _no_rebuild(*_a, **_k):
        raise AssertionError("the inspector rebuilt the system prompt")

    agent = SimpleNamespace(tools=[_tool("terminal", "live terminal"), _tool("mcp__gh__issues")],
                            model="model-b", provider="nous", _build_system_prompt=_no_rebuild)
    monkeypatch.setattr("agent.system_prompt.build_system_prompt", _no_rebuild)
    monkeypatch.setattr("agent.system_prompt.build_system_prompt_parts", _no_rebuild)
    sid = _live(monkeypatch, "seg-1", agent)

    result = srv._methods["session.render_snapshot"](1, {"session_id": sid})["result"]

    assert result["session_id"] == "seg-2"
    assert result["system_prompt"] == "You are Hermes, compacted."
    assert (result["model"], result["provider"], result["tools_source"]) == ("model-b", "nous", "live")
    assert result["tools"][0]["description"] == "live terminal"
    assert result["mcp_servers"] == ["gh"]
    assert [s["session_id"] for s in result["segments"]] == ["seg-1", "seg-2"]
    assert result["segments"][0]["changes"] == []
    assert {(c["kind"], c.get("name")) for c in result["segments"][1]["changes"]} == {
        ("model_changed", None), ("prompt_changed", None),
        ("tool_added", "mcp__gh__issues"), ("tool_removed", "read_file")}


def test_render_snapshot_redacts_secrets_in_the_stored_prompt_without_an_agent(db, monkeypatch):
    secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
    db.create_session("solo", source="desktop", model="model-a", system_prompt=f"Use key {secret} carefully.")
    db.update_session_tool_names("solo", {"version": "v", "tools": [_tool("web_search", f"auth {secret}")]})
    sid = _live(monkeypatch, "solo")

    result = srv._methods["session.render_snapshot"](1, {"session_id": sid})["result"]

    assert result["tools_source"] == "stored"
    assert [t["name"] for t in result["tools"]] == ["web_search"]
    assert secret not in result["system_prompt"] and "carefully" in result["system_prompt"]
    assert secret not in result["tools"][0]["description"]
    assert db.get_session("solo")["system_prompt"] == f"Use key {secret} carefully."  # stored row untouched
