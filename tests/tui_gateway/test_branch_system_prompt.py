"""A TUI/Desktop branch child sends the bytes its parent already sends.

Without a stored prompt the child's first turn rebuilds (a fresh workspace probe), so the warm
cache the copied transcript buys is lost at byte 0 whenever the repo moved since session start.
"""

from __future__ import annotations

from pathlib import Path


def test_persist_branch_copies_the_parent_system_prompt(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    with SessionDB(home / "state.db") as db:
        db.create_session("parent", source="desktop", model="test-model")
        db.update_system_prompt("parent", "PARENT PROMPT\n\nWorkspace snapshot: session start")
        server._persist_branch(db, "child", "parent", "Branch", [{"role": "user", "content": "hello"}],
                               source="desktop", cwd=str(tmp_path), profile_name="default",
                               model="test-model")
        # A parent with no stored prompt yields a child with none — never a phantom empty string.
        db.create_session("bare", source="desktop", model="test-model")
        server._persist_branch(db, "bare-child", "bare", "Branch 2", [{"role": "user", "content": "hi"}],
                               source="desktop", cwd=str(tmp_path), profile_name="default",
                               model="test-model")

    with SessionDB(home / "state.db") as db:
        assert db.get_session("child")["system_prompt"] == "PARENT PROMPT\n\nWorkspace snapshot: session start"
        assert db.get_session("bare-child")["system_prompt"] is None


def test_persist_branch_carries_the_parent_tools_pin(tmp_path, monkeypatch):
    """tools[] heads the request ahead of the system prompt, so the child needs the parent's pin too:
    unpinned, the branch's fresh agent pins its own derivation (order and bytes) on its first turn."""
    import json
    from types import SimpleNamespace

    from agent.conversation_loop import _restore_pinned_tools
    from hermes_state import SessionDB
    from tools.mcp_tool_agent import tool_pin_version
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    def tool(name, description):
        return {"type": "function",
                "function": {"name": name, "description": description, "parameters": {"type": "object"}}}

    parent_sent = [tool("alpha", "pinned bytes"), tool("beta", "b"), tool("gamma", "c")]
    with SessionDB(home / "state.db") as db:
        db.create_session("parent", source="desktop", model="test-model")
        db.update_session_tool_names("parent", {"version": tool_pin_version(), "tools": parent_sent})
        server._persist_branch(db, "child", "parent", "Branch", [{"role": "user", "content": "hello"}],
                               source="desktop", cwd=str(tmp_path), profile_name="default",
                               model="test-model")
        db.create_session("bare", source="desktop", model="test-model")
        server._persist_branch(db, "bare-child", "bare", "Branch 2", [{"role": "user", "content": "hi"}],
                               source="desktop", cwd=str(tmp_path), profile_name="default",
                               model="test-model")
        child_row = db.get_session("child")
        assert db.get_session("bare-child")["tool_names"] is None

    # The branch's agent is built fresh: this surface derives the same tools in another order/bytes.
    fresh = SimpleNamespace(tools=[tool("beta", "b"), tool("gamma", "c"), tool("alpha", "surface bytes")],
                            session_id=None, _session_db=None, enabled_toolsets=None,
                            disabled_toolsets=None, _persist_disabled=True)
    _restore_pinned_tools(fresh, child_row)
    assert json.dumps(fresh.tools, sort_keys=True) == json.dumps(parent_sent, sort_keys=True)
