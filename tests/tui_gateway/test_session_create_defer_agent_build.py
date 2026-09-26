"""``session.create {defer_agent_build: true}`` mints a draft without pre-warming its agent.

A client that opens a new-chat draft the user may discard must not start an agent (and the
MCP fleet its build spawns) for it; the agent builds on first use instead.
"""

import pytest


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server as srv

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(srv, "_sessions", {})
    monkeypatch.setattr(srv, "_load_cfg", lambda: {})
    monkeypatch.setattr(srv, "_profile_home", lambda *a: None)
    monkeypatch.setattr(srv, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(srv, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(srv, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(srv, "_project_info_for_cwd", lambda *a: None)
    return srv


@pytest.mark.parametrize(("params", "prewarmed"), [({}, True), ({"defer_agent_build": True}, False)])
def test_create_prewarms_unless_deferred(server, monkeypatch, params, prewarmed):
    scheduled = []
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid, *a: scheduled.append(sid))
    resp = server.handle_request({"id": "c", "method": "session.create", "params": {"cols": 80, **params}})
    sid = resp["result"]["session_id"]
    assert resp["result"]["info"]["lazy"] is True
    assert scheduled == ([sid] if prewarmed else [])
    assert server._sessions[sid].get("agent") is None


def test_deferred_draft_builds_its_agent_on_the_first_prompt(server, monkeypatch):
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a: pytest.fail("deferred draft pre-warmed"))
    resp = server.handle_request({"id": "c", "method": "session.create",
                                  "params": {"cols": 80, "defer_agent_build": True}})
    sid = resp["result"]["session_id"]
    built = []
    monkeypatch.setattr(server, "_start_agent_build", lambda s, session: built.append(s))
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *a, **k: None)
    monkeypatch.setattr(server, "_persist_session_row_for_submit", lambda *a, **k: None, raising=False)
    server.handle_request({"id": "p", "method": "prompt.submit", "params": {"session_id": sid, "text": "hi"}})
    assert built == [sid]
