"""session.create / terminal.resize must tolerate malformed ``cols``.

``session.resume`` already wraps ``int(cols)`` in try/except. ``session.create``
is an inline (non-_LONG_HANDLERS) RPC, so a bare ``int(...)`` raising
ValueError/TypeError can tear down the TUI gateway stdin/WS reader loop.
``terminal.resize`` has the same bare ``int`` sibling.
"""

from __future__ import annotations

import threading

from tui_gateway import server


def _stub_create_side_effects(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *a, **k: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_resolve_session_source", lambda s=None: "tui")
    monkeypatch.setattr(server, "_new_session_key", lambda: "key-test")
    monkeypatch.setattr(server, "_coerce_seed_history", lambda _m: [])
    monkeypatch.setattr(server, "_git_branch_for_cwd", lambda _cwd: "")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda _cwd: {})
    monkeypatch.setattr(server, "_profile_home", lambda _p: None)
    monkeypatch.setattr(server, "_response_profile_name", lambda _p: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")


def test_session_create_tolerates_non_int_cols(monkeypatch, tmp_path):
    _stub_create_side_effects(monkeypatch, tmp_path)

    resp = server.dispatch(
        {
            "id": "1",
            "method": "session.create",
            "params": {"cols": "wide"},
        }
    )
    assert "error" not in resp, resp
    sid = resp["result"]["session_id"]
    assert server._sessions[sid]["cols"] == 80
    server._sessions.pop(sid, None)


def test_session_create_tolerates_list_cols(monkeypatch, tmp_path):
    _stub_create_side_effects(monkeypatch, tmp_path)

    resp = server.dispatch(
        {
            "id": "2",
            "method": "session.create",
            "params": {"cols": [120]},
        }
    )
    assert "error" not in resp, resp
    sid = resp["result"]["session_id"]
    assert server._sessions[sid]["cols"] == 80
    server._sessions.pop(sid, None)


def test_terminal_resize_tolerates_non_int_cols(monkeypatch):
    ready = threading.Event()
    ready.set()
    server._sessions["sid-r"] = {
        "cols": 80,
        "agent_ready": ready,
        "session_key": "k",
    }
    try:
        resp = server.handle_request(
            {
                "id": "3",
                "method": "terminal.resize",
                "params": {"session_id": "sid-r", "cols": "nope"},
            }
        )
        assert "error" not in resp, resp
        assert resp["result"]["cols"] == 80
        assert server._sessions["sid-r"]["cols"] == 80
    finally:
        server._sessions.pop("sid-r", None)
