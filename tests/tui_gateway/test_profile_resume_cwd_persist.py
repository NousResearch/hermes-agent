"""Every ``session.resume`` path repairs a named profile's saved workspace (#87584).

``test_profile_resume_cwd.py`` proves the LIVE session lands in the profile's own home, then calls
``_hydrate_session_cwd`` by hand to show the row can be repaired. Here each real path runs end to end — the eager
build (``eager_build``), the default cold resume, and the deferred one (``defer_history``) — and the STORED row is
read back: a sidebar listing or the next resume reads the row, not the live record, so a path that leaves the old
folder there brings it back.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from tests.tui_gateway.test_tui_gateway_server import _write_profile_cfg
from tui_gateway import server

KEY = "20261012_120000_b07c4a8"


class _Agent:
    model = "test-model"
    provider = "test"

    def __init__(self, **_kwargs):
        pass

    def close(self):
        pass


@pytest.fixture
def resume_path(monkeypatch):
    """``session.resume`` for named profile ``austin`` (placeholder ``terminal.cwd``) whose row holds the launch
    profile's root; returns ``(run(flags) -> stored row cwd, profile home)``."""
    root = server.get_hermes_home()
    home = _write_profile_cfg(root / "profiles" / "austin", ".")
    monkeypatch.setattr(server, "_profile_home", lambda name: home if name else None)
    monkeypatch.setenv("TERMINAL_CWD", str(root))
    for name, value in {
        "_resolve_model": lambda: "test-model",
        "_enable_gateway_prompts": lambda: None,
        "_find_live_session_by_key": lambda _key, _home=None, _incarnation=None: None,
        "_schedule_session_cap_enforcement": lambda *a, **k: None,
        "_maybe_schedule_auto_continue": lambda *a, **k: None,
        "_child_run_active": lambda _key, _home=None: False,
        "_make_agent": lambda *a, **k: _Agent(),
        "_wire_session_agent": lambda *a, **k: False,
        "_start_session_services": lambda *a, **k: None,
        "_schedule_mcp_late_refresh": lambda *a, **k: None,
        "_session_info": lambda *a, **k: {},
        "_emit": lambda *a, **k: None,
    }.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setattr(server.git_probe, "branch", lambda _cwd: None)
    monkeypatch.setattr(server.git_probe, "common_repo_root", lambda _cwd: None)
    known = set(server._sessions)

    def run(flags: dict) -> str:
        db = SessionDB(db_path=home / "state.db")
        try:
            db.create_session(KEY, source="desktop", model="test-model", cwd=str(root))
            resp = server.handle_request({"id": "1", "method": "session.resume",
                                          "params": {"session_id": KEY, "profile": "austin", **flags}})
            assert "error" not in resp, resp
            live = server._sessions[resp["result"]["session_id"]]
            assert live["cwd"] == str(home)
            if (build := live.get("agent_ready")) is not None:
                build.wait(5)
        finally:
            db.close()
        reread = SessionDB(db_path=home / "state.db")
        try:
            return reread.get_session(KEY)["cwd"]
        finally:
            reread.close()

    yield run, home
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


@pytest.mark.parametrize("flags", [
    pytest.param({"eager_build": True}, id="eager"),
    pytest.param({}, id="cold"),
    pytest.param({"defer_history": True}, id="deferred"),
])
def test_every_resume_path_repairs_the_saved_workspace(resume_path, flags):
    run, home = resume_path
    assert run(flags) == str(home)
