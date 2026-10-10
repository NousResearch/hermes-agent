"""A ``type: exec`` quick command dispatched for a session runs in THAT session's profile env.

Under multiplex ``os.environ`` holds the launch profile's ``.env``; ``command.dispatch`` bound only
the session's home, so a secondary profile's snippet saw the launch profile's values and none of its
own. Proven A→B→A: the launch session keeps its own values after a secondary dispatch.
"""

import threading

import pytest

from agent.secret_scope import set_multiplex_active

_PROBE = "echo A=${A_MARKER:-unset} B=${B_MARKER:-unset} TENV=${TERMINAL_ENV:-unset}"


@pytest.fixture
def mux_homes(tmp_path, monkeypatch):
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    (a / ".env").write_text("A_MARKER=a\nTERMINAL_ENV=docker\n", encoding="utf-8")
    (b / ".env").write_text("B_MARKER=b\n", encoding="utf-8")
    quick = "quick_commands:\n  probe:\n    type: exec\n    command: '" + _PROBE + "'\n"
    (a / "config.yaml").write_text(quick, encoding="utf-8")
    (b / "config.yaml").write_text(quick, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setenv("A_MARKER", "a")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.delenv("B_MARKER", raising=False)
    set_multiplex_active(True)
    try:
        yield a, b
    finally:
        set_multiplex_active(False)


@pytest.mark.platforms("posix")
def test_quick_exec_runs_in_the_dispatching_sessions_profile_env(mux_homes, monkeypatch):
    import tui_gateway.server as server

    a, b = mux_homes

    def dispatch(sid, home):
        monkeypatch.setitem(server._sessions, sid, {
            "session_key": f"key-{sid}", "profile_home": home, "history": [],
            "history_lock": threading.Lock(), "agent": None})
        resp = server._methods["command.dispatch"]("1", {"name": "probe", "arg": "", "session_id": sid})
        return resp["result"]["output"]

    assert dispatch("sid-a1", None) == "A=a B=unset TENV=docker"
    assert dispatch("sid-b", str(b)) == "A=unset B=b TENV=unset"
    assert dispatch("sid-a2", None) == "A=a B=unset TENV=docker"
