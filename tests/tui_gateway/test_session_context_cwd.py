import pytest

from tui_gateway import server


@pytest.fixture(autouse=True)
def _neuter_agent_prewarm_timer(request, monkeypatch):
    """Stub the deferred agent pre-warm timer for every test in this module.

    ``session.create`` and non-eager ``session.resume`` fire a 50 ms
    background ``threading.Timer`` (``_schedule_agent_build``) that calls
    whatever ``server._make_agent`` is patched in AT FIRE TIME. Left live,
    a timer armed by one test outlives it and lands in the NEXT test's
    ``_make_agent`` mock, racily corrupting its captured state (the
    ``'tip' == 'cont_tip'`` flakes in the session_resume tests). Tests that
    exercise the deferred build itself opt back in with
    ``@pytest.mark.real_agent_prewarm``.
    """
    if request.node.get_closest_marker("real_agent_prewarm"):
        yield
        return
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    yield


def test_session_context_uses_session_cwd(monkeypatch, tmp_path):
    """Desktop/TUI sessions must pin the agent cwd per session.

    The gateway process itself is often launched from apps/desktop in dev, so
    falling back to os.getcwd() makes agents answer from the desktop app folder
    even when the sidebar/session cwd is a real project.
    """
    from agent.runtime_cwd import resolve_agent_cwd

    sid = "cwd-sid"
    session_key = "cwd-key"
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)

    server._sessions[sid] = {"session_key": session_key, "cwd": str(project)}
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launcher)

    tokens = server._set_session_context(session_key)
    try:
        assert resolve_agent_cwd() == project
    finally:
        server._clear_session_context(tokens)
        server._sessions.pop(sid, None)


def test_session_context_explicit_cwd_for_ephemeral_task(monkeypatch, tmp_path):
    """Background/preview tasks use ephemeral ids absent from `_sessions`, so the
    parent workspace is passed explicitly; it must pin instead of clearing back
    to the gateway launch dir."""
    from agent.runtime_cwd import resolve_agent_cwd

    project = tmp_path / "project"
    project.mkdir()
    launcher = tmp_path / "apps" / "desktop"
    launcher.mkdir(parents=True)

    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(launcher)

    tokens = server._set_session_context("bg_deadbe", cwd=str(project))
    try:
        assert resolve_agent_cwd() == project
    finally:
        server._clear_session_context(tokens)
