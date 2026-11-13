"""``/init`` on the TUI/desktop surface must target the SESSION's active directory.

The desktop app launches the backend from the home directory, so the old bare-``os.getcwd()``
resolution scanned and merge-updated the HOME's ``AGENTS.md`` while the session's own
workspace sat elsewhere. These drive the real ``command.dispatch`` handler (the surface the
desktop uses) rather than the builder alone, so a regression at the call site — passing a
stale terminal record straight through as ``cwd`` — cannot slip past the builder's own guard.
"""

from __future__ import annotations

from tools.terminal_tool import clear_session_cwd, record_session_cwd
from tui_gateway import server


def _register(sid: str, session_cwd: str) -> None:
    """A minimal session record: the handler needs ``session_key`` and its attached ``cwd``."""
    server._sessions[sid] = {"agent": None, "cwd": session_cwd, "history": [], "session_key": sid}


def _dispatch_init(sid: str) -> str:
    envelope = server._methods["command.dispatch"](1, {"name": "init", "arg": "", "session_id": sid})
    return envelope["result"]["message"]


def test_init_targets_the_sessions_recorded_cwd(tmp_path, monkeypatch):
    workspace = tmp_path / "proj"
    workspace.mkdir()
    launch_dir = tmp_path / "home"  # the desktop backend's launch dir
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)

    sid = "init-live-record"
    _register(sid, str(launch_dir))
    record_session_cwd(sid, str(workspace))
    try:
        prompt = _dispatch_init(sid)
        assert f"for the project at: {workspace}" in prompt
        assert str(launch_dir) not in prompt
    finally:
        clear_session_cwd(sid)
        server._sessions.pop(sid, None)


def test_init_ignores_a_stale_record_in_favour_of_the_session_workspace(tmp_path, monkeypatch):
    """A record for a deleted directory (removed worktree) falls through to the session cwd."""
    workspace = tmp_path / "proj"
    workspace.mkdir()
    launch_dir = tmp_path / "home"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)

    gone = tmp_path / "removed-worktree"
    gone.mkdir()
    gone.rmdir()

    sid = "init-stale-record"
    _register(sid, str(workspace))
    record_session_cwd(sid, str(gone))
    try:
        prompt = _dispatch_init(sid)
        assert f"for the project at: {workspace}" in prompt
        assert str(gone) not in prompt
    finally:
        clear_session_cwd(sid)
        server._sessions.pop(sid, None)
