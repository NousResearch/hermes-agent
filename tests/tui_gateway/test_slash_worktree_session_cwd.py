"""Desktop /worktree must use the session project cwd, not the gateway launch dir.

#102268: slash.exec runs /worktree in a worker spawned with os.getcwd() (the
gateway process). Desktop sessions carry the project on session['cwd'], so
the command either refused ('not inside a git repository') or created the
tree in an unrelated repo, and the live session never moved.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import tui_gateway.server as server


def test_slash_worker_cwd_prefers_session_directory(tmp_path):
    session_cwd = tmp_path / "project"
    session_cwd.mkdir()
    assert server._slash_worker_cwd(str(session_cwd)) == str(session_cwd.resolve())


def test_slash_worker_cwd_falls_back_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "gone"
    assert server._slash_worker_cwd(str(missing)) == str(tmp_path)


def test_slash_worker_popen_uses_session_cwd(tmp_path):
    session_cwd = tmp_path / "project"
    session_cwd.mkdir()
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value=Path("/tmp/hermes_test")),
        ),
    }):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = MagicMock()
            mock_popen.return_value.stderr = MagicMock()
            worker = server._SlashWorker(
                session_key="test_key",
                model="test-model",
                cwd=str(session_cwd),
            )
            assert mock_popen.call_args[1]["cwd"] == str(session_cwd.resolve())
            assert worker.cwd == str(session_cwd.resolve())


def test_parse_worktree_ready_path_from_worker_output():
    output = (
        "  ✓ Worktree created: /tmp/proj/.worktrees/repro-wt\n"
        "  Branch: hermes/repro-wt\n"
        "  Worktree ready: /tmp/proj/.worktrees/repro-wt\n"
        "  Terminal and file tools now operate in the worktree.\n"
    )
    assert server._parse_worktree_ready_path(output) == "/tmp/proj/.worktrees/repro-wt"
    assert server._parse_worktree_ready_path("nothing here") is None


def test_mirror_worktree_new_retargets_session_cwd(tmp_path, monkeypatch):
    tree = tmp_path / "proj" / ".worktrees" / "repro-wt"
    tree.mkdir(parents=True)
    session = {
        "cwd": str(tmp_path / "proj"),
        "session_key": "sess-1",
        "agent": None,
    }
    emitted = []

    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_git_branch_for_cwd", lambda cwd: "hermes/repro-wt")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda cwd: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(
        server, "_persist_session_cwd_and_schedule_git_meta", lambda *a, **k: None
    )
    monkeypatch.setattr(server, "cleanup_vm", lambda *a, **k: None, raising=False)

    with patch("tools.terminal_tool.cleanup_vm", lambda *a, **k: None):
        warning = server._mirror_slash_side_effects(
            "sid-1",
            session,
            "/worktree new repro-wt",
            output=f"  Worktree ready: {tree}\n",
        )

    assert warning == ""
    assert Path(session["cwd"]) == tree.resolve()
    assert session["explicit_cwd"] is True
    assert emitted and emitted[0][0] == "session.info"


def test_mirror_worktree_list_does_not_move_cwd(tmp_path):
    session = {"cwd": str(tmp_path), "session_key": "sess-1", "agent": None}
    warning = server._mirror_slash_side_effects(
        "sid-1",
        session,
        "/worktree list",
        output="  /tmp/proj  abc123 [main]\n",
    )
    assert warning == ""
    assert session["cwd"] == str(tmp_path)
