"""session.workspace.move must refuse a session that goes busy mid-call.

The handler checks ``running`` once, then runs two git subprocesses before it
mutates anything, and it runs on the RPC pool. A prompt submitted inside that
window flips ``running`` after the check — and ``_set_session_cwd`` does not
re-check — so the move re-anchors a live turn's terminal/file tools onto a
different directory. The handler's own docstring says it refuses that case.
"""

import pytest


@pytest.fixture()
def gw(tmp_path, monkeypatch):
    from tui_gateway import server, methods_session

    old = tmp_path / "old"
    old.mkdir()
    new = tmp_path / "new"
    new.mkdir()

    sess = {"session_key": "sk1", "running": False, "cwd": str(old), "agent": None}
    monkeypatch.setitem(server._sessions, "sid1", sess)
    monkeypatch.setattr(
        server, "_git_common_repo_root_for_cwd", lambda _c: ""
    )
    return server, methods_session, sess, old, new


def _call(server, new):
    return server._methods["session.workspace.move"](
        1, {"session_key": "sk1", "cwd": str(new)}
    )


def test_refuses_when_a_turn_starts_during_the_git_probe(gw, monkeypatch):
    server, methods_session, sess, old, new = gw

    def _probe_then_turn_starts(_cwd):
        # A prompt lands while the git subprocess runs.
        sess["running"] = True
        return "main"

    monkeypatch.setattr(
        server, "_git_branch_for_cwd", _probe_then_turn_starts
    )

    resp = _call(server, new)

    assert resp["error"]["message"] == "session busy"
    assert sess["cwd"] == str(old), "workspace moved out from under a live turn"


def test_still_refuses_when_already_busy_before_the_call(gw, monkeypatch):
    server, methods_session, sess, old, new = gw
    sess["running"] = True
    monkeypatch.setattr(
        server, "_git_branch_for_cwd", lambda _c: "main"
    )

    resp = _call(server, new)

    assert resp["error"]["message"] == "session busy"
    assert sess["cwd"] == str(old)


def test_missing_target_directory_is_rejected(gw, monkeypatch):
    server, methods_session, sess, old, new = gw
    monkeypatch.setattr(
        server, "_git_branch_for_cwd", lambda _c: "main"
    )

    resp = server._methods["session.workspace.move"](
        1, {"session_key": "sk1", "cwd": str(new / "does-not-exist")}
    )

    assert "does not exist" in resp["error"]["message"]
    assert sess["cwd"] == str(old)
