"""A worker-routed slash command's exit verdict must reach its caller (#133725).

``HermesCLI.process_command`` returns False to say the caller should exit (``/handoff``
completed — same semantics as /quit). The classic REPL honours it
(``cli_tui_runtime_mixin._tui_run_slash_input``), but the persistent slash worker
discarded it and built its reply from captured stdout only: on the Ink TUI ``/handoff``
completed gateway-side while the pane stayed alive holding the session lease. The
verdict is now harvested like the prompt seed (#107800) and rides back on the reply's
``exit`` field, which slash.exec forwards so the client can terminate the pane.
"""

from __future__ import annotations

import threading

from tui_gateway import server


class _FakeCLI:
    """Records the command; returns a canned ``process_command`` verdict."""

    console = None

    def __init__(self, verdict=None):
        self._verdict = verdict
        self.commands = []

    def process_command(self, cmd):
        self.commands.append(cmd)
        return self._verdict


def test_run_harvests_false_verdict_as_exit():
    from tui_gateway import slash_worker

    cli = _FakeCLI(verdict=False)
    slash_worker._run(cli, "/handoff discord")
    assert cli._slash_exit_verdict is True


def test_run_treats_none_and_true_verdicts_as_keep():
    from tui_gateway import slash_worker

    for verdict in (None, True):
        cli = _FakeCLI(verdict=verdict)
        slash_worker._run(cli, "/status")
        # process_command normalizes ``result is not False``; a None return from a
        # command with no verdict must not read as "exit" (the old fakes rely on it).
        assert cli._slash_exit_verdict is False


def test_run_resets_stale_verdict_on_next_command():
    from tui_gateway import slash_worker

    cli = _FakeCLI(verdict=False)
    slash_worker._run(cli, "/handoff discord")
    cli._verdict = True
    slash_worker._run(cli, "/status")
    assert cli._slash_exit_verdict is False


def test_run_empty_command_leaves_verdict_cleared():
    from tui_gateway import slash_worker

    cli = _FakeCLI(verdict=False)
    slash_worker._run(cli, "")
    assert cli._slash_exit_verdict is False


class _ExitingWorker:
    """String ``run()`` contract plus the exit verdict carried out of band via ``pop_exit()``."""

    def __init__(self, *a, **kw):
        self._last_exit = True

    def run(self, command):
        return "handoff complete"

    def pop_exit(self):
        verdict, self._last_exit = self._last_exit, False
        return verdict

    def pop_seed(self):
        return ""

    def close(self):
        pass


def _live_session():
    return {"session_key": "handoff-key", "running": False, "history": [],
            "history_lock": threading.Lock(), "agent": None, "cwd": ""}


def test_slash_exec_forwards_exit_verdict(monkeypatch):
    sid = "handoff-sid"
    server._sessions[sid] = _live_session()
    monkeypatch.setattr(server, "_SlashWorker", _ExitingWorker)
    try:
        resp = server._methods["slash.exec"](1, {"session_id": sid, "command": "/handoff discord"})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in resp, resp
    assert resp["result"]["exit"] is True
    assert resp["result"]["output"] == "handoff complete"


def test_slash_exec_has_no_exit_field_without_a_verdict(monkeypatch):
    sid = "plain-sid"
    server._sessions[sid] = _live_session()

    class _PlainWorker:
        def __init__(self, *a, **kw):
            pass

        def run(self, command):
            return "worker says hi"

        def close(self):
            pass

    monkeypatch.setattr(server, "_SlashWorker", _PlainWorker)
    try:
        resp = server._methods["slash.exec"](1, {"session_id": sid, "command": "/journey"})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in resp, resp
    assert "exit" not in resp["result"]
    assert resp["result"]["output"] == "worker says hi"
