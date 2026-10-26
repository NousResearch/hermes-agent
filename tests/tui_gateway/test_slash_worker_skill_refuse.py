"""Skill slashes must not ok-reply a loading banner from the slash worker."""

import pytest


def test_slash_worker_refuses_skill_before_process_command(monkeypatch):
    """A skill command is refused before process_command, so the banner is never the reply."""
    from tui_gateway import slash_worker

    monkeypatch.setattr("cli.get_skill_commands", lambda: {"/grilling": {"name": "grilling"}})

    class _CLI:
        def __init__(self):
            self.console = None
            self.called = False

        def process_command(self, cmd):
            self.called = True
            print("\n⚡ Loading skill: grilling")

    cli = _CLI()
    with pytest.raises(Exception, match="skill command refused before process: /grilling"):
        slash_worker._run(cli, "/grilling tighten this")
    assert cli.called is False


def test_slash_worker_still_runs_non_skill_commands(monkeypatch):
    from tui_gateway import slash_worker

    monkeypatch.setattr("cli.get_skill_commands", lambda: {"/grilling": {"name": "grilling"}})

    class _CLI:
        def __init__(self):
            self.console = None

        def process_command(self, cmd):
            print("status ok")

    assert slash_worker._run(_CLI(), "/status") == "status ok"
