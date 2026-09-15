"""TUI startup prompt: a card's dead skill pin must not abort the run it owns.

``_startup_system_prompt`` mirrors cli.py's preload contract for HERMES_TUI_SKILLS
(the TUI is also the Desktop/dashboard backend). Its own docstring promised that
a bad pin "must not auto-block the Kanban task", but an all-missing set still
raised — the same crash the worker path had.
"""

from __future__ import annotations

import pytest


def _patch_preload(monkeypatch, loaded, missing):
    import agent.skill_commands as sc

    monkeypatch.setattr(
        sc, "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("", list(loaded), list(missing)),
    )
    from tui_gateway import server

    monkeypatch.setattr(server, "_parse_tui_skills_env", lambda: list(loaded) + list(missing))
    return server


def test_worker_session_survives_an_all_dead_pin_set(monkeypatch):
    from tui_gateway import server

    server = _patch_preload(monkeypatch, [], ["seo", "marketing"])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadpin")

    assert isinstance(server._startup_system_prompt({}, "session-1"), (str, type(None)))


def test_interactive_tui_still_fails_loud_without_worker_identity(monkeypatch):
    server = _patch_preload(monkeypatch, [], ["seo"])
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    with pytest.raises(ValueError, match=r"Unknown skill\(s\): seo"):
        server._startup_system_prompt({}, "session-1")
