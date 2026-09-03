"""Regression for #102526.

The launch backend's lazy ``_get_db()`` handle must bind to the import-time
launch home, not whatever ``get_hermes_home()`` resolves to at first-touch
time. The desktop multiplex cron ticker installs per-profile override windows
at startup; if the first ``session.*`` RPC races into a foreign window, the
backend permanently serves the wrong profile's state.db.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server


@pytest.fixture()
def launch_db_env(monkeypatch, tmp_path):
    launch_home = tmp_path / "launch"
    foreign_home = tmp_path / "foreign"
    launch_home.mkdir()
    foreign_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setattr(server, "_hermes_home", str(launch_home))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_error", None)

    captured: list[Path | None] = []

    def _factory(db_path=None, **_kwargs):
        captured.append(Path(db_path) if db_path is not None else None)
        return SimpleNamespace(db_path=db_path)

    monkeypatch.setattr("hermes_state.get_shared_session_db", _factory)
    return launch_home, foreign_home, captured


def test_get_db_first_touch_under_foreign_override_uses_launch_path(launch_db_env):
    launch_home, foreign_home, captured = launch_db_env
    token = set_hermes_home_override(str(foreign_home))
    try:
        db = server._get_db()
        assert db is not None
        assert captured == [launch_home / "state.db"]
        assert server._get_db() is db
    finally:
        reset_hermes_home_override(token)
