"""Regression for #102526.

The launch backend's lazy ``_get_db()`` handle must bind to the import-time
launch home, not whatever ``get_hermes_home()`` resolves to at first-touch
time. The desktop multiplex cron ticker installs per-profile override windows
at startup; if the first ``session.*`` RPC races into a foreign window, the
backend permanently serves the wrong profile's state.db.
"""

from __future__ import annotations

import pytest

import hermes_state_registry as registry
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
    try:
        yield launch_home, foreign_home
    finally:
        registry.close_all()


def test_get_db_first_touch_under_foreign_override_uses_launch_path(launch_db_env):
    launch_home, foreign_home = launch_db_env
    token = set_hermes_home_override(str(foreign_home))
    try:
        db = server._get_db()
        assert db is not None
        assert db.db_path.resolve() == (launch_home / "state.db").resolve()
        assert not (foreign_home / "state.db").exists()
        assert server._get_db() is db
    finally:
        reset_hermes_home_override(token)
