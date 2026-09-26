"""``session.status`` bound by ``session_id`` alone must report the session's own profile home.

Desktop/TUI send ``session.status`` with ``session_id`` and no ``profile`` param. The session record
carries ``profile_home``, but the handler was not ``@_profile_scoped``, so ``display_hermes_home()``
read the process env — on a multiplexed pooled backend that is the LAUNCH profile's home, and the
``Path:`` line made a session look like it lived in another profile (#124500).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tui_gateway.server as server


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch, worker = tmp_path / "launch", tmp_path / "profiles" / "worker"
    for home in (launch, worker):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    return launch, worker


def _status_output(session_id: str) -> str:
    resp = server._methods["session.status"]("rid", {"session_id": session_id})
    return resp["result"]["output"]


def test_session_bound_status_reports_the_session_profile_path(homes):
    launch, worker = homes
    session = {
        "agent": None,
        "profile_home": str(worker),
        "session_key": "worker-session",
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server, "_sessions", {"s-worker": session})
        output = _status_output("s-worker")
    # The Path line is the session's own home, not the launch profile the pooled process env carries.
    assert str(worker) in output
    assert str(launch) not in output


def test_unbound_status_keeps_the_launch_profile_path(homes):
    launch, worker = homes
    session = {
        "agent": None,
        "profile_home": str(launch),
        "session_key": "launch-session",
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server, "_sessions", {"s-launch": session})
        output = _status_output("s-launch")
    assert str(launch) in output
    assert str(worker) not in output
