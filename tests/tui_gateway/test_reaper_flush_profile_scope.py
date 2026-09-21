"""The idle reaper's incremental transcript flush must land in the SESSION's profile home.

``session_reaper._flush_session_messages`` runs on the reaper tick and on the exit-flush worker —
threads with no turn on the stack. Unscoped, ``agent._persist_session`` resolved
``get_hermes_home()`` to the LAUNCH profile, so a served profile's transcript was written into
another tenant's state.db. ``_finalize_session`` binds the same session at the same chokepoint.
"""
import threading

import pytest

from hermes_constants import get_hermes_home
from tui_gateway import server as tui_server


class _Agent:
    def __init__(self, seen):
        self._session_messages = [{"role": "user", "content": "hi"}]
        self._seen = seen

    def _persist_session(self, _messages):
        self._seen.append(str(get_hermes_home()))


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch, served = tmp_path / "launch", tmp_path / "served"
    for home in (launch, served):
        home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    return launch, served


def test_incremental_flush_persists_into_the_sessions_own_home(homes, monkeypatch):
    launch, served = homes
    seen: list[str] = []
    session = {"agent": _Agent(seen), "profile_home": str(served)}
    monkeypatch.setattr(tui_server, "_sessions", {"sid": session}, raising=False)
    monkeypatch.setattr(tui_server, "_sessions_lock", threading.RLock(), raising=False)
    monkeypatch.setattr(tui_server, "_INCREMENTAL_FLUSH_INTERVAL_S", 30.0, raising=False)

    assert tui_server._flush_dirty_sessions(now=1000.0) == 1
    assert seen == [str(served)], f"transcript flushed into {seen} instead of the served home"
    assert str(launch) not in seen
