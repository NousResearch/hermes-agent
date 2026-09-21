"""The reaper/exit-flush transcript persistence: right home, no external-secret hydration.

``session_reaper._flush_session_messages`` runs on the reaper tick and on the exit-flush worker —
threads with no turn on the stack. The transcript ROWS were never at risk (a profile session gets a
dedicated ``SessionDB`` whose ``db_path`` is frozen at construction), but everything
``agent._persist_session`` resolves at CALL time — config.yaml, the token ledger, memory/provider
lookups — resolved against the LAUNCH profile for a served profile's flush. ``_finalize_session``
binds the same session at the same chokepoint.

Binding that scope must stay CHEAP: it happens inside a 5s total exit-flush budget, and hydrating
the profile's external secret sources shells out to the operator's secret command (30s CLI budget)
— persisting a transcript needs no external credential.
"""
import threading
import time

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


def _install_session(monkeypatch, session):
    monkeypatch.setattr(tui_server, "_sessions", {"sid": session}, raising=False)
    monkeypatch.setattr(tui_server, "_sessions_lock", threading.RLock(), raising=False)
    monkeypatch.setattr(tui_server, "_INCREMENTAL_FLUSH_INTERVAL_S", 30.0, raising=False)


def test_incremental_flush_persists_into_the_sessions_own_home(homes, monkeypatch):
    launch, served = homes
    seen: list[str] = []
    _install_session(monkeypatch, {"agent": _Agent(seen), "profile_home": str(served)})

    assert tui_server._flush_dirty_sessions(now=1000.0) == 1
    assert seen == [str(served)], f"transcript flushed into {seen} instead of the served home"
    assert str(launch) not in seen


def test_exit_flush_never_waits_on_an_external_secret_source(homes, monkeypatch):
    """A slow ``op run`` / ``bws`` source must not cost the transcript the exit flush exists to save."""
    _launch, served = homes
    from hermes_cli import env_loader

    def _slow_source(_home):
        time.sleep(2.0)  # a real secret CLI gets a 30s budget; this whole flush gets 1s
        return {}

    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", _slow_source)
    _install_session(monkeypatch, {"agent": _Agent([]), "profile_home": str(served)})

    assert tui_server._flush_sessions_before_exit(budget_s=1.0) == 1


def test_exit_flush_warns_when_it_loses_a_transcript(homes, monkeypatch, caplog):
    """Both callers discard the return value, so an overrun is invisible without this line."""
    _launch, served = homes

    class _Stuck(_Agent):
        def _persist_session(self, _messages):
            time.sleep(2.0)

    _install_session(monkeypatch, {"agent": _Stuck([]), "profile_home": str(served)})

    with caplog.at_level("WARNING"):
        assert tui_server._flush_sessions_before_exit(budget_s=0.2) == 0
    assert any("Exit flush persisted 0 of 1" in r.getMessage() for r in caplog.records), caplog.text
