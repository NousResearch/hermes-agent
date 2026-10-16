"""Concurrent retry requests share one profile-owned update (TH1-1177)."""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import plugin_updates


def test_retry_joins_running_update_without_crossing_profile(monkeypatch, tmp_path):
    started, release, joined = Event(), Event(), Event()
    calls = []

    class ObservedFuture(Future):
        def result(self, timeout=None):
            joined.set()
            return super().result(timeout)

    monkeypatch.setattr(plugin_updates, "Future", ObservedFuture)

    def invoke(home, operation):
        token = set_hermes_home_override(home)
        try:
            return plugin_updates.join_update("demo", operation)
        finally:
            reset_hermes_home_override(token)

    def update():
        calls.append("update")
        started.set()
        assert release.wait(10)
        return {"ok": True}

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(invoke, tmp_path / "a", update)
        try:
            assert started.wait(10)
            retry = pool.submit(invoke, tmp_path / "a", lambda: calls.append("duplicate"))
            assert joined.wait(10)
            other = pool.submit(invoke, tmp_path / "b", lambda: {"ok": False, "consent_required": True})
            assert other.result(10) == {"ok": False, "consent_required": True}
        finally:
            release.set()
        assert first.result(10) == retry.result(10) == {"ok": True}
    assert calls == ["update"]
    assert invoke(tmp_path / "a", lambda: "recheck pin") == "recheck pin"
