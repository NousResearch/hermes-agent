"""Concurrent retry requests share one profile-owned update."""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def test_retry_joins_running_update_without_crossing_profile(monkeypatch, tmp_path):
    from tui_gateway import plugin_updates
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


def test_gateway_retry_joins_consent_refusal_then_accepts_on_new_call(monkeypatch):
    """A retry cannot skip pending consent or race the original plugin update."""
    from unittest.mock import patch

    from tui_gateway import plugin_updates, server

    started, release, joined = Event(), Event(), Event()
    calls = []

    class ObservedFuture(Future):
        def result(self, timeout=None):
            joined.set()
            return super().result(timeout)

    monkeypatch.setattr(plugin_updates, "Future", ObservedFuture)

    def update(name, *, accept_capabilities=False, force=False):
        calls.append((name, accept_capabilities, force))
        if not accept_capabilities:
            started.set()
            assert release.wait(10)
            return {"ok": False, "name": name, "consent_required": True, "sha": "f" * 40,
                    "delta_lines": ["New capability requires consent"]}
        return {"ok": True, "name": name, "unchanged": True}

    def request(accept=False):
        params = {"action": "update", "name": "demo"}
        if accept:
            params["accept_capabilities"] = True
        return server.handle_request({"id": "1", "method": "plugins.manage", "params": params})

    with patch("hermes_cli.plugins_cmd.dashboard_update_user_plugin", side_effect=update):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(request)
            try:
                assert started.wait(10)
                retry = pool.submit(request, True)
                assert joined.wait(10)
            finally:
                release.set()
            original = first.result(10)
            coalesced = retry.result(10)

        assert original["result"]["consent_required"] is True
        assert coalesced["result"] == original["result"]
        assert calls == [("demo", False, False)]

        fresh = request(True)
        assert fresh["result"]["ok"] is True
        assert calls == [("demo", False, False), ("demo", True, False)]
