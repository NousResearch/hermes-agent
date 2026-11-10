"""Background delegations cannot outlive the admission proof used to retire their process."""

import threading


def test_async_delegations_hold_busy_accounting_through_finalization(tmp_path, monkeypatch):
    from hermes_cli import backend_retirement
    from hermes_cli.web_server_idle_proof import idle_proof
    from tools import async_delegation

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    entered, release = threading.Event(), threading.Event()

    def runner():
        entered.set()
        assert release.wait(10)
        return {"summary": "test done"}

    def dispatch():
        return async_delegation.dispatch_async_delegation(
            goal="test", context=None, toolsets=None, role="leaf", model=None,
            session_key="test", runner=runner)

    token = fence.prepare()["token"]
    try:
        assert dispatch()["status"] == "rejected"
        assert not entered.is_set()
        assert fence.cancel(token) == {"ok": True}
        handle = dispatch()
        assert handle["status"] == "dispatched"
        assert entered.wait(10)
        assert async_delegation.active_count() > 0
        assert idle_proof()["idle"] is False
        assert fence.prepare() == {"ok": False, "idle": False}
        # A stalled record may be force-finalized before its stuck runner really unwinds.
        async_delegation._finalize(handle["delegation_id"], {"error": "test stall"}, "stalled")
        assert async_delegation.active_count() == 0
        assert fence.prepare() == {"ok": False, "idle": False}
    finally:
        release.set()
        if async_delegation._executor is not None:
            async_delegation._executor.shutdown(wait=True)
    assert fence.prepare()["ok"] is True
