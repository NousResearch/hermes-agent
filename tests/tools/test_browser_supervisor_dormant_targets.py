"""Restored pages must be activated before renderer-dependent CDP setup."""

import asyncio

import pytest

from tools import browser_supervisor as bs


class RestoredTargets:
    def __init__(self, targets, *, broken=()):
        self.targets = targets
        self.broken = set(broken)
        self.awake = set()
        self.sessions = set()
        self.calls = []

    async def call(self, method, params=None, *, session_id=None, timeout=10.0):
        params = params or {}
        self.calls.append((method, params, session_id))
        if method == "Target.getTargets":
            return {"result": {"targetInfos": self.targets}}
        if method == "Target.attachToTarget":
            sid = params["targetId"]
            self.sessions.add(sid)
            return {"result": {"sessionId": sid}}
        if method == "Target.activateTarget":
            self.awake.add(params["targetId"])
        if method == "Target.detachFromTarget":
            self.sessions.remove(params["sessionId"])
        if method == "Page.enable" and (session_id not in self.awake or session_id in self.broken):
            raise TimeoutError()
        if method == "Runtime.evaluate":
            return {"result": {"result": {"value": True}}}
        return {"result": {}}


def page(target_id, url="https://example.test/login"):
    return {"targetId": target_id, "type": "page", "url": url}


async def select(supervisor, mode, timeout=2.0):
    if mode == "startup":
        await supervisor._attach_initial_page()
        return {"ok": True}
    supervisor._loop = asyncio.get_running_loop()
    return await asyncio.to_thread(supervisor.focus_page, "https://example.test", accept="true", timeout=timeout)


@pytest.mark.parametrize("mode", ["startup", "focus"])
def test_restored_page_is_woken_before_setup(monkeypatch, mode):
    async def run():
        transport = RestoredTargets([page("restored", "about:blank" if mode == "startup" else "https://example.test/login")])
        supervisor = bs.CDPSupervisor("test", "ws://unused")
        monkeypatch.setattr(supervisor, "_cdp", transport.call)
        result = await select(supervisor, mode)
        assert result["ok"], result
        assert supervisor._page_session_id == "restored"
        assert transport.awake == {"restored"}
        assert not any(method in {"Target.closeTarget", "Page.navigate"} for method, _, _ in transport.calls)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["startup", "focus"])
def test_failed_candidate_is_detached_before_trying_next(monkeypatch, mode):
    async def run():
        transport = RestoredTargets([page("broken"), page("usable")], broken={"broken"})
        supervisor = bs.CDPSupervisor("test", "ws://unused")
        supervisor._page_session_id = "previous"
        monkeypatch.setattr(supervisor, "_cdp", transport.call)
        result = await select(supervisor, mode)
        assert result["ok"], result
        assert supervisor._page_session_id == "usable"
        assert transport.sessions == {"usable"}
        methods = [(method, params) for method, params, _ in transport.calls]
        assert methods.index(("Target.detachFromTarget", {"sessionId": "broken"})) < methods.index(
            ("Target.attachToTarget", {"targetId": "usable", "flatten": True})
        )

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["startup", "focus"])
@pytest.mark.parametrize("stalled_method", ["Page.enable", "Page.addScriptToEvaluateOnNewDocument", "Runtime.evaluate"])
def test_selection_deadline_cancels_stalled_setup_and_cleans_session(monkeypatch, mode, stalled_method):
    async def run():
        transport = RestoredTargets([page(str(i)) for i in range(96)])
        supervisor = bs.CDPSupervisor("test", "ws://unused")
        supervisor._page_session_id = "previous"
        blocked = asyncio.Event()
        cancelled = asyncio.Event()

        async def call(method, *args, **kwargs):
            if method == stalled_method:
                try:
                    await blocked.wait()
                finally:
                    cancelled.set()
            return await transport.call(method, *args, **kwargs)

        monkeypatch.setattr(supervisor, "_cdp", call)
        monkeypatch.setattr(bs, "PAGE_SELECTION_TIMEOUT_S", 0.05, raising=False)
        monkeypatch.setattr(bs, "PAGE_CANDIDATE_TIMEOUT_S", 0.02, raising=False)
        safety = asyncio.timeout(2.0)
        try:
            async with safety:
                result = await select(supervisor, mode, timeout=0.05)
        except TimeoutError:
            result = {"ok": False}
        assert not safety.expired(), "selection exhausted the outer safety deadline"
        assert not result["ok"]
        assert cancelled.is_set()
        assert not transport.sessions
        assert supervisor._page_session_id == "previous"
        assert len(transport.awake) < 96

    asyncio.run(run())


def test_start_failure_identifies_blank_timeout(monkeypatch):
    supervisor = bs.CDPSupervisor("test", "ws://unused")

    async def fail():
        raise TimeoutError()

    monkeypatch.setattr(supervisor, "_run", fail)
    with pytest.raises(RuntimeError, match="TimeoutError"):
        supervisor.start()


def test_crash_diagnostic_redacts_endpoint_credentials(monkeypatch, caplog):
    supervisor = bs.CDPSupervisor("test", "ws://unused")
    supervisor._ready_event.set()

    async def fail():
        raise TimeoutError("ws://user:private-pass@example.test/devtools?token=private-token")

    monkeypatch.setattr(supervisor, "_run", fail)
    supervisor._thread_main()
    assert "TimeoutError" in caplog.text
    assert "private-pass" not in caplog.text
    assert "private-token" not in caplog.text


@pytest.mark.parametrize("mode", ["startup", "focus"])
def test_unresponsive_detach_does_not_block_next_candidate(monkeypatch, mode):
    async def run():
        transport = RestoredTargets([page("broken"), page("usable")], broken={"broken"})
        supervisor = bs.CDPSupervisor("test", "ws://unused")
        cancelled = asyncio.Event()

        async def call(method, *args, **kwargs):
            if method == "Target.detachFromTarget":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return await transport.call(method, *args, **kwargs)

        monkeypatch.setattr(supervisor, "_cdp", call)
        monkeypatch.setattr(bs, "PAGE_DETACH_TIMEOUT_S", 0.02)
        async with asyncio.timeout(2.0):
            result = await select(supervisor, mode)
        assert result["ok"], result
        assert cancelled.is_set()
        assert supervisor._page_session_id == "usable"

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["activate", "bridge", "probe", "reject"])
def test_focus_preserves_previous_session_and_origin_filter(monkeypatch, failure):
    async def run():
        transport = RestoredTargets([
            page("foreign", "https://foreign.test/login"),
            page("internal", "chrome://settings"),
            page("candidate"),
        ])
        supervisor = bs.CDPSupervisor("test", "ws://unused")
        supervisor._page_session_id = "previous"

        async def call(method, params=None, **kwargs):
            if method == "Target.activateTarget" and failure == "activate":
                raise RuntimeError("target disappeared")
            if method == "Runtime.evaluate" and (params or {}).get("expression") == "true":
                if failure == "probe":
                    raise RuntimeError("context destroyed")
                return {"result": {"result": {"value": False}}}
            return await transport.call(method, params, **kwargs)

        async def broken_bridge(sid):
            raise RuntimeError("bridge setup failed")

        monkeypatch.setattr(supervisor, "_cdp", call)
        if failure == "bridge":
            monkeypatch.setattr(supervisor, "_install_dialog_bridge", broken_bridge)
        result = await select(supervisor, "focus")
        assert not result["ok"]
        assert supervisor._page_session_id == "previous"
        assert not transport.sessions
        assert transport.awake <= {"candidate"}
        assert not any(method in {"Target.closeTarget", "Page.navigate", "Target.createTarget"}
                       for method, _, _ in transport.calls)

    asyncio.run(run())
