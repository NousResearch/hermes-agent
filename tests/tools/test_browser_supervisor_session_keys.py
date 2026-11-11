"""One CDP supervisor per browser session, keyed like the session itself.

browser_exec attaches a supervisor (dialogs, frame tree, vault fill) to the browser each call drives. When
the session is torn down the browser goes away, so a supervisor left attached keeps a WebSocket into a
stopped browser and retries it. ``browser_dialog`` acts on the session the task last drove.
"""

import asyncio
import json
import stat
import threading

import pytest
import websockets

import tools.browser_use_cli as bu_cli
from tests.waiting import wait_until
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud
from tools import browser_session_lifecycle as lifecycle
from tools.browser_supervisor import SUPERVISOR_REGISTRY


class _CDPEndpoint:
    """A loopback CDP browser endpoint: one page target; every other command succeeds empty.
    ``open_dialog`` makes the page raise a native dialog on every attached connection."""

    def __init__(self, name: str):
        self.name = name
        self.url = ""
        self._clients: set = set()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)

    async def _handle(self, ws):
        self._clients.add(ws)
        try:
            async for raw in ws:
                msg = json.loads(raw)
                result = {}
                if msg["method"] == "Target.getTargets":
                    result = {"targetInfos": [{"targetId": "page-1", "type": "page", "url": "about:blank"}]}
                elif msg["method"] == "Target.attachToTarget":
                    result = {"sessionId": "session-1"}
                await ws.send(json.dumps({"id": msg["id"], "result": result}))
        finally:
            self._clients.discard(ws)

    async def _serve(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/devtools/browser/{self.name}"

    def open_dialog(self, message: str) -> None:
        event = json.dumps({"method": "Page.javascriptDialogOpening", "sessionId": "session-1",
                            "params": {"type": "alert", "message": message, "url": "about:blank"}})

        async def _send():
            for ws in list(self._clients):
                await ws.send(event)
        asyncio.run_coroutine_threadsafe(_send(), self._loop).result(timeout=10)

    def __enter__(self):
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._serve(), self._loop).result(timeout=10)
        return self

    def __exit__(self, *exc):
        async def _close():
            self._server.close()
            await self._server.wait_closed()
        asyncio.run_coroutine_threadsafe(_close(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)


class _CloudProvider:
    """A cloud browser provider (the ``BrowserProvider`` lifecycle trio); each session key gets the
    endpoint mapped to it, else the default one."""

    name = "loopback"

    def __init__(self, cdp_url: str, by_key: dict | None = None):
        self.cdp_url = cdp_url
        self.by_key = by_key or {}
        self.closed: list[str] = []

    def is_available(self) -> bool:
        return True

    def create_session(self, task_id: str) -> dict:
        return {"session_name": f"lb-{task_id}", "bb_session_id": f"bb-{task_id}",
                "cdp_url": self.by_key.get(task_id, self.cdp_url), "features": {}}

    def close_session(self, session_id: str) -> bool:
        self.closed.append(session_id)
        return True

    def emergency_cleanup(self, session_id: str) -> None:
        self.closed.append(session_id)


@pytest.fixture
def session_state(tmp_path, monkeypatch):
    for name in ("_active_sessions", "_session_last_activity", "_session_owner_homes", "_cleanup_failures",
                 "_task_driven_keys"):
        monkeypatch.setattr(bs, name, {})
    monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: None)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.delenv("BROWSER_USE_API_KEY", raising=False)
    cli = tmp_path / "browser-harness"
    cli.write_text("#!/bin/sh\ncat > /dev/null\necho ok\n", encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])
    bs_cloud.reset_caches()
    yield
    SUPERVISOR_REGISTRY.stop_all()
    bs_cloud.reset_caches()


def _supervisors_on(cdp_url: str) -> list[str]:
    with SUPERVISOR_REGISTRY._lock:
        return sorted(key for key, sup in SUPERVISOR_REGISTRY._by_task.items() if sup.cdp_url == cdp_url)


def _exec(session: str, task_id: str) -> dict:
    from model_tools import handle_function_call
    return json.loads(handle_function_call("browser_exec", {"code": "print(1)", "session": session}, task_id=task_id))


def test_named_session_teardown_stops_every_supervisor_on_its_browser(monkeypatch, session_state):
    with _CDPEndpoint("cloud-1") as endpoint:
        provider = _CloudProvider(endpoint.url)
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)
        assert _exec("research", "t-named")["success"] is True
        assert _supervisors_on(endpoint.url)

        # The named session idles out; the janitor tears it down and the provider closes the browser.
        monkeypatch.setattr(bs, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 0)
        lifecycle.cleanup_inactive_sessions()
        assert provider.closed == ["bb-bu-named-research"]

        assert _supervisors_on(endpoint.url) == []


def test_browser_dialog_answers_the_dialog_of_the_session_the_task_last_drove(monkeypatch, session_state):
    from model_tools import handle_function_call

    with _CDPEndpoint("research") as research, _CDPEndpoint("notes") as notes:
        provider = _CloudProvider(research.url, {"bu-named-notes": notes.url})
        monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: provider)
        assert _exec("research", "t-two")["success"] is True
        assert _exec("notes", "t-two")["success"] is True
        assert _supervisors_on(research.url) == ["bu-named-research"]
        assert _supervisors_on(notes.url) == ["bu-named-notes"]

        research.open_dialog("asked by research")
        notes.open_dialog("asked by notes")
        for key in ("bu-named-research", "bu-named-notes"):
            wait_until(lambda key=key: SUPERVISOR_REGISTRY.get(key).snapshot().pending_dialogs,
                       what=f"{key} dialog captured")

        result = json.loads(handle_function_call("browser_dialog", {"action": "accept"}, task_id="t-two"))
        assert result["success"] is True, result
        assert result["dialog"]["message"] == "asked by notes"
