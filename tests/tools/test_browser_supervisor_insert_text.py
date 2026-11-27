"""``CDPSupervisor.insert_text`` and a pinned ``evaluate_runtime`` reach only the page session the
caller validated.

A real ``CDPSupervisor`` talks to a local WebSocket server that speaks just enough CDP: one http
page target, and a fresh session id per ``Target.attachToTarget`` (as Chrome hands out). The test
records which session every ``Input.insertText`` frame was sent on.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from tools.browser_supervisor import _SupervisorRegistry


class _FakeCDP:
    def __init__(self) -> None:
        self.inserts: list[tuple[str, str | None]] = []  # (text, sessionId)
        self.evaluates: list[tuple[str, str | None]] = []  # (expression, sessionId)
        self.attaches = 0
        self.url = ""
        self.before_evaluate_reply: list[dict] = []  # CDP events sent ahead of the next evaluate's reply
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._serve, args=(ready,), daemon=True).start()
        assert ready.wait(5)

    def _serve(self, ready: threading.Event) -> None:
        from websockets.asyncio.server import serve

        async def main():
            async with serve(self._handle, "127.0.0.1", 0) as server:
                self.url = f"ws://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}/devtools/browser/x"
                ready.set()
                await asyncio.Event().wait()

        self.loop.run_until_complete(main())

    def _targets(self, msg: dict) -> dict:
        return {"targetInfos": [{"targetId": "T1", "type": "page", "url": "https://example.com/login"}]}

    def _attach(self, msg: dict) -> dict:
        self.attaches += 1
        return {"sessionId": f"S{self.attaches}"}

    def _insert(self, msg: dict) -> dict:
        self.inserts.append((msg["params"]["text"], msg.get("sessionId")))
        return {}

    def _evaluate(self, msg: dict) -> dict:
        self.evaluates.append((msg["params"]["expression"], msg.get("sessionId")))
        return {"result": {"type": "string", "value": "ok"}}

    def push(self, event: dict) -> None:
        """Send a CDP event to the supervisor, as Chrome does when the page navigates."""
        asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps(event)), self.loop).result(5)

    async def _handle(self, ws) -> None:
        self.ws = ws
        handlers = {"Target.getTargets": self._targets, "Target.attachToTarget": self._attach,
                    "Input.insertText": self._insert, "Runtime.evaluate": self._evaluate}
        async for raw in ws:
            msg = json.loads(raw)
            handler = handlers.get(msg["method"])
            result = handler(msg) if handler else {}
            if msg["method"] == "Runtime.evaluate":
                while self.before_evaluate_reply:
                    await ws.send(json.dumps(self.before_evaluate_reply.pop(0)))
            await ws.send(json.dumps({"id": msg["id"], "result": result}))


@pytest.fixture
def supervisor():
    cdp = _FakeCDP()
    reg = _SupervisorRegistry()
    sup = reg.get_or_start("t", cdp.url, start_timeout=5)
    try:
        yield sup, cdp
    finally:
        reg.stop_all()


def _checked(sup, session_id):
    """The document a focus check on ``session_id`` ran on, as the vault reads it."""
    out = sup.evaluate_runtime("focusCheck()", session_id=session_id)
    assert out["ok"] is True
    return out["document"]


def _navigate(sup, cdp, session_id, *, url, parent=None):
    """Commit a document in a frame of ``session_id`` and wait until the supervisor saw it."""
    frame_id = "child" if parent else "main"
    frame = {"id": frame_id, "url": url, "securityOrigin": url.rsplit("/", 1)[0], **({"parentId": parent} if parent else {})}
    cdp.push({"method": "Page.frameNavigated", "sessionId": session_id, "params": {"frame": frame}})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with sup._state_lock:
            seen = sup._frames.get(frame_id)
        if seen is not None and seen.url == url:
            return
        time.sleep(0.01)
    raise AssertionError("the supervisor never saw the navigation")


def test_insert_goes_to_the_pinned_session(supervisor):
    sup, cdp = supervisor
    pinned = sup.current_page_session()
    assert sup.insert_text("abc", session_id=pinned, document=_checked(sup, pinned)) == {"ok": True}
    assert cdp.inserts == [("abc", pinned)]


def test_insert_refuses_once_the_checked_document_was_replaced(supervisor):
    """A CDP session belongs to the tab, not to the document: a redirect that commits after the
    focus check but before the keystroke keeps the session id. The text checked on the old
    document must not land in the new one's focused input."""
    sup, cdp = supervisor
    pinned = sup.current_page_session()
    document = _checked(sup, pinned)
    _navigate(sup, cdp, pinned, url="https://evil.test/phish")

    out = sup.insert_text("s3cret", session_id=pinned, document=document)
    assert out["ok"] is False and out["code"] == "document_changed"
    assert cdp.inserts == []
    # A check on the new document vouches for that one.
    assert sup.insert_text("x", session_id=pinned, document=_checked(sup, pinned)) == {"ok": True}


def test_a_navigation_while_the_focus_check_runs_refuses_the_keystroke(supervisor):
    """The check ran on the document that was current when it was sent. A redirect that commits while
    it runs (its event arrives before the check's reply) is a document the check never looked at."""
    sup, cdp = supervisor
    pinned = sup.current_page_session()
    cdp.before_evaluate_reply.append({"method": "Page.frameNavigated", "sessionId": pinned, "params": {
        "frame": {"id": "main", "url": "https://evil.test/phish", "securityOrigin": "https://evil.test"}}})
    document = _checked(sup, pinned)

    out = sup.insert_text("s3cret", session_id=pinned, document=document)
    assert out["ok"] is False and out["code"] == "document_changed"
    assert cdp.inserts == []


def test_a_subframe_navigation_does_not_cancel_typing(supervisor):
    """An ad or analytics iframe reloading is not the page the keystroke goes to."""
    sup, cdp = supervisor
    pinned = sup.current_page_session()
    document = _checked(sup, pinned)
    _navigate(sup, cdp, pinned, url="https://ads.test/frame", parent="main")
    assert sup.insert_text("abc", session_id=pinned, document=document) == {"ok": True}


def test_insert_refuses_once_the_supervisor_moved_to_another_session(supervisor):
    """Between the caller's focus check and its keystroke, another tool can retarget the supervisor
    (focus_page attaches a new session). The keystroke must not follow it to the new page."""
    sup, cdp = supervisor
    pinned = sup.current_page_session()
    assert sup.focus_page("https://example.com")["ok"] is True
    assert sup.current_page_session() != pinned

    out = sup.insert_text("s3cret", session_id=pinned, document=0)
    assert out["ok"] is False
    assert cdp.inserts == []


def test_a_pinned_evaluate_refuses_once_the_supervisor_moved(supervisor):
    """The vault's read-back carries the expected values: pinned to the session the fill validated,
    it runs there or nowhere. Unpinned, it follows the supervisor (the browser_exec contract)."""
    sup, cdp = supervisor
    pinned = sup.current_page_session()
    assert sup.evaluate_runtime("check('s3cret')", session_id=pinned)["ok"] is True
    assert sup.focus_page("https://example.com")["ok"] is True
    moved = sup.current_page_session()

    assert sup.evaluate_runtime("check('s3cret')", session_id=pinned)["ok"] is False
    assert sup.evaluate_runtime("1 + 1")["ok"] is True
    assert [sid for expr, sid in cdp.evaluates if "s3cret" in expr] == [pinned]
    assert cdp.evaluates[-1] == ("1 + 1", moved)


def test_focus_page_names_the_session_it_attached(supervisor):
    """A vault fill pins the session its origin check ran on: ``focus_page`` reports the session it
    attached, and that is the one the supervisor then holds."""
    sup, _cdp = supervisor
    before = sup.current_page_session()
    out = sup.focus_page("https://example.com")
    assert out["ok"] is True
    assert out["session_id"] == sup.current_page_session() != before
