"""Persistent CDP supervisor for browser dialog + frame detection.

One ``CDPSupervisor`` per Hermes ``task_id`` with a reachable CDP endpoint: one
persistent WebSocket, ``Page`` / ``Runtime`` / ``Target`` events on every attached
session (top page + auto-attached OOPIF / worker targets), pending dialogs + frame
tree exposed via a thread-safe snapshot. Not in the tool schema — output reaches the
agent via ``browser_snapshot`` / ``browser_dialog``. Dialog capture and frame tracking
are mixins (``browser_supervisor_dialogs`` / ``browser_supervisor_frames``).
Design spec: ``website/docs/developer-guide/browser-supervisor.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from tools.browser_supervisor_dialogs import (
    DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S, RECENT_DIALOGS_MAX, _VALID_POLICIES, DialogRecord,
    DialogSupervisionMixin, PendingDialog,
)
from tools.browser_supervisor_frames import FrameInfo, FrameTrackingMixin

# ``websockets`` costs ~22 ms at import and is only needed once a supervisor connects.
if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


def _redact_cdp_error_text(exc: object) -> str:
    """Redact CDP endpoint credentials from an exception's (or URL's) string form.
    ``websockets`` bakes the raw URL (``?token=`` / ``user:pass@``) into its exception
    messages, so every egress point turning one into log/re-raise text MUST route through
    here; falls back to a fixed sentinel if redaction itself raises (err toward masking)."""
    try:
        from agent.redact import redact_cdp_url

        return redact_cdp_url(str(exc))
    except Exception:
        return "<error redacted>"


class _LoopUnavailable(RuntimeError):
    """The supervisor loop refused new work (closed / shutting down)."""


def _schedule(coro, loop, *, timeout: float):
    """Run ``coro`` on the supervisor loop from a sync caller and wait for its result."""
    from agent.async_utils import safe_schedule_threadsafe

    fut = safe_schedule_threadsafe(coro, loop)
    if fut is None:
        raise _LoopUnavailable("Browser supervisor loop unavailable")
    return fut.result(timeout=timeout)


def _fail(error: str) -> Dict[str, Any]:
    return {"ok": False, "error": error}


def _err(exc: BaseException) -> Dict[str, Any]:
    return _fail(f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class SupervisorSnapshot:
    """Read-only snapshot of supervisor state for tool handlers."""

    pending_dialogs: Tuple[PendingDialog, ...]
    recent_dialogs: Tuple[DialogRecord, ...]
    frame_tree: Dict[str, Any]
    active: bool  # False if supervisor is detached/stopped
    cdp_url: str
    task_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for inclusion in ``browser_snapshot`` output."""
        out: Dict[str, Any] = {"pending_dialogs": [d.to_dict() for d in self.pending_dialogs], "frame_tree": self.frame_tree}
        if self.recent_dialogs:
            out["recent_dialogs"] = [d.to_dict() for d in self.recent_dialogs]
        return out


class CDPSupervisor(DialogSupervisionMixin, FrameTrackingMixin):
    """One supervisor per (task_id, cdp_url) pair. ``start()`` spawns a daemon thread
    running its own asyncio loop, connects, attaches to the first page target, enables
    domains and auto-attach. ``snapshot()`` / ``respond_to_dialog()`` / ``evaluate_runtime()``
    are sync, thread-safe bridges onto that loop; all CDP I/O lives on the loop."""

    def __init__(self, task_id: str, cdp_url: str, *, dialog_policy: str = DEFAULT_DIALOG_POLICY,
                 dialog_timeout_s: float = DEFAULT_DIALOG_TIMEOUT_S) -> None:
        if dialog_policy not in _VALID_POLICIES:
            raise ValueError(f"Invalid dialog_policy {dialog_policy!r}; must be one of {sorted(_VALID_POLICIES)}")
        self.task_id = task_id
        self.cdp_url = cdp_url
        self.dialog_policy = dialog_policy
        self.dialog_timeout_s = float(dialog_timeout_s)

        # State protected by ``_state_lock`` for cross-thread reads.
        self._state_lock = threading.Lock()
        self._pending_dialogs: Dict[str, PendingDialog] = {}
        self._recent_dialogs: List[DialogRecord] = []
        self._frames: Dict[str, FrameInfo] = {}
        self._active = False
        # Supervisor loop machinery — populated in start().
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._stop_requested = False
        # CDP call tracking (runs on supervisor loop only).
        self._next_call_id = 1
        self._pending_calls: Dict[int, asyncio.Future] = {}
        self._ws: Optional[ClientConnection] = None
        self._page_session_id: Optional[str] = None
        # Dialog auto-dismiss watchdog handles (per dialog id) + id generator.
        self._dialog_watchdogs: Dict[str, asyncio.TimerHandle] = {}
        self._dialog_seq = 0

    # ── Public sync API ──────────────────────────────────────────────────────

    def start(self, timeout: float = 15.0) -> None:
        """Launch the background loop and block until attachment completes; raises what attach failed with (redacted)."""
        if self._thread and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._start_error, self._stop_requested = None, False
        self._thread = threading.Thread(target=self._thread_main, name=f"cdp-supervisor-{self.task_id}", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            self.stop()
            raise TimeoutError(f"CDP supervisor did not attach within {timeout}s "
                               f"(cdp_url={_redact_cdp_error_text(self.cdp_url)[:80]}...)")
        if (err := self._start_error) is not None:
            self.stop()
            # ``err`` is a raw ``websockets`` exception embedding the full cdp_url (token /
            # userinfo): re-raise redacted, ``from None`` so the traceback chain leaks nothing.
            raise RuntimeError(f"CDP supervisor failed to start: {_redact_cdp_error_text(err)}") from None

    def stop(self, timeout: float = 5.0) -> None:
        """Cancel the supervisor task and join the thread."""
        self._stop_requested = True
        loop = self._loop
        if loop is not None and loop.is_running():
            # Close the WebSocket from inside the loop so ``async for raw in self._ws``
            # returns cleanly, ``_run`` hits its ``finally``, THEN the thread exits.
            with contextlib.suppress(Exception):  # loop already shutting down / close timed out
                _schedule(self._close_ws(), loop, timeout=2.0)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._set_active(False)

    def _set_active(self, value: bool) -> None:
        with self._state_lock:
            self._active = value

    def snapshot(self) -> SupervisorSnapshot:
        """Return an immutable snapshot of current state."""
        with self._state_lock:
            return SupervisorSnapshot(
                pending_dialogs=tuple(self._pending_dialogs.values()),
                recent_dialogs=tuple(self._recent_dialogs[-RECENT_DIALOGS_MAX:]),
                frame_tree=self._build_frame_tree_locked(),
                active=self._active, cdp_url=self.cdp_url, task_id=self.task_id,
            )

    def respond_to_dialog(self, action: str, *, prompt_text: Optional[str] = None,
                          dialog_id: Optional[str] = None, timeout: float = 10.0) -> Dict[str, Any]:
        """Accept/dismiss a pending dialog (sync bridge onto the supervisor loop). Returns
        ``{"ok": True, "dialog"}`` or ``{"ok": False, "error"}`` for recoverable errors."""
        if action not in {"accept", "dismiss"}:
            return _fail(f"action must be 'accept' or 'dismiss', got {action!r}")
        with self._state_lock:
            if not self._active:
                return _fail("supervisor is not active")
            pending = list(self._pending_dialogs.values())
            if not pending:
                return _fail("no dialog is currently open")
            if dialog_id:
                dialog = self._pending_dialogs.get(dialog_id)
                if dialog is None:
                    return _fail(f"dialog_id {dialog_id!r} not found (known: {sorted(self._pending_dialogs)})")
            elif len(pending) > 1:
                return _fail(f"{len(pending)} pending dialogs; specify dialog_id. Candidates: {[d.id for d in pending]}")
            else:
                dialog = pending[0]
        loop = self._loop
        if loop is None:
            return _fail("supervisor loop is not running")
        try:
            coro = self._handle_dialog_cdp(dialog, accept=(action == "accept"), prompt_text=prompt_text or "")
            _schedule(coro, loop, timeout=timeout)
        except _LoopUnavailable as e:
            return _fail(str(e))
        except Exception as e:
            return _err(e)
        return {"ok": True, "dialog": dialog.to_dict()}

    def evaluate_runtime(self, expression: str, *, return_by_value: bool = True,
                         await_promise: bool = True, timeout: float = 10.0) -> Dict[str, Any]:
        """Evaluate ``expression`` in the page's Runtime context over the live WS.
        Returns ``{"ok": True, "result", "result_type"}`` or ``{"ok": False, "error"}``.
        ``return_by_value=True`` JSON-serializes the result (DevTools-console
        semantics); non-serializable objects come back as a description string."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return _fail("supervisor loop is not running")
        with self._state_lock:
            active, session_id = self._active, self._page_session_id
        if not active:
            return _fail("supervisor is not active")
        if not session_id:
            return _fail("supervisor has no attached page session")

        def _run_eval(by_value: bool) -> Dict[str, Any]:
            # userGesture: clipboard / fullscreen APIs need user activation.
            params = {"expression": expression, "returnByValue": by_value,
                      "awaitPromise": await_promise, "userGesture": True}
            coro = self._cdp("Runtime.evaluate", params, session_id=session_id, timeout=timeout)
            return _schedule(coro, loop, timeout=timeout + 1)

        try:
            response = _run_eval(return_by_value)
        except Exception as exc:
            # Deep-serializing live DOM nodes / NodeLists / Window can blow past
            # CDP's recursion guard (``Object reference chain is too long``).
            # Retry once with returnByValue=False so Chrome returns the description.
            if not (return_by_value and "reference chain is too long" in str(exc).lower()):
                return _err(exc)
            try:
                response = _run_eval(False)
            except Exception as exc2:
                return _err(exc2)

        # Response: {"result": {"result": {"type", "value", ...}, "exceptionDetails"?}}
        result_payload = response.get("result", {}) if isinstance(response, dict) else {}
        exception_details = result_payload.get("exceptionDetails")
        if exception_details:
            exc_text = exception_details.get("text") or "JavaScript exception"
            description = (exception_details.get("exception") or {}).get("description")
            return _fail(f"{exc_text}: {description}" if description else exc_text)

        result_obj = result_payload.get("result", {})
        result_type = result_obj.get("type", "undefined")
        if "value" in result_obj:
            value = result_obj["value"]
        elif result_type == "undefined":
            value = None
        else:
            # Non-serializable (functions, DOM nodes…) — give the model the browser's description.
            value = result_obj.get("description") or result_obj.get("unserializableValue")
        return {"ok": True, "result": value, "result_type": result_type}

    # ── Supervisor loop internals ────────────────────────────────────────────

    def _thread_main(self) -> None:
        """Entry point for the supervisor's dedicated thread."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run())
        except BaseException as e:  # noqa: BLE001 — propagate via _start_error
            if not self._fail_start(e):
                logger.warning("CDP supervisor %s crashed: %s", self.task_id, e)
        finally:
            # Cancel + flush remaining tasks before closing the loop to avoid
            # "Task was destroyed but it is pending" warnings.
            with contextlib.suppress(Exception):
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            with contextlib.suppress(Exception):
                loop.close()
            self._set_active(False)

    def _fail_start(self, e: BaseException) -> bool:
        """Propagate ``e`` to ``start()`` if we never got ready; True if it was consumed."""
        if self._ready_event.is_set():
            return False
        self._start_error = e
        self._ready_event.set()
        return True

    async def _close_ws(self) -> None:
        """Detach and close the current WebSocket, swallowing close errors."""
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    async def _run(self) -> None:
        """Top-level reconnecting supervisor coroutine. Browserbase tears down the CDP
        socket whenever a short-lived client (agent-browser's per-command CDP client)
        disconnects, so on drop we reset per-session ids, re-attach, and keep going.
        A failure before the first successful attach is fatal for ``start()``."""
        attempt, last_success_at, backoff = 0, 0.0, 0.5
        import websockets  # deferred: only supervisors that connect pay the import
        while not self._stop_requested:
            try:
                self._ws = await asyncio.wait_for(websockets.connect(self.cdp_url, max_size=50 * 1024 * 1024), timeout=10.0)
            except Exception as e:
                attempt += 1
                if self._fail_start(e):
                    return
                logger.warning("CDP supervisor %s: connect failed (attempt %s): %s",
                               self.task_id, attempt, _redact_cdp_error_text(e))
                await asyncio.sleep(min(backoff, 10.0))
                backoff = min(backoff * 2, 10.0)
                continue

            reader_task = asyncio.create_task(self._read_loop(), name="cdp-reader")
            try:
                # Reset the per-connection page session id; ``_pending_dialogs`` / ``_frames``
                # are deliberately kept — they reconcile as fresh events arrive (worst case a
                # stale dialog entry is rejected with "no dialog is showing", logged only).
                self._page_session_id = None
                await self._attach_initial_page()
                self._set_active(True)
                last_success_at = time.time()
                backoff = 0.5  # reset after a successful attach
                self._ready_event.set()
                await reader_task
            except BaseException as e:
                if self._fail_start(e):
                    raise
                logger.warning("CDP supervisor %s: session dropped after %.1fs: %s",
                               self.task_id, time.time() - last_success_at, _redact_cdp_error_text(e))
            finally:
                self._set_active(False)
                if not reader_task.done():
                    reader_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await reader_task
                for handle in self._dialog_watchdogs.values():
                    handle.cancel()
                self._dialog_watchdogs.clear()
                await self._close_ws()

            if self._stop_requested:
                return
            logger.debug("CDP supervisor %s: reconnecting in %.1fs...", self.task_id, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _attach_initial_page(self) -> None:
        """Find (or create) a page target, attach flattened, enable domains, install dialog bridge."""
        targets = (await self._cdp("Target.getTargets")).get("result", {}).get("targetInfos", [])
        page_target = next((t for t in targets if t.get("type") == "page"), None)
        if page_target is None:
            page_target = (await self._cdp("Target.createTarget", {"url": "about:blank"}))["result"]
        attach = await self._cdp("Target.attachToTarget", {"targetId": page_target["targetId"], "flatten": True})
        self._page_session_id = sid = attach["result"]["sessionId"]
        await self._enable_page_domains(sid, timeout=10.0)
        await self._install_dialog_bridge(sid)

    async def _cdp(self, method: str, params: Optional[Dict[str, Any]] = None, *,
                   session_id: Optional[str] = None, timeout: float = 10.0) -> Dict[str, Any]:
        """Send a CDP command and await its response."""
        if self._ws is None:
            raise RuntimeError("supervisor WebSocket is not connected")
        call_id, self._next_call_id = self._next_call_id, self._next_call_id + 1
        payload: Dict[str, Any] = {"id": call_id, "method": method}
        payload.update({k: v for k, v in (("params", params), ("sessionId", session_id)) if v})
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_calls[call_id] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_calls.pop(call_id, None)

    async def _read_loop(self) -> None:
        """Continuously dispatch incoming CDP frames (responses → futures, events → handlers)."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if self._stop_requested:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    logger.debug("CDP supervisor: non-JSON frame dropped")
                    continue
                if "id" in msg:
                    fut = self._pending_calls.pop(msg["id"], None)
                    if fut is None or fut.done():
                        continue
                    if "error" in msg:
                        fut.set_exception(RuntimeError(f"CDP error on id={msg['id']}: {msg['error']}"))
                    else:
                        fut.set_result(msg)
                elif handler := self._EVENT_HANDLERS.get(msg.get("method")):
                    result = handler(self, msg.get("params", {}), msg.get("sessionId"))
                    if result is not None:
                        await result
        except Exception as e:
            logger.debug("CDP read loop exited: %s", e)

    # CDP event → handler(self, params, session_id). Async handlers return an
    # awaitable that ``_read_loop`` awaits; sync handlers return None.
    _EVENT_HANDLERS: Dict[str, Callable[..., Any]] = {
        **DialogSupervisionMixin.EVENT_HANDLERS, **FrameTrackingMixin.EVENT_HANDLERS
    }


class _SupervisorRegistry:
    """Process-global (task_id → supervisor) map with idempotent start/stop (``SUPERVISOR_REGISTRY``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_task: Dict[str, CDPSupervisor] = {}

    def get(self, task_id: str) -> Optional[CDPSupervisor]:
        with self._lock:
            return self._by_task.get(task_id)

    def _pop(self, task_id: str) -> Optional[CDPSupervisor]:
        with self._lock:
            return self._by_task.pop(task_id, None)

    def get_or_start(self, task_id: str, cdp_url: str, *, dialog_policy: str = DEFAULT_DIALOG_POLICY,
                     dialog_timeout_s: float = DEFAULT_DIALOG_TIMEOUT_S, start_timeout: float = 15.0) -> CDPSupervisor:
        """Idempotently ensure a supervisor runs for ``(task_id, cdp_url)``; one bound to a
        different ``cdp_url`` or unhealthy (dead thread / stopped loop) is stopped and replaced."""
        with self._lock:
            existing = self._by_task.get(task_id)
            if existing is not None:
                thread, loop = existing._thread, existing._loop
                healthy = thread is not None and thread.is_alive() and loop is not None and loop.is_running()
                if existing.cdp_url == cdp_url and healthy:
                    return existing
                self._by_task.pop(task_id, None)
        if existing is not None:
            existing.stop()

        supervisor = CDPSupervisor(task_id=task_id, cdp_url=cdp_url,
                                   dialog_policy=dialog_policy, dialog_timeout_s=dialog_timeout_s)
        supervisor.start(timeout=start_timeout)
        with self._lock:
            # Guard against a concurrent get_or_start from another thread.
            already = self._by_task.get(task_id)
            if already is not None and already.cdp_url == cdp_url:
                supervisor.stop()
                return already
            self._by_task[task_id] = supervisor
        return supervisor

    def stop(self, task_id: str) -> None:
        supervisor = self._pop(task_id)
        if supervisor is not None:
            supervisor.stop()

    def stop_all(self) -> None:
        """Stop every running supervisor. For shutdown / test teardown."""
        with self._lock:
            items = list(self._by_task.values())
            self._by_task.clear()
        for supervisor in items:
            supervisor.stop()


SUPERVISOR_REGISTRY = _SupervisorRegistry()


__all__ = ["CDPSupervisor", "SUPERVISOR_REGISTRY", "SupervisorSnapshot", "_SupervisorRegistry"]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

CONSOLE_HISTORY_MAX = 50

@dataclass
class ConsoleEvent:
    """Ring buffer entry for console + exception traffic."""

    ts: float
    level: str  # "log" | "error" | "warning" | "exception"
    text: str
    url: Optional[str] = None


_PLUGIN_COMPAT_LAZY = {
    'DIALOG_BRIDGE_HOST': ('tools.browser_supervisor_dialogs', 'DIALOG_BRIDGE_HOST'),
    'DIALOG_BRIDGE_URL_PATTERN': ('tools.browser_supervisor_dialogs', 'DIALOG_BRIDGE_URL_PATTERN'),
    'DIALOG_POLICY_AUTO_ACCEPT': ('tools.browser_supervisor_dialogs', 'DIALOG_POLICY_AUTO_ACCEPT'),
    'DIALOG_POLICY_AUTO_DISMISS': ('tools.browser_supervisor_dialogs', 'DIALOG_POLICY_AUTO_DISMISS'),
    'DIALOG_POLICY_MUST_RESPOND': ('tools.browser_supervisor_dialogs', 'DIALOG_POLICY_MUST_RESPOND'),
    'FRAME_TREE_MAX_ENTRIES': ('tools.browser_supervisor_frames', 'FRAME_TREE_MAX_ENTRIES'),
    'FRAME_TREE_MAX_OOPIF_DEPTH': ('tools.browser_supervisor_frames', 'FRAME_TREE_MAX_OOPIF_DEPTH'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
