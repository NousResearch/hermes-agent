"""Persistent CDP supervisor for browser dialog + frame detection.

One ``CDPSupervisor`` runs per Hermes ``task_id`` that has a reachable CDP
endpoint. It holds a single persistent WebSocket to the backend, subscribes
to ``Page`` / ``Runtime`` / ``Target`` events on every attached session
(top-level page and every OOPIF / worker target that auto-attaches), and
surfaces observable state — pending dialogs and frame tree — through a
thread-safe snapshot object that tool handlers consume synchronously.

The supervisor is NOT in the agent's tool schema. Its output reaches the
agent via two channels:

1. ``browser_snapshot`` merges supervisor state into its return payload
   (see ``tools/browser_tool.py``).
2. ``browser_dialog`` tool responds to a pending dialog by calling
   ``respond_to_dialog()`` on the active supervisor.

Design spec: ``website/docs/developer-guide/browser-supervisor.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


# ── Config defaults ───────────────────────────────────────────────────────────

DIALOG_POLICY_MUST_RESPOND = "must_respond"
DIALOG_POLICY_AUTO_DISMISS = "auto_dismiss"
DIALOG_POLICY_AUTO_ACCEPT = "auto_accept"

_VALID_POLICIES = frozenset(
    {DIALOG_POLICY_MUST_RESPOND, DIALOG_POLICY_AUTO_DISMISS, DIALOG_POLICY_AUTO_ACCEPT}
)

DEFAULT_DIALOG_POLICY = DIALOG_POLICY_MUST_RESPOND
DEFAULT_DIALOG_TIMEOUT_S = 300.0

# Snapshot caps for frame_tree — keep payloads bounded on ad-heavy pages.
FRAME_TREE_MAX_ENTRIES = 30
FRAME_TREE_MAX_OOPIF_DEPTH = 2

# Ring buffer of recent console-level events (used later by PR 2 diagnostics).
CONSOLE_HISTORY_MAX = 50

# Keep the last N closed dialogs in ``recent_dialogs`` so agents on backends
# that auto-dismiss server-side (e.g. Browserbase) can still observe that a
# dialog fired, even if they couldn't respond to it in time.
RECENT_DIALOGS_MAX = 20

# Magic host the injected dialog bridge XHRs to.  Intercepted via the CDP
# Fetch domain before any network resolution happens, so the hostname never
# has to exist.  Keep this ASCII + URL-safe; we also gate Fetch patterns on it.
DIALOG_BRIDGE_HOST = "hermes-dialog-bridge.invalid"
DIALOG_BRIDGE_URL_PATTERN = f"http://{DIALOG_BRIDGE_HOST}/*"

# Script injected into every frame via Page.addScriptToEvaluateOnNewDocument.
# Overrides alert/confirm/prompt to round-trip through a sync XHR that we
# intercept via Fetch.requestPaused. Works on Browserbase (whose CDP proxy
# auto-dismisses REAL native dialogs) because the native dialogs never fire
# in the first place — the overrides take precedence.
_DIALOG_BRIDGE_SCRIPT = r"""
(() => {
  if (window.__hermesDialogBridgeInstalled) return;
  window.__hermesDialogBridgeInstalled = true;
  const ENDPOINT = "http://hermes-dialog-bridge.invalid/";
  function ask(kind, message, defaultPrompt) {
    try {
      const xhr = new XMLHttpRequest();
      // Use GET with query params so we don't need to worry about request
      // body encoding in the Fetch interceptor.
      const params = new URLSearchParams({
        kind: String(kind || ""),
        message: String(message == null ? "" : message),
        default_prompt: String(defaultPrompt == null ? "" : defaultPrompt),
      });
      xhr.open("GET", ENDPOINT + "?" + params.toString(), false);  // sync
      xhr.send(null);
      if (xhr.status !== 200) return null;
      const body = xhr.responseText || "";
      let parsed;
      try { parsed = JSON.parse(body); } catch (e) { return null; }
      if (kind === "alert") return undefined;
      if (kind === "confirm") return Boolean(parsed && parsed.accept);
      if (kind === "prompt") {
        if (!parsed || !parsed.accept) return null;
        return parsed.prompt_text == null ? "" : String(parsed.prompt_text);
      }
      return null;
    } catch (e) {
      // If the bridge is unreachable, fall back to the native call so the
      // page still sees *some* behavior (the backend will auto-dismiss).
      return null;
    }
  }
  const realAlert   = window.alert;
  const realConfirm = window.confirm;
  const realPrompt  = window.prompt;
  window.alert   = function(message) { ask("alert",   message, ""); };
  window.confirm = function(message) {
    const r = ask("confirm", message, "");
    return r === null ? false : Boolean(r);
  };
  window.prompt  = function(message, def) {
    const r = ask("prompt", message, def == null ? "" : def);
    return r === null ? null : String(r);
  };
  // onbeforeunload — we can't really synchronously prompt the user from this
  // event without racing navigation.  Leave native behavior for now; the
  // supervisor's native-dialog fallback path still surfaces them in
  // recent_dialogs.
})();
"""


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class PendingDialog:
    """A JS dialog currently open on some frame's session."""

    id: str
    type: str  # "alert" | "confirm" | "prompt" | "beforeunload"
    message: str
    default_prompt: str
    opened_at: float
    cdp_session_id: str  # which attached CDP session the dialog fired in
    frame_id: Optional[str] = None
    # When set, the dialog was captured via the bridge XHR path (Fetch domain).
    # Response must be delivered via Fetch.fulfillRequest, NOT
    # Page.handleJavaScriptDialog — the native dialog never fired.
    bridge_request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "message": self.message,
            "default_prompt": self.default_prompt,
            "opened_at": self.opened_at,
            "frame_id": self.frame_id,
        }


@dataclass
class DialogRecord:
    """A historical record of a dialog that was opened and then handled.

    Retained in ``recent_dialogs`` for a short window so agents on backends
    that auto-dismiss dialogs server-side (Browserbase) can still observe
    that a dialog fired, even though they couldn't respond to it.
    """

    id: str
    type: str
    message: str
    opened_at: float
    closed_at: float
    closed_by: str  # "agent" | "auto_policy" | "remote" | "watchdog"
    frame_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "message": self.message,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "closed_by": self.closed_by,
            "frame_id": self.frame_id,
        }


@dataclass
class FrameInfo:
    """One frame in the page's frame tree.

    ``is_oopif`` means the frame has its own CDP target (separate process,
    reachable via ``cdp_session_id``). Same-origin / srcdoc iframes share
    the parent process and have ``is_oopif=False`` + ``cdp_session_id=None``.
    """

    frame_id: str
    url: str
    origin: str
    parent_frame_id: Optional[str]
    is_oopif: bool
    cdp_session_id: Optional[str] = None
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "frame_id": self.frame_id,
            "url": self.url,
            "origin": self.origin,
            "is_oopif": self.is_oopif,
        }
        if self.cdp_session_id:
            d["session_id"] = self.cdp_session_id
        if self.parent_frame_id:
            d["parent_frame_id"] = self.parent_frame_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ConsoleEvent:
    """Ring buffer entry for console + exception traffic."""

    ts: float
    level: str  # "log" | "error" | "warning" | "exception"
    text: str
    url: Optional[str] = None


@dataclass(frozen=True)
class SupervisorSnapshot:
    """Read-only snapshot of supervisor state.

    Frozen dataclass so tool handlers can freely dereference without
    worrying about mutation under their feet.
    """

    pending_dialogs: Tuple[PendingDialog, ...]
    recent_dialogs: Tuple[DialogRecord, ...]
    frame_tree: Dict[str, Any]
    console_errors: Tuple[ConsoleEvent, ...]
    active: bool  # False if supervisor is detached/stopped
    cdp_url: str
    task_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for inclusion in ``browser_snapshot`` output."""
        out: Dict[str, Any] = {
            "pending_dialogs": [d.to_dict() for d in self.pending_dialogs],
            "frame_tree": self.frame_tree,
        }
        if self.recent_dialogs:
            out["recent_dialogs"] = [d.to_dict() for d in self.recent_dialogs]
        return out


# ── Supervisor core ───────────────────────────────────────────────────────────


class CDPSupervisor:
    """One supervisor per (task_id, cdp_url) pair.

    Lifecycle:
      * ``start()`` — kicked off by ``SupervisorRegistry.get_or_start``; spawns
        a daemon thread running its own asyncio loop, connects the WebSocket,
        attaches to the first page target, enables domains, starts
        auto-attaching to child targets.
      * ``snapshot()`` — sync, thread-safe, called from tool handlers.
      * ``respond_to_dialog(action, ...)`` — sync bridge; schedules a coroutine
        on the supervisor's loop and waits (with timeout) for the CDP ack.
      * ``stop()`` — cancels task, closes WebSocket, joins thread.

    All CDP I/O lives on the supervisor's own loop. External callers never
    touch the loop directly; they go through the sync API above.
    """

    def __init__(
        self,
        task_id: str,
        cdp_url: str,
        *,
        dialog_policy: str = DEFAULT_DIALOG_POLICY,
        dialog_timeout_s: float = DEFAULT_DIALOG_TIMEOUT_S,
    ) -> None:
        if dialog_policy not in _VALID_POLICIES:
            raise ValueError(
                f"Invalid dialog_policy {dialog_policy!r}; "
                f"must be one of {sorted(_VALID_POLICIES)}"
            )
        self.task_id = task_id
        self.cdp_url = cdp_url
        self.dialog_policy = dialog_policy
        self.dialog_timeout_s = float(dialog_timeout_s)

        # State protected by ``_state_lock`` for cross-thread reads.
        self._state_lock = threading.Lock()
        self._pending_dialogs: Dict[str, PendingDialog] = {}
        self._recent_dialogs: List[DialogRecord] = []
        self._frames: Dict[str, FrameInfo] = {}
        self._console_events: List[ConsoleEvent] = []
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
        self._child_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> info

        # Dialog auto-dismiss watchdog handles (per dialog id).
        self._dialog_watchdogs: Dict[str, asyncio.TimerHandle] = {}
        # Monotonic id generator for dialogs (human-readable in snapshots).
        self._dialog_seq = 0

    # ── Public sync API ──────────────────────────────────────────────────────

    def start(self, timeout: float = 15.0) -> None:
        """Launch the background loop and wait until attachment is complete.

        Raises whatever exception attach failed with (connect error, bad
        WebSocket URL, CDP domain enable failure, etc.). On success, the
        supervisor is fully wired up — pending-dialog events will be captured
        as of the moment ``start()`` returns.
        """
        if self._thread and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._start_error = None
        self._stop_requested = False
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"cdp-supervisor-{self.task_id}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            self.stop()
            raise TimeoutError(
                f"CDP supervisor did not attach within {timeout}s "
                f"(cdp_url={self.cdp_url[:80]}...)"
            )
        if self._start_error is not None:
            err = self._start_error
            self.stop()
            raise err

    def stop(self, timeout: float = 5.0) -> None:
        """Cancel the supervisor task and join the thread."""
        self._stop_requested = True
        loop = self._loop
        if loop is not None and loop.is_running():
            # Close the WebSocket from inside the loop — this makes ``async for
            # raw in self._ws`` return cleanly, ``_run`` hits its ``finally``,
            # pending tasks get cancelled in order, THEN the thread exits.
            async def _close_ws():
                ws = self._ws
                self._ws = None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass

            try:
                fut = asyncio.run_coroutine_threadsafe(_close_ws(), loop)
                try:
                    fut.result(timeout=2.0)
                except Exception:
                    pass
            except RuntimeError:
                pass  # loop already shutting down
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        with self._state_lock:
            self._active = False

    def snapshot(self) -> SupervisorSnapshot:
        """Return an immutable snapshot of current state."""
        with self._state_lock:
            dialogs = tuple(self._pending_dialogs.values())
            recent = tuple(self._recent_dialogs[-RECENT_DIALOGS_MAX:])
            frames_tree = self._build_frame_tree_locked()
            console = tuple(self._console_events[-CONSOLE_HISTORY_MAX:])
            active = self._active
        return SupervisorSnapshot(
            pending_dialogs=dialogs,
            recent_dialogs=recent,
            frame_tree=frames_tree,
            console_errors=console,
            active=active,
            cdp_url=self.cdp_url,
            task_id=self.task_id,
        )

    def respond_to_dialog(
        self,
        action: str,
        *,
        prompt_text: Optional[str] = None,
        dialog_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Accept/dismiss a pending dialog. Sync bridge onto the supervisor loop.

        Returns ``{"ok": True, "dialog": {...}}`` on success,
        ``{"ok": False, "error": "..."}`` on a recoverable error (no dialog,
        ambiguous dialog_id, supervisor inactive).
        """
        if action not in ("accept", "dismiss"):
            return {"ok": False, "error": f"action must be 'accept' or 'dismiss', got {action!r}"}

        with self._state_lock:
            if not self._active:
                return {"ok": False, "error": "supervisor is not active"}
            pending = list(self._pending_dialogs.values())
            if not pending:
                return {"ok": False, "error": "no dialog is currently open"}
            if dialog_id:
                dialog = self._pending_dialogs.get(dialog_id)
                if dialog is None:
                    return {
                        "ok": False,
                        "error": f"dialog_id {dialog_id!r} not found "
                        f"(known: {sorted(self._pending_dialogs)})",
                    }
            elif len(pending) > 1:
                return {
                    "ok": False,
                    "error": (
                        f"{len(pending)} pending dialogs; specify dialog_id. "
                        f"Candidates: {[d.id for d in pending]}"
                    ),
                }
            else:
                dialog = pending[0]
            snapshot_copy = dialog

        loop = self._loop
        if loop is None:
            return {"ok": False, "error": "supervisor loop is not running"}

        async def _do_respond():
            return await self._handle_dialog_cdp(
                snapshot_copy, accept=(action == "accept"), prompt_text=prompt_text or ""
            )

        try:
            fut = asyncio.run_coroutine_threadsafe(_do_respond(), loop)
            fut.result(timeout=timeout)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "dialog": snapshot_copy.to_dict()}

    # ── Supervisor loop internals ────────────────────────────────────────────

    def _thread_main(self) -> None:
        """Entry point for the supervisor's dedicated thread."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run())
        except BaseException as e:  # noqa: BLE001 — propagate via _start_error
            if not self._ready_event.is_set():
                self._start_error = e
                self._ready_event.set()
            else:
                logger.warning("CDP supervisor %s crashed: %s", self.task_id, e)
        finally:
            # Flush any remaining tasks before closing the loop so we don't
            # emit "Task was destroyed but it is pending" warnings.
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            with self._state_lock:
                self._active = False

    async def _run(self) -> None:
        """Top-level supervisor coroutine.

        Holds a reconnecting loop so we survive the remote closing the
        WebSocket — Browserbase in particular tears down the CDP socket
        every time a short-lived client (e.g. agent-browser's per-command
        CDP client) disconnects.  We drop our state snapshot keys that
        depend on specific CDP session ids, re-attach, and keep going.
        """
        attempt = 0
        last_success_at = 0.0
        backoff = 0.5
        while not self._stop_requested:
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(self.cdp_url, max_size=50 * 1024 * 1024),
                    timeout=10.0,
                )
            except Exception as e:
                attempt += 1
                if not self._ready_event.is_set():
                    # Never connected once — fatal for start().
                    self._start_error = e
                    self._ready_event.set()
                    return
                logger.warning(
                    "CDP supervisor %s: connect failed (attempt %s): %s",
                    self.task_id, attempt, e,
                )
                await asyncio.sleep(min(backoff, 10.0))
                backoff = min(backoff * 2, 10.0)
                continue

            reader_task = asyncio.create_task(self._read_loop(), name="cdp-reader")
            try:
                # Reset per-connection session state so stale ids don't hang
                # around after a reconnect.
                self._page_session_id = None
                self._child_sessions.clear()
                # We deliberately keep `_pending_dialogs` and `_frames` —
                # they're reconciled as the supervisor resubscribes and
                # receives fresh events.  Worst case: an agent sees a stale
                # dialog entry that the new session's handleJavaScriptDialog
                # call rejects with "no dialog is showing" (logged, not
                # surfaced).
                await self._attach_initial_page()
                with self._state_lock:
                    self._active = True
                last_success_at = time.time()
                backoff = 0.5  # reset after a successful attach
                if not self._ready_event.is_set():
                    self._ready_event.set()
                # Run until the reader returns.
                await reader_task
            except BaseException as e:
                if not self._ready_event.is_set():
                    # Never got to ready — propagate to start().
                    self._start_error = e
                    self._ready_event.set()
                    raise
                logger.warning(
                    "CDP supervisor %s: session dropped after %.1fs: %s",
                    self.task_id,
                    time.time() - last_success_at,
                    e,
                )
            finally:
                with self._state_lock:
                    self._active = False
                if not reader_task.done():
                    reader_task.cancel()
                    try:
                        await reader_task
                    except (asyncio.CancelledError, Exception):
                        pass
                for handle in list(self._dialog_watchdogs.values()):
                    handle.cancel()
                self._dialog_watchdogs.clear()
                ws = self._ws
                self._ws = None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass

            if self._stop_requested:
                return

            # Reconnect: brief backoff, then reattach.
            logger.debug(
                "CDP supervisor %s: reconnecting in %.1fs...", self.task_id, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _attach_initial_page(self) -> None:
        """Find a page target, attach flattened session, enable domains, install dialog bridge."""
        resp = await self._cdp("Target.getTargets")
        targets = resp.get("result", {}).get("targetInfos", [])
        page_target = next((t for t in targets if t.get("type") == "page"), None)
        if page_target is None:
            created = await self._cdp("Target.createTarget", {"url": "about:blank"})
            target_id = created["result"]["targetId"]
        else:
            target_id = page_target["targetId"]

        attach = await self._cdp(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
        )
        self._page_session_id = attach["result"]["sessionId"]
        await self._cdp("Page.enable", session_id=self._page_session_id)
        await self._cdp("Runtime.enable", session_id=self._page_session_id)
        await self._cdp(
            "Target.setAutoAttach",
            {"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True},
            session_id=self._page_session_id,
        )
        # Install the dialog bridge — overrides native alert/confirm/prompt with
        # a synchronous XHR we intercept via Fetch domain. This is how we make
        # dialog response work on Browserbase (whose CDP proxy auto-dismisses
        # real native dialogs before we can call handleJavaScriptDialog).
        await self._install_dialog_bridge(self._page_session_id)

    async def _install_dialog_bridge(self, session_id: str) -> None:
        """Install the dialog-bridge init script + Fetch interceptor on a session.

        Two CDP calls:
          1. ``Page.addScriptToEvaluateOnNewDocument`` — the JS override runs
             in every frame before any page script. Replaces alert/confirm/
             prompt with a sync XHR to our bridge URL.
          2. ``Fetch.enable`` scoped to the bridge URL — we catch those XHRs,
             surface them as pending dialogs, then fulfill once the agent
             responds.

        Idempotent at the CDP level: Chromium de-duplicates identical
        add-script calls by source, and Fetch.enable replaces prior patterns.
        """
        try:
            await self._cdp(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": _DIALOG_BRIDGE_SCRIPT, "runImmediately": True},
                session_id=session_id,
                timeout=5.0,
            )
        except Exception as e:
            logger.debug(
                "dialog bridge: addScriptToEvaluateOnNewDocument failed on sid=%s: %s",
                (session_id or "")[:16], e,
            )
        try:
            await self._cdp(
                "Fetch.enable",
                {
                    "patterns": [
                        {
                            "urlPattern": DIALOG_BRIDGE_URL_PATTERN,
                            "requestStage": "Request",
                        }
                    ],
                    "handleAuthRequests": False,
                },
                session_id=session_id,
                timeout=5.0,
            )
        except Exception as e:
            logger.debug(
                "dialog bridge: Fetch.enable failed on sid=%s: %s",
                (session_id or "")[:16], e,
            )
        # Also try to inject into the already-loaded document so existing
        # pages pick up the override on reconnect. Best-effort.
        try:
            await self._cdp(
                "Runtime.evaluate",
                {"expression": _DIALOG_BRIDGE_SCRIPT, "returnByValue": True},
                session_id=session_id,
                timeout=3.0,
            )
        except Exception:
            pass

    async def _cdp(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Send a CDP command and await its response."""
        if self._ws is None:
            raise RuntimeError("supervisor WebSocket is not connected")
        call_id = self._next_call_id
        self._next_call_id += 1
        payload: Dict[str, Any] = {"id": call_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_calls[call_id] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_calls.pop(call_id, None)

    async def _read_loop(self) -> None:
        """Continuously dispatch incoming CDP frames."""
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
                    if fut is not None and not fut.done():
                        if "error" in msg:
                            fut.set_exception(
                                RuntimeError(f"CDP error on id={msg['id']}: {msg['error']}")
                            )
                        else:
                            fut.set_result(msg)
                elif "method" in msg:
                    await self._on_event(msg["method"], msg.get("params", {}), msg.get("sessionId"))
        except Exception as e:
            logger.debug("CDP read loop exited: %s", e)

    # ── Event dispatch ──────────────────────────────────────────────────────

    async def _on_event(
        self, method: str, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        if method == "Page.javascriptDialogOpening":
            await self._on_dialog_opening(params, session_id)
        elif method == "Page.javascriptDialogClosed":
            await self._on_dialog_closed(params, session_id)
        elif method == "Fetch.requestPaused":
            await self._on_fetch_paused(params, session_id)
        elif method == "Page.frameAttached":
            self._on_frame_attached(params, session_id)
        elif method == "Page.frameNavigated":
            self._on_frame_navigated(params, session_id)
        elif method == "Page.frameDetached":
            self._on_frame_detached(params, session_id)
        elif method == "Target.attachedToTarget":
            await self._on_target_attached(params)
        elif method == "Target.detachedFromTarget":
            self._on_target_detached(params)
        elif method == "Runtime.consoleAPICalled":
            self._on_console(params, level_from="api")
        elif method == "Runtime.exceptionThrown":
            self._on_console(params, level_from="exception")

    async def _on_dialog_opening(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        self._dialog_seq += 1
        dialog = PendingDialog(
            id=f"d-{self._dialog_seq}",
            type=str(params.get("type") or ""),
            message=str(params.get("message") or ""),
            default_prompt=str(params.get("defaultPrompt") or ""),
            opened_at=time.time(),
            cdp_session_id=session_id or self._page_session_id or "",
            frame_id=params.get("frameId"),
        )

        if self.dialog_policy == DIALOG_POLICY_AUTO_DISMISS:
            # Archive immediately with the policy tag so the ``closed`` event
            # arriving right after our handleJavaScriptDialog call doesn't
            # re-archive it as "remote".
            with self._state_lock:
                self._archive_dialog_locked(dialog, "auto_policy")
            asyncio.create_task(
                self._auto_handle_dialog(dialog, accept=False, prompt_text="")
            )
        elif self.dialog_policy == DIALOG_POLICY_AUTO_ACCEPT:
            with self._state_lock:
                self._archive_dialog_locked(dialog, "auto_policy")
            asyncio.create_task(
                self._auto_handle_dialog(
                    dialog, accept=True, prompt_text=dialog.default_prompt
                )
            )
        else:
            # must_respond → add to pending and arm watchdog.
            with self._state_lock:
                self._pending_dialogs[dialog.id] = dialog
            loop = asyncio.get_running_loop()
            handle = loop.call_later(
                self.dialog_timeout_s,
                lambda: asyncio.create_task(self._dialog_timeout_expired(dialog.id)),
            )
            self._dialog_watchdogs[dialog.id] = handle

    async def _auto_handle_dialog(
        self, dialog: PendingDialog, *, accept: bool, prompt_text: str
    ) -> None:
        """Send handleJavaScriptDialog for auto_dismiss/auto_accept.

        Dialog has already been archived by the caller (``_on_dialog_opening``);
        this just fires the CDP call so the page unblocks.
        """
        params: Dict[str, Any] = {"accept": accept}
        if dialog.type == "prompt":
            params["promptText"] = prompt_text
        try:
            await self._cdp(
                "Page.handleJavaScriptDialog",
                params,
                session_id=dialog.cdp_session_id or None,
                timeout=5.0,
            )
        except Exception as e:
            logger.debug("auto-handle CDP call failed for %s: %s", dialog.id, e)

    async def _dialog_timeout_expired(self, dialog_id: str) -> None:
        with self._state_lock:
            dialog = self._pending_dialogs.get(dialog_id)
        if dialog is None:
            return
        logger.warning(
            "CDP supervisor %s: dialog %s (%s) auto-dismissed after %ss timeout",
            self.task_id,
            dialog_id,
            dialog.type,
            self.dialog_timeout_s,
        )
        try:
            # Archive with watchdog tag BEFORE fulfilling / dismissing.
            with self._state_lock:
                if dialog_id in self._pending_dialogs:
                    self._pending_dialogs.pop(dialog_id, None)
                    self._archive_dialog_locked(dialog, "watchdog")
            # Unblock the page — via bridge Fetch fulfill for bridge dialogs,
            # else native Page.handleJavaScriptDialog for real dialogs.
            if dialog.bridge_request_id:
                await self._fulfill_bridge_request(dialog, accept=False, prompt_text="")
            else:
                await self._cdp(
                    "Page.handleJavaScriptDialog",
                    {"accept": False},
                    session_id=dialog.cdp_session_id or None,
                    timeout=5.0,
                )
        except Exception as e:
            logger.debug("auto-dismiss failed for %s: %s", dialog_id, e)

    def _archive_dialog_locked(self, dialog: PendingDialog, closed_by: str) -> None:
        """Move a pending dialog to the recent_dialogs ring buffer. Must hold state_lock."""
        record = DialogRecord(
            id=dialog.id,
            type=dialog.type,
            message=dialog.message,
            opened_at=dialog.opened_at,
            closed_at=time.time(),
            closed_by=closed_by,
            frame_id=dialog.frame_id,
        )
        self._recent_dialogs.append(record)
        if len(self._recent_dialogs) > RECENT_DIALOGS_MAX * 2:
            self._recent_dialogs = self._recent_dialogs[-RECENT_DIALOGS_MAX:]

    async def _handle_dialog_cdp(
        self, dialog: PendingDialog, *, accept: bool, prompt_text: str
    ) -> None:
        """Send the Page.handleJavaScriptDialog CDP command (agent path only).

        Routes to the bridge-fulfill path when the dialog was captured via
        the injected XHR override (see ``_on_fetch_paused``).
        """
        if dialog.bridge_request_id:
            try:
                await self._fulfill_bridge_request(
                    dialog, accept=accept, prompt_text=prompt_text
                )
            finally:
                with self._state_lock:
                    if dialog.id in self._pending_dialogs:
                        self._pending_dialogs.pop(dialog.id, None)
                        self._archive_dialog_locked(dialog, "agent")
                handle = self._dialog_watchdogs.pop(dialog.id, None)
                if handle is not None:
                    handle.cancel()
            return

        params: Dict[str, Any] = {"accept": accept}
        if dialog.type == "prompt":
            params["promptText"] = prompt_text
        try:
            await self._cdp(
                "Page.handleJavaScriptDialog",
                params,
                session_id=dialog.cdp_session_id or None,
                timeout=5.0,
            )
        finally:
            # Clear regardless — the CDP error path usually means the dialog
            # already closed (browser auto-dismissed after navigation, etc.).
            with self._state_lock:
                if dialog.id in self._pending_dialogs:
                    self._pending_dialogs.pop(dialog.id, None)
                    self._archive_dialog_locked(dialog, "agent")
            handle = self._dialog_watchdogs.pop(dialog.id, None)
            if handle is not None:
                handle.cancel()

    async def _on_dialog_closed(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        # ``Page.javascriptDialogClosed`` spec has only ``result`` (bool) and
        # ``userInput`` (string), not the original ``message``.  Match by
        # session id and clear the oldest dialog on that session — if Chrome
        # closed one on us (e.g. our disconnect auto-dismissed it, or the
        # browser navigated, or Browserbase's CDP proxy auto-dismissed), there
        # shouldn't be more than one in flight per session anyway because the
        # JS thread is blocked while a dialog is up.
        with self._state_lock:
            candidate_ids = [
                d.id
                for d in self._pending_dialogs.values()
                if d.cdp_session_id == session_id
                # Bridge-captured dialogs aren't cleared by native close events;
                # they're resolved via Fetch.fulfillRequest instead. Only the
                # real-native-dialog path uses Page.javascriptDialogClosed.
                and d.bridge_request_id is None
            ]
            if candidate_ids:
                did = candidate_ids[0]
                dialog = self._pending_dialogs.pop(did, None)
                if dialog is not None:
                    self._archive_dialog_locked(dialog, "remote")
                handle = self._dialog_watchdogs.pop(did, None)
                if handle is not None:
                    handle.cancel()

    async def _on_fetch_paused(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        """Bridge XHR captured mid-flight — materialize as a pending dialog.

        The injected script (``_DIALOG_BRIDGE_SCRIPT``) fires a synchronous
        XHR to ``DIALOG_BRIDGE_HOST`` whenever page code calls alert/confirm/
        prompt. We catch it via Fetch.enable pattern; the page's JS thread
        is blocked on the XHR's response until we call Fetch.fulfillRequest
        (which happens from ``respond_to_dialog``) or until the watchdog
        fires (at which point we fulfill with a cancel response).
        """
        url = str(params.get("request", {}).get("url") or "")
        request_id = params.get("requestId")
        if not request_id:
            return
        # Only care about our bridge URLs. Fetch can still deliver other
        # intercepted requests if patterns were ever broadened.
        if DIALOG_BRIDGE_HOST not in url:
            # Not ours — forward unchanged so the page sees its own request.
            try:
                await self._cdp(
                    "Fetch.continueRequest", {"requestId": request_id},
                    session_id=session_id, timeout=3.0,
                )
            except Exception:
                pass
            return

        # Parse query string for dialog metadata. Use urllib to be robust.
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(url).query)

        def _q(name: str) -> str:
            v = q.get(name, [""])
            return v[0] if v else ""

        kind = _q("kind") or "alert"
        message = _q("message")
        default_prompt = _q("default_prompt")

        self._dialog_seq += 1
        dialog = PendingDialog(
            id=f"d-{self._dialog_seq}",
            type=kind,
            message=message,
            default_prompt=default_prompt,
            opened_at=time.time(),
            cdp_session_id=session_id or self._page_session_id or "",
            frame_id=params.get("frameId"),
            bridge_request_id=str(request_id),
        )

        # Apply policy exactly as for native dialogs.
        if self.dialog_policy == DIALOG_POLICY_AUTO_DISMISS:
            with self._state_lock:
                self._archive_dialog_locked(dialog, "auto_policy")
            asyncio.create_task(
                self._fulfill_bridge_request(dialog, accept=False, prompt_text="")
            )
        elif self.dialog_policy == DIALOG_POLICY_AUTO_ACCEPT:
            with self._state_lock:
                self._archive_dialog_locked(dialog, "auto_policy")
            asyncio.create_task(
                self._fulfill_bridge_request(
                    dialog, accept=True, prompt_text=default_prompt
                )
            )
        else:
            # must_respond — add to pending + arm watchdog.
            with self._state_lock:
                self._pending_dialogs[dialog.id] = dialog
            loop = asyncio.get_running_loop()
            handle = loop.call_later(
                self.dialog_timeout_s,
                lambda: asyncio.create_task(self._dialog_timeout_expired(dialog.id)),
            )
            self._dialog_watchdogs[dialog.id] = handle

    async def _fulfill_bridge_request(
        self, dialog: PendingDialog, *, accept: bool, prompt_text: str
    ) -> None:
        """Resolve a bridge XHR via Fetch.fulfillRequest so the page unblocks."""
        if not dialog.bridge_request_id:
            return
        payload = {
            "accept": bool(accept),
            "prompt_text": prompt_text if dialog.type == "prompt" else "",
            "dialog_id": dialog.id,
        }
        body = json.dumps(payload).encode()
        try:
            import base64 as _b64
            await self._cdp(
                "Fetch.fulfillRequest",
                {
                    "requestId": dialog.bridge_request_id,
                    "responseCode": 200,
                    "responseHeaders": [
                        {"name": "Content-Type", "value": "application/json"},
                        {"name": "Access-Control-Allow-Origin", "value": "*"},
                    ],
                    "body": _b64.b64encode(body).decode(),
                },
                session_id=dialog.cdp_session_id or None,
                timeout=5.0,
            )
        except Exception as e:
            logger.debug("bridge fulfill failed for %s: %s", dialog.id, e)

    # ── Frame / target tracking ─────────────────────────────────────────────

    def _on_frame_attached(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        frame_id = params.get("frameId")
        if not frame_id:
            return
        with self._state_lock:
            self._frames[frame_id] = FrameInfo(
                frame_id=frame_id,
                url="",
                origin="",
                parent_frame_id=params.get("parentFrameId"),
                is_oopif=False,
                cdp_session_id=session_id,
            )

    def _on_frame_navigated(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        frame = params.get("frame") or {}
        frame_id = frame.get("id")
        if not frame_id:
            return
        with self._state_lock:
            existing = self._frames.get(frame_id)
            info = FrameInfo(
                frame_id=frame_id,
                url=str(frame.get("url") or ""),
                origin=str(frame.get("securityOrigin") or frame.get("origin") or ""),
                parent_frame_id=frame.get("parentId") or (existing.parent_frame_id if existing else None),
                is_oopif=bool(existing.is_oopif if existing else False),
                cdp_session_id=existing.cdp_session_id if existing else session_id,
                name=str(frame.get("name") or (existing.name if existing else "")),
            )
            self._frames[frame_id] = info

    def _on_frame_detached(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        """Remove a frame from our state only when it's truly gone.

        CDP emits ``Page.frameDetached`` with a ``reason`` of either
        ``"remove"`` (the frame is actually gone from the DOM) or ``"swap"``
        (the frame is migrating to a new process — typical when a
        same-process iframe becomes an OOPIF, or when history navigates).
        Dropping on ``swap`` would hide OOPIFs from the agent the moment
        Chromium promotes them to their own process, so treat swap as a
        no-op.

        Even with ``reason=remove``, the parent page's perspective is
        "the child frame left MY process tree" — which is what happens
        when a same-origin iframe gets promoted to an OOPIF. If we
        already have a live child CDP session attached for that frame_id,
        the frame is still very much alive; only drop it when we have
        no session record.
        """
        frame_id = params.get("frameId")
        if not frame_id:
            return
        reason = str(params.get("reason") or "remove").lower()
        if reason == "swap":
            return
        with self._state_lock:
            existing = self._frames.get(frame_id)
            # Keep OOPIF records even when the parent says the frame was
            # "removed" — the iframe is still visible, just in a different
            # process. If the frame truly goes away later, Target.detached
            # + the next Page.frameDetached without a live session will
            # clear it.
            if existing and existing.is_oopif and existing.cdp_session_id:
                return
            self._frames.pop(frame_id, None)

    async def _on_target_attached(self, params: Dict[str, Any]) -> None:
        info = params.get("targetInfo") or {}
        sid = params.get("sessionId")
        target_type = info.get("type")
        if not sid or target_type not in ("iframe", "worker"):
            return
        self._child_sessions[sid] = {"info": info, "type": target_type}

        # Record the frame with its OOPIF session id for interaction routing.
        if target_type == "iframe":
            target_id = info.get("targetId")
            with self._state_lock:
                existing = self._frames.get(target_id)
                self._frames[target_id] = FrameInfo(
                    frame_id=target_id,
                    url=str(info.get("url") or ""),
                    origin="",  # filled by frameNavigated on the child session
                    parent_frame_id=(existing.parent_frame_id if existing else None),
                    is_oopif=True,
                    cdp_session_id=sid,
                    name=str(info.get("title") or (existing.name if existing else "")),
                )

        # Enable domains on the child off-loop so the reader keeps pumping.
        # Awaiting the CDP replies here would deadlock because only the
        # reader can resolve those replies' Futures.
        asyncio.create_task(self._enable_child_domains(sid))

    async def _enable_child_domains(self, sid: str) -> None:
        """Enable Page+Runtime (+nested setAutoAttach) on a child CDP session.

        Also installs the dialog bridge so iframe-scoped alert/confirm/prompt
        calls round-trip through Fetch too.
        """
        try:
            await self._cdp("Page.enable", session_id=sid, timeout=3.0)
            await self._cdp("Runtime.enable", session_id=sid, timeout=3.0)
            await self._cdp(
                "Target.setAutoAttach",
                {"autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True},
                session_id=sid,
                timeout=3.0,
            )
        except Exception as e:
            logger.debug("child session %s setup failed: %s", sid[:16], e)
        # Install the dialog bridge on the child so iframe dialogs are captured.
        await self._install_dialog_bridge(sid)

    def _on_target_detached(self, params: Dict[str, Any]) -> None:
        """Handle a child CDP session detaching.

        We deliberately DO NOT drop frames from ``_frames`` here — Browserbase
        fires transient detach events during page transitions even while the
        iframe is still visible to the user, and dropping the record hides
        OOPIFs from the agent between the detach and the next
        ``Target.attachedToTarget``. Instead, we just clear the session
        binding so stale ``cdp_session_id`` values aren't used for routing.
        If the iframe truly goes away, ``Page.frameDetached`` will clean up.
        """
        sid = params.get("sessionId")
        if not sid:
            return
        self._child_sessions.pop(sid, None)
        with self._state_lock:
            for fid, frame in list(self._frames.items()):
                if frame.cdp_session_id == sid:
                    # Replace with a copy that has cdp_session_id cleared so
                    # routing falls back to top-level page session if retried.
                    self._frames[fid] = FrameInfo(
                        frame_id=frame.frame_id,
                        url=frame.url,
                        origin=frame.origin,
                        parent_frame_id=frame.parent_frame_id,
                        is_oopif=frame.is_oopif,
                        cdp_session_id=None,
                        name=frame.name,
                    )

    # ── Console / exception ring buffer ─────────────────────────────────────

    def _on_console(self, params: Dict[str, Any], *, level_from: str) -> None:
        if level_from == "exception":
            details = params.get("exceptionDetails") or {}
            text = str(details.get("text") or "")
            url = details.get("url")
            event = ConsoleEvent(ts=time.time(), level="exception", text=text, url=url)
        else:
            raw_level = str(params.get("type") or "log")
            level = "error" if raw_level in ("error", "assert") else (
                "warning" if raw_level == "warning" else "log"
            )
            args = params.get("args") or []
            parts: List[str] = []
            for a in args[:4]:
                if isinstance(a, dict):
                    parts.append(str(a.get("value") or a.get("description") or ""))
            event = ConsoleEvent(ts=time.time(), level=level, text=" ".join(parts))
        with self._state_lock:
            self._console_events.append(event)
            if len(self._console_events) > CONSOLE_HISTORY_MAX * 2:
                # Keep last CONSOLE_HISTORY_MAX; allow 2x slack to reduce churn.
                self._console_events = self._console_events[-CONSOLE_HISTORY_MAX:]

    # ── Frame tree building (bounded) ───────────────────────────────────────

    def _build_frame_tree_locked(self) -> Dict[str, Any]:
        """Build the capped frame_tree payload. Must be called under state lock."""
        frames = self._frames
        if not frames:
            return {"top": None, "children": [], "truncated": False}

        # Identify a top frame — one with no parent, preferring oopif=False.
        tops = [f for f in frames.values() if not f.parent_frame_id]
        top = next((f for f in tops if not f.is_oopif), tops[0] if tops else None)

        # BFS from top, capped by FRAME_TREE_MAX_ENTRIES and
        # FRAME_TREE_MAX_OOPIF_DEPTH for OOPIF branches.
        children: List[Dict[str, Any]] = []
        truncated = False
        if top is None:
            return {"top": None, "children": [], "truncated": False}

        queue: List[Tuple[FrameInfo, int]] = [
            (f, 1) for f in frames.values() if f.parent_frame_id == top.frame_id
        ]
        visited: set[str] = {top.frame_id}
        while queue and len(children) < FRAME_TREE_MAX_ENTRIES:
            frame, depth = queue.pop(0)
            if frame.frame_id in visited:
                continue
            visited.add(frame.frame_id)
            if frame.is_oopif and depth > FRAME_TREE_MAX_OOPIF_DEPTH:
                truncated = True
                continue
            children.append(frame.to_dict())
            for f in frames.values():
                if f.parent_frame_id == frame.frame_id and f.frame_id not in visited:
                    queue.append((f, depth + 1))
        if queue:
            truncated = True

        return {
            "top": top.to_dict(),
            "children": children,
            "truncated": truncated,
        }


# ── Registry ─────────────────────────────────────────────────────────────────


class _SupervisorRegistry:
    """Process-global (task_id → supervisor) map with idempotent start/stop.

    One instance, exposed as ``SUPERVISOR_REGISTRY``. Safe to call from any
    thread — mutations go through ``_lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_task: Dict[str, CDPSupervisor] = {}

    def get(self, task_id: str) -> Optional[CDPSupervisor]:
        """Return the supervisor for ``task_id`` if running, else ``None``."""
        with self._lock:
            return self._by_task.get(task_id)

    def get_or_start(
        self,
        task_id: str,
        cdp_url: str,
        *,
        dialog_policy: str = DEFAULT_DIALOG_POLICY,
        dialog_timeout_s: float = DEFAULT_DIALOG_TIMEOUT_S,
        start_timeout: float = 15.0,
    ) -> CDPSupervisor:
        """Idempotently ensure a supervisor is running for ``(task_id, cdp_url)``.

        If a supervisor exists for this task but was bound to a different
        ``cdp_url``, the old one is stopped and a fresh one is started.
        """
        with self._lock:
            existing = self._by_task.get(task_id)
            if existing is not None:
                if existing.cdp_url == cdp_url:
                    return existing
                # URL changed — tear down old, fall through to re-create.
                self._by_task.pop(task_id, None)
        if existing is not None:
            existing.stop()

        supervisor = CDPSupervisor(
            task_id=task_id,
            cdp_url=cdp_url,
            dialog_policy=dialog_policy,
            dialog_timeout_s=dialog_timeout_s,
        )
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
        """Stop and discard the supervisor for ``task_id`` if it exists."""
        with self._lock:
            supervisor = self._by_task.pop(task_id, None)
        if supervisor is not None:
            supervisor.stop()

    def stop_all(self) -> None:
        """Stop every running supervisor. For shutdown / test teardown."""
        with self._lock:
            items = list(self._by_task.items())
            self._by_task.clear()
        for _, supervisor in items:
            supervisor.stop()


SUPERVISOR_REGISTRY = _SupervisorRegistry()


__all__ = [
    "CDPSupervisor",
    "ConsoleEvent",
    "DEFAULT_DIALOG_POLICY",
    "DEFAULT_DIALOG_TIMEOUT_S",
    "DIALOG_POLICY_AUTO_ACCEPT",
    "DIALOG_POLICY_AUTO_DISMISS",
    "DIALOG_POLICY_MUST_RESPOND",
    "DialogRecord",
    "FrameInfo",
    "PendingDialog",
    "SUPERVISOR_REGISTRY",
    "SupervisorSnapshot",
    "_SupervisorRegistry",
]
