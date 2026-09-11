"""Dialog capture + response half of the CDP supervisor.

Two capture paths feed one ``PendingDialog`` queue: native ``Page.javascriptDialogOpening``
events (answered with ``Page.handleJavaScriptDialog``), and the injected *dialog bridge* —
a page script rewriting alert/confirm/prompt into a sync XHR to a magic host we intercept
via the CDP ``Fetch`` domain and answer with ``Fetch.fulfillRequest``. The bridge works on
Browserbase (whose CDP proxy auto-dismisses native dialogs) because the native dialog never fires.
``DialogSupervisionMixin`` relies on state ``CDPSupervisor.__init__`` sets.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

# Logger-name parity with the origin module (records must look unchanged).
logger = logging.getLogger("tools.browser_supervisor")


def _redact_supervisor_text(value: str) -> str:
    """Redact page-originated text before exposing supervisor snapshots."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(value, force=True)


def _trim_ring(events: list, keep: int) -> list:
    """Cap a ring buffer at ``keep`` entries once it overflows to 2x (slack reduces churn)."""
    return events[-keep:] if len(events) > keep * 2 else events


_REDACTED_FIELDS = frozenset({"message", "default_prompt"})


def _dialog_dict(obj: Any, keys: tuple) -> Dict[str, Any]:
    """Snapshot dict of ``keys`` with page-originated text fields redacted."""
    return {k: _redact_supervisor_text(getattr(obj, k)) if k in _REDACTED_FIELDS else getattr(obj, k) for k in keys}


DIALOG_POLICY_MUST_RESPOND = "must_respond"
DIALOG_POLICY_AUTO_DISMISS = "auto_dismiss"
DIALOG_POLICY_AUTO_ACCEPT = "auto_accept"
_VALID_POLICIES = frozenset(
    {DIALOG_POLICY_MUST_RESPOND, DIALOG_POLICY_AUTO_DISMISS, DIALOG_POLICY_AUTO_ACCEPT}
)
DEFAULT_DIALOG_POLICY = DIALOG_POLICY_MUST_RESPOND
DEFAULT_DIALOG_TIMEOUT_S = 300.0

# Last N closed dialogs kept so agents on backends that auto-dismiss server-side
# (Browserbase) can still observe that a dialog fired.
RECENT_DIALOGS_MAX = 20

# Magic host the bridge XHRs to; intercepted via CDP Fetch before any network
# resolution, so it never has to exist. Keep ASCII + URL-safe (Fetch patterns gate on it).
DIALOG_BRIDGE_HOST = "hermes-dialog-bridge.invalid"
DIALOG_BRIDGE_URL_PATTERN = f"http://{DIALOG_BRIDGE_HOST}/*"

# Injected into every frame via Page.addScriptToEvaluateOnNewDocument. Sync GET with
# query params so the Fetch interceptor never parses a body; unreachable bridge → null
# so the page still sees *some* behavior. onbeforeunload is left native (can't be
# prompted synchronously without racing navigation); the native path still records it.
_DIALOG_BRIDGE_SCRIPT = r"""
(() => {
  if (window.__hermesDialogBridgeInstalled) return;
  window.__hermesDialogBridgeInstalled = true;
  const ENDPOINT = "http://hermes-dialog-bridge.invalid/";
  function ask(kind, message, defaultPrompt) {
    try {
      const xhr = new XMLHttpRequest();
      const params = new URLSearchParams({
        kind: String(kind || ""),
        message: String(message == null ? "" : message),
        default_prompt: String(defaultPrompt == null ? "" : defaultPrompt),
      });
      xhr.open("GET", ENDPOINT + "?" + params.toString(), false);  // sync
      xhr.send(null);
      if (xhr.status !== 200) return null;
      let parsed;
      try { parsed = JSON.parse(xhr.responseText || ""); } catch (e) { return null; }
      if (kind === "alert") return undefined;
      if (kind === "confirm") return Boolean(parsed && parsed.accept);
      if (kind === "prompt") {
        if (!parsed || !parsed.accept) return null;
        return parsed.prompt_text == null ? "" : String(parsed.prompt_text);
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  window.alert   = function(message) { ask("alert",   message, ""); };
  window.confirm = function(message) {
    const r = ask("confirm", message, "");
    return r === null ? false : Boolean(r);
  };
  window.prompt  = function(message, def) {
    const r = ask("prompt", message, def == null ? "" : def);
    return r === null ? null : String(r);
  };
})();
"""


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
    # Bridge XHR path: respond via Fetch.fulfillRequest, NOT Page.handleJavaScriptDialog.
    bridge_request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _dialog_dict(self, ("id", "type", "message", "default_prompt", "opened_at", "frame_id"))


@dataclass
class DialogRecord:
    """A dialog that was opened and then handled (kept briefly in ``recent_dialogs``)."""

    id: str
    type: str
    message: str
    opened_at: float
    closed_at: float
    closed_by: str  # "agent" | "auto_policy" | "remote" | "watchdog"
    frame_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _dialog_dict(self, ("id", "type", "message", "opened_at", "closed_at", "closed_by", "frame_id"))


class DialogSupervisionMixin:
    """Dialog event handling for ``CDPSupervisor`` (all methods run on its loop)."""

    async def _cdp_quiet(self, method: str, params: Dict[str, Any], *, session_id: Optional[str],
                         timeout: float, what: str) -> None:
        """Best-effort CDP call: failures are logged at debug and swallowed."""
        try:
            await self._cdp(method, params, session_id=session_id, timeout=timeout)
        except Exception as e:
            logger.debug("%s failed (%s): %s", method, what, e)

    async def _install_dialog_bridge(self, session_id: str) -> None:
        """Install the dialog-bridge init script + Fetch interceptor on a session. Idempotent at
        the CDP level (Chromium de-dupes identical add-script calls; Fetch.enable replaces prior
        patterns); the final Runtime.evaluate injects into the already-loaded document so
        existing pages pick up the override on reconnect."""
        sid = (session_id or "")[:16]
        steps = (
            ("Page.addScriptToEvaluateOnNewDocument", {"source": _DIALOG_BRIDGE_SCRIPT, "runImmediately": True},
             5.0, f"dialog bridge sid={sid}"),
            ("Fetch.enable", {"patterns": [{"urlPattern": DIALOG_BRIDGE_URL_PATTERN, "requestStage": "Request"}],
                              "handleAuthRequests": False}, 5.0, f"dialog bridge sid={sid}"),
            ("Runtime.evaluate", {"expression": _DIALOG_BRIDGE_SCRIPT, "returnByValue": True},
             3.0, f"dialog bridge inject sid={sid}"),
        )
        for method, params, timeout, what in steps:
            await self._cdp_quiet(method, params, session_id=session_id, timeout=timeout, what=what)

    # ── Capture ──────────────────────────────────────────────────────────────

    async def _on_dialog_opening(self, params: Dict[str, Any], session_id: Optional[str]) -> None:
        self._admit_dialog(
            type=str(params.get("type") or ""), message=str(params.get("message") or ""),
            default_prompt=str(params.get("defaultPrompt") or ""), session_id=session_id, frame_id=params.get("frameId"),
        )

    async def _on_fetch_paused(self, params: Dict[str, Any], session_id: Optional[str]) -> None:
        """Bridge XHR captured mid-flight — materialize as a pending dialog. The page's JS
        thread is blocked on the XHR until we Fetch.fulfillRequest (agent or watchdog);
        requests for other hosts are forwarded unchanged."""
        url = str(params.get("request", {}).get("url") or "")
        request_id = params.get("requestId")
        if not request_id:
            return
        if DIALOG_BRIDGE_HOST not in url:
            await self._cdp_quiet("Fetch.continueRequest", {"requestId": request_id},
                                  session_id=session_id, timeout=3.0, what="passthrough")
            return
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        self._admit_dialog(
            type=q.get("kind") or "alert", message=q.get("message", ""), default_prompt=q.get("default_prompt", ""),
            session_id=session_id, frame_id=params.get("frameId"), bridge_request_id=str(request_id),
        )

    def _admit_dialog(self, *, type: str, message: str, default_prompt: str, session_id: Optional[str],
                      frame_id: Optional[str], bridge_request_id: Optional[str] = None) -> None:
        """Create the dialog and apply the policy: auto-respond, or queue + arm the watchdog.
        Auto policies archive FIRST (tagged ``auto_policy``) so the ``closed`` event that
        follows our own response isn't re-archived as ``remote``."""
        self._dialog_seq += 1
        dialog = PendingDialog(
            id=f"d-{self._dialog_seq}", type=type, message=message, default_prompt=default_prompt,
            opened_at=time.time(), cdp_session_id=session_id or self._page_session_id or "",
            frame_id=frame_id, bridge_request_id=bridge_request_id,
        )
        auto = {DIALOG_POLICY_AUTO_DISMISS: (False, ""), DIALOG_POLICY_AUTO_ACCEPT: (True, default_prompt)}.get(
            self.dialog_policy
        )
        if auto is not None:
            with self._state_lock:
                self._archive_dialog_locked(dialog, "auto_policy")
            asyncio.create_task(self._respond_quiet(dialog, accept=auto[0], prompt_text=auto[1]))
            return
        with self._state_lock:
            self._pending_dialogs[dialog.id] = dialog
        self._dialog_watchdogs[dialog.id] = asyncio.get_running_loop().call_later(
            self.dialog_timeout_s,
            lambda: asyncio.create_task(self._dialog_timeout_expired(dialog.id)),
        )

    # ── Responding ───────────────────────────────────────────────────────────

    async def _respond(self, dialog: PendingDialog, *, accept: bool, prompt_text: Optional[str]) -> None:
        """Bridge-fulfill for XHR-captured dialogs (swallows failures so the page
        unblocks), else native CDP — ``promptText`` only for prompt dialogs when
        given; raises on CDP failure."""
        session_id = dialog.cdp_session_id or None
        if dialog.bridge_request_id:
            body = json.dumps({"accept": bool(accept), "dialog_id": dialog.id,
                               "prompt_text": (prompt_text or "") if dialog.type == "prompt" else ""}).encode()
            await self._cdp_quiet(
                "Fetch.fulfillRequest",
                {"requestId": dialog.bridge_request_id, "responseCode": 200,
                 "responseHeaders": [{"name": "Content-Type", "value": "application/json"},
                                     {"name": "Access-Control-Allow-Origin", "value": "*"}],
                 "body": base64.b64encode(body).decode()},
                session_id=session_id, timeout=5.0, what=f"bridge fulfill {dialog.id}",
            )
            return
        params: Dict[str, Any] = {"accept": accept}
        if prompt_text is not None and dialog.type == "prompt":
            params["promptText"] = prompt_text
        await self._cdp("Page.handleJavaScriptDialog", params, session_id=session_id, timeout=5.0)

    async def _respond_quiet(self, dialog: PendingDialog, *, accept: bool, prompt_text: Optional[str]) -> None:
        """Auto-policy / watchdog response (already archived by the caller); failures logged only."""
        try:
            await self._respond(dialog, accept=accept, prompt_text=prompt_text)
        except Exception as e:
            logger.debug("auto response failed for %s: %s", dialog.id, e)

    async def _handle_dialog_cdp(self, dialog: PendingDialog, *, accept: bool, prompt_text: str) -> None:
        """Agent response path. The dialog is retired regardless of outcome — a CDP
        error usually means it already closed (browser auto-dismissed after navigation)."""
        try:
            await self._respond(dialog, accept=accept, prompt_text=prompt_text)
        finally:
            self._retire_dialog(dialog.id, "agent")

    async def _dialog_timeout_expired(self, dialog_id: str) -> None:
        with self._state_lock:
            dialog = self._pending_dialogs.get(dialog_id)
        if dialog is None:
            return
        logger.warning("CDP supervisor %s: dialog %s (%s) auto-dismissed after %ss timeout",
                       self.task_id, dialog_id, dialog.type, self.dialog_timeout_s)
        # Archive with watchdog tag BEFORE unblocking the page.
        self._retire_dialog(dialog_id, "watchdog")
        await self._respond_quiet(dialog, accept=False, prompt_text=None)

    # ── Bookkeeping ──────────────────────────────────────────────────────────

    def _retire_dialog(self, dialog_id: str, closed_by: str) -> None:
        """Remove a pending dialog (archiving it with ``closed_by``) and cancel its watchdog."""
        with self._state_lock:
            dialog = self._pending_dialogs.pop(dialog_id, None)
            if dialog is not None:
                self._archive_dialog_locked(dialog, closed_by)
        handle = self._dialog_watchdogs.pop(dialog_id, None)
        if handle is not None:
            handle.cancel()

    def _archive_dialog_locked(self, dialog: PendingDialog, closed_by: str) -> None:
        """Move a pending dialog to the recent_dialogs ring buffer. Must hold state_lock."""
        record = DialogRecord(id=dialog.id, type=dialog.type, message=dialog.message, opened_at=dialog.opened_at,
                              closed_at=time.time(), closed_by=closed_by, frame_id=dialog.frame_id)
        self._recent_dialogs = _trim_ring([*self._recent_dialogs, record], RECENT_DIALOGS_MAX)

    async def _on_dialog_closed(self, params: Dict[str, Any], session_id: Optional[str]) -> None:
        # ``Page.javascriptDialogClosed`` carries only ``result``/``userInput``: match by
        # session id and clear the oldest native dialog on it (the JS thread blocks while
        # a dialog is up, so at most one is in flight). Bridge dialogs resolve via Fetch.
        with self._state_lock:
            candidate = next((d.id for d in self._pending_dialogs.values()
                              if d.cdp_session_id == session_id and d.bridge_request_id is None), None)
        if candidate:
            self._retire_dialog(candidate, "remote")

    # CDP event → handler(self, params, session_id); merged into CDPSupervisor._EVENT_HANDLERS.
    EVENT_HANDLERS: Dict[str, Callable[..., Any]] = {
        "Page.javascriptDialogOpening": _on_dialog_opening,
        "Page.javascriptDialogClosed": _on_dialog_closed,
        "Fetch.requestPaused": _on_fetch_paused,
    }
