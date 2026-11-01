"""Keep a filled payment card out of screenshots until its page is gone.

A vault card fill writes the number and CVC into the page. Text that echoes them is scrubbed at the
egress boundary (``agent.redact_cards``), but pixels are not: a screenshot taken to "check the form"
carries the full card number into the model's context, the provider's logs and the session history.

From the moment a card is written until the filled document is gone (submit or any navigation loads a
new document, the tab closes, the browser session ends), every capture path for that task refuses:

- ``browser_exec``: the harness's ``cdp()`` refuses the capture methods for the call, nothing is attached;
- ``browser_cdp``: ``Page.captureScreenshot`` and the other pixel-producing methods refuse;
- ``vision_analyze``: local images written after the fill refuse.

The filled document is identified by its tab's target id plus ``performance.timeOrigin``, which every new
document (including a reload or a form POST) changes; the tab is re-read wherever the supervisor points
later, so filling or focusing another tab never ends the first tab's protection. A single-page checkout that submits without loading a new document stays
protected until it does: the guard errs toward refusing a capture. State is per profile and task, in memory.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

_DOCUMENT_ID_JS = "String(performance.timeOrigin)"
# CDP text when the supervisor's page session died with its tab: that page (and the card in it) is gone.
_GONE_MARKERS = ("session with given id not found", "no attached page session", "target closed")
_PROBE_TIMEOUT_S = 5.0

CDP_CAPTURE_METHODS = frozenset({
    "Page.captureScreenshot", "Page.startScreencast", "Page.printToPDF", "HeadlessExperimental.beginFrame",
})


@dataclass
class _CardFill:
    origin: str
    target_id: str  # the tab written into; "" when it could not be read (the supervisor's tab is judged then)
    document_id: str
    filled_at: float


# (profile home, task id) → {tab target id → _CardFill}: every live fill is kept, so filling a second tab
# (which refocuses the supervisor) never forgets the first one.
_LIVE: dict = {}
_LOCK = threading.Lock()


def _key(task_id: Optional[str]) -> tuple[str, str]:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home()), task_id or "default"


def _cvc_key(task: str, target_id: str) -> str:
    return f"{task}\x00{target_id}"


def _current_document(task_id: str) -> dict:
    """The supervisor's tab: {"id"}, {"gone": True}, or {} when it cannot be read now."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    supervisor = SUPERVISOR_REGISTRY.get(task_id)
    if supervisor is None:
        return {"gone": True}
    res = supervisor.evaluate_runtime(_DOCUMENT_ID_JS)
    if res.get("ok"):
        return {"id": str(res.get("result") or "")}
    error = str(res.get("error") or "").lower()
    return {"gone": True} if any(marker in error for marker in _GONE_MARKERS) else {}


def _current_target(task_id: str) -> str:
    """Target id of the supervisor's tab (the one a fill writes into), "" when it cannot be read."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    try:
        cdp = SUPERVISOR_REGISTRY.capture(task_id, timeout=_PROBE_TIMEOUT_S)
        reply = cdp.call("Target.getTargetInfo", session_id=cdp.page_session_id, timeout=_PROBE_TIMEOUT_S)
        return str(reply["result"]["targetInfo"]["targetId"])
    except Exception:  # health: allow BLE001 -- unknown tab: the fill is then judged on the supervisor's tab
        return ""


def _tab_document(task_id: str, fill: _CardFill) -> dict:
    """The filled tab's document wherever the supervisor points now: {"id"}, {"gone": True} once the tab
    closed or the browser session ended, {} when it cannot be read (protection stays)."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    if SUPERVISOR_REGISTRY.get(task_id) is None:
        return {"gone": True}
    if not fill.target_id:
        return _current_document(task_id)
    try:
        cdp = SUPERVISOR_REGISTRY.capture(task_id, timeout=_PROBE_TIMEOUT_S)
        targets = cdp.call("Target.getTargets", timeout=_PROBE_TIMEOUT_S)["result"]["targetInfos"]
        if fill.target_id not in {t.get("targetId") for t in targets}:
            return {"gone": True}
        attach = cdp.call("Target.attachToTarget", {"targetId": fill.target_id, "flatten": True},
                          timeout=_PROBE_TIMEOUT_S)
        sid = attach["result"]["sessionId"]
        try:
            reply = cdp.call("Runtime.evaluate", {"expression": _DOCUMENT_ID_JS, "returnByValue": True},
                             session_id=sid, timeout=_PROBE_TIMEOUT_S)
        finally:
            cdp.call("Target.detachFromTarget", {"sessionId": sid}, timeout=_PROBE_TIMEOUT_S)
        return {"id": str(reply["result"]["result"].get("value") or "")}
    except Exception:  # health: allow BLE001 -- a tab that cannot be read now keeps its protection
        return {}


def mark_card_filled(task_id: Optional[str], origin: str, cvc: str = "") -> None:
    """Called before a card's values are written into ``origin``'s page (the fill itself does not
    navigate, so the tab and document read here are the ones that will hold the card)."""
    from agent.redact_cards import set_page_cvc

    key = _key(task_id)
    target_id = _current_target(key[1])
    fill = _CardFill(origin=origin, target_id=target_id,
                     document_id=str(_current_document(key[1]).get("id") or ""), filled_at=time.time())
    with _LOCK:
        _LIVE.setdefault(key, {})[target_id] = fill
    if cvc:
        set_page_cvc(_cvc_key(key[1], target_id), cvc)


def clear(task_id: Optional[str]) -> None:
    from agent.redact_cards import drop_page_cvc

    key = _key(task_id)
    with _LOCK:
        fills = _LIVE.pop(key, {})
    for target_id in fills:
        drop_page_cvc(_cvc_key(key[1], target_id))


def _drop(key: tuple[str, str], fill: _CardFill) -> None:
    from agent.redact_cards import drop_page_cvc

    with _LOCK:
        fills = _LIVE.get(key, {})
        if fills.get(fill.target_id) is not fill:
            return  # a newer fill of the same tab replaced it
        del fills[fill.target_id]
        if not fills:
            _LIVE.pop(key, None)
    drop_page_cvc(_cvc_key(key[1], fill.target_id))


def live_fill(task_id: Optional[str]) -> Optional[_CardFill]:
    """A card fill of the task whose document still lives, each filled tab re-checked now; fills whose
    tab navigated or closed (or whose browser session ended) are dropped with their CVC mask. None when
    no fill is live. A tab that cannot be read for another reason keeps the protection."""
    key = _key(task_id)
    with _LOCK:
        fills = list(_LIVE.get(key, {}).values())
    alive = None
    for fill in fills:
        doc = _tab_document(key[1], fill)
        if doc.get("gone") or (doc.get("id") and fill.document_id and doc["id"] != fill.document_id):
            _drop(key, fill)
        elif alive is None:
            alive = fill
    return alive


def capture_refusal(task_id: Optional[str]) -> Optional[str]:
    """The refusal text while a filled card is on the task's page, else None."""
    fill = live_fill(task_id)
    if fill is None:
        return None
    return (f"Screenshots are blocked while a payment card is filled into {fill.origin}: the image would "
            "carry the card number. Submit the form or navigate away first; check the fields with text "
            "reads instead (card values come back masked).")


def image_refusal(task_id: Optional[str], image_path: str) -> Optional[str]:
    """``vision_analyze`` on a local image written after one of the task's card fills, while one is live."""
    with _LOCK:
        fills = list(_LIVE.get(_key(task_id), {}).values())
    if not fills:
        return None
    from tools.image_source import ResolveContext, host_file_for

    target = host_file_for(image_path, ResolveContext(task_id=task_id))  # the file vision_analyze would read
    if target is None:
        return None  # a URL or a sandbox-only path: not a capture of the page
    try:
        written = os.path.getmtime(target)
    except (OSError, ValueError):
        return None  # a missing file
    return capture_refusal(task_id) if written >= min(f.filled_at for f in fills) - 1 else None
