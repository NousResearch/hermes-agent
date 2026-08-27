"""`computer_use` tool entry point: any-model desktop control (macOS/Windows/Linux) via cua-driver.
Return contract: text-only results are a JSON string; captures / `capture_after=True` return
``{"_multimodal": True, "content": [text, image_url], "text_summary": <fallback>}`` (run_agent.py /
the Anthropic adapter turn it into provider-specific image tool content)."""

from __future__ import annotations

import atexit
import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import threading
import uuid
from collections import namedtuple
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement, image_dimensions_from_bytes

logger = logging.getLogger(__name__)

# ── Approval & safety ───────────────────────────────────────────────────────
# Optional computer_use-specific prompt handed to the shared gate as its explicit ``approval_callback``; when None the
# gate resolves the per-thread CLI callback (``tools.terminal_tool.set_approval_callback``) like every other tool, so
# in-tree hosts never call this. Same contract as that callback: ``cb(command, description, **kw)`` ->
# "once" | "session" | "always" | "deny" | "timeout".
_approval_callback = None

def set_approval_callback(cb) -> None:
    global _approval_callback
    _approval_callback = cb


# Actions that read, not mutate. Always allowed.
_SAFE_ACTIONS = frozenset({
    "capture", "wait", "list_apps", "list_windows",
})

# Actions that mutate user-visible state. Go through approval.
_DESTRUCTIVE_ACTIONS = frozenset({
    "click", "double_click", "right_click", "middle_click",
    "drag", "scroll", "type", "key", "set_value", "focus_app",
})

# Hard-blocked key combinations. Mirrored from #4562 — these are destructive
# regardless of approval level (e.g. logout kills the session Hermes runs in).
_BLOCKED_KEY_COMBOS = {
    frozenset({"cmd", "shift", "backspace"}), frozenset({"cmd", "option", "backspace"}),  # empty trash / force delete
    frozenset({"cmd", "ctrl", "q"}), frozenset({"cmd", "shift", "q"}),                    # lock screen / log out
    frozenset({"cmd", "option", "shift", "q"}), frozenset({"win", "l"}),                  # force log out / lock
    frozenset({"ctrl", "option", "delete"}), frozenset({"ctrl", "option", "del"}), frozenset({"option", "f4"}),
}
_KEY_ALIASES = {"command": "cmd", "control": "ctrl", "alt": "option", "⌘": "cmd", "⌥": "option",
                "windows": "win", "super": "win", "meta": "win"}
_BLOCKED_TYPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (  # dangerous shell patterns for `type` (last one: fork bomb)
    r"curl\s+[^|]*\|\s*bash", r"curl\s+[^|]*\|\s*sh", r"wget\s+[^|]*\|\s*bash",
    r"\bsudo\s+rm\s+-[rf]", r"\brm\s+-rf\s+/\s*$", r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}")]

def _canon_key_combo(keys: str) -> frozenset:
    # Split on "+" AND "-": cua-driver accepts hyphenated combos, so "ctrl-alt-delete" would bypass otherwise.
    return frozenset(_KEY_ALIASES.get(p, p) for p in (q.strip().lower() for q in re.split(r"\s*[+\-]\s*", keys)) if p)

def _reject_unsafe(action: str, args: Dict[str, Any]) -> Optional[str]:
    """JSON error for hard-blocked input, else None. Runs BEFORE the approval prompt."""
    if action == "type" and (pat := next((p.pattern for p in _BLOCKED_TYPE_PATTERNS if p.search(args.get("text", ""))), None)):
        return json.dumps({"error": f"blocked pattern in type text: {pat!r}",
                           "hint": "Dangerous shell patterns cannot be typed via computer_use."})
    if action == "key" and (blocked := next((b for b in _BLOCKED_KEY_COMBOS
                                             if b.issubset(_canon_key_combo(args.get("keys", "")))), None)) is not None:
        return json.dumps({"error": f"blocked key combo: {sorted(blocked)}", "hint": "Destructive system shortcuts are hard-blocked."})
    if args.get("bring_to_front") and args.get("delivery_mode") != "foreground":
        return json.dumps({"error": "bring_to_front requires delivery_mode='foreground'",
                           "code": "bring_to_front_requires_foreground"})
    return None

def _input_target_mismatch(backend, requested_app: str) -> Optional[str]:
    """Current sticky-target app when it provably differs from *requested_app*: both known and neither a substring
    of the other ('Google-chrome' vs 'chrome'). Unknown target -> None (fail open; the verify ladder catches it)."""
    last_app = getattr(backend, "_last_app", None)
    current, wanted = (last_app or "").strip().lower(), requested_app.strip().lower()
    return None if not current or not wanted or wanted in current or current in wanted else last_app

# ── Backend selection — env-swappable for tests ─────────────────────────────
# Per-Hermes-session cached backends (own cua-driver session, native target, refs, grant namespace).
_backend_lock = threading.Lock()
_backend: Optional[ComputerUseBackend] = None  # backward-compatible empty-session injection hook (older tests)
_backends: Dict[str, ComputerUseBackend] = {}
_backend_call_locks: Dict[str, threading.RLock] = {}
_backend_permission_modes: Dict[str, str] = {}
# (home key, provider, model) → bool. The decision reads the active profile's config (auxiliary.vision
# override, declared supports_vision), so a multiplexed process must not serve profile A's verdict to B.
_AUX_VISION_ROUTE_CACHE: Dict[Tuple[str, str, str], bool] = {}
# Approval grants live in the shared store (``tools.approval``: session set + permanent allowlist), keyed by the
# gate's session key, so a computer_use "always" is one allowlist entry like any terminal pattern. Only the
# once-per-session escalation warning is tracked here.
_approval_lock = threading.Lock()
_escalation_warned: set = set()               # sids already warned that a bypass widened the driver mode

# Screenshot dedup: a tight capture→act→capture loop on a static screen resends the same ~1200px image every step.
# Every delivered frame is hashed (sha256 over mime + base64 payload); when the next capture of the SAME target
# (app, window) in the SAME session is byte-identical, the tool returns the normal text metadata (element index
# included) plus an explicit "screen unchanged" note and omits the image block. Append-only — no prior transcript
# message is rewritten, so prompt-cache prefixes stay intact. Staleness is bounded by a consecutive-omission streak
# cap: full pixels are re-delivered before compaction (which keeps only the newest image-bearing tool results)
# could evict the image the note refers to. State is per session; sessionless calls never dedup.
_screenshot_dedup_lock = threading.Lock()
_last_screenshot_state: Dict[str, Dict[str, Any]] = {}  # session_id -> {"digest", "target": (app, window), "streak"}
_SCREENSHOT_DEDUP_MAX_STREAK = 2

def _screenshot_dedup_check(session_id: str, digest: str, target: Tuple[str, str]) -> bool:
    """True when this capture should be delivered WITHOUT its image: the previous frame for this session had identical
    bytes for the same target and the omission streak is below _SCREENSHOT_DEDUP_MAX_STREAK. Any miss (new pixels,
    new target, streak exhausted, first capture) resets the stored state to this digest so the image goes out."""
    with _screenshot_dedup_lock:
        state = _last_screenshot_state.get(session_id)
        if (state is not None and state.get("digest") == digest and state.get("target") == target
                and int(state.get("streak", 0)) < _SCREENSHOT_DEDUP_MAX_STREAK):
            state["streak"] = int(state.get("streak", 0)) + 1
            return True
        _last_screenshot_state[session_id] = {"digest": digest, "target": target, "streak": 0}
        return False

def _reset_screenshot_dedup(session_id: Optional[str] = None) -> None:
    """Forget dedup state (all sessions, or one scoped key)."""
    with _screenshot_dedup_lock:
        if session_id is None:
            _last_screenshot_state.clear()
        else:
            _last_screenshot_state.pop(session_id, None)

def reset_screenshot_dedup(session_id: str) -> None:
    """Compaction boundary hook (mirrors ``reset_file_dedup``): the summary may have dropped the frame an
    "unchanged" note would point at, so the next capture of this session must deliver pixels again."""
    _reset_screenshot_dedup(_scoped_sid(session_id))

def _cua_permission_mode(session_id: str) -> str:
    """Map Hermes's approval bypass onto Cua's immutable mode; fails closed. Both identity namespaces are consulted
    (DB ``session_id`` and gateway ``session_key`` contextvar) or a gateway ``/yolo`` would be invisible here.
    Warns once per session that ``-z``/``--yolo`` swapped the driver onto a private ``unrestricted`` daemon, dropping
    the configured ceiling: deliberate (``unrestricted`` is not a config value) but easy to trigger by accident."""
    # Configured cua mode (standard | bounded; "standard" if unresolvable). bounded needs a
    # computer_use.capability_manifest — the backend fails loudly without it.
    configured = "standard"
    with contextlib.suppress(Exception):
        from tools.computer_use.cua_backend import _cua_configured_permission_mode
        configured = _cua_configured_permission_mode()
    with contextlib.suppress(Exception):
        from tools.approval import is_approval_bypass_active_for_session
        from tools.approval_context import get_current_session_key
        if is_approval_bypass_active_for_session(session_id) or (
                bool(key := get_current_session_key(default="")) and is_approval_bypass_active_for_session(key)):
            with _approval_lock:
                warn = (key := str(session_id or "")) not in _escalation_warned
                _escalation_warned.add(key)
            if warn:
                logger.warning(
                    "computer_use: approval bypass (--yolo / -z) escalated the cua-driver permission mode from the "
                    "configured '%s' to 'unrestricted' for this session. Runtime approval prompts are disabled and the "
                    "driver's residual ceilings no longer apply. Drop the bypass flag to keep '%s', or declare a "
                    "version-3 computer_use.capability_manifest to keep a ceiling on bypassed runs.", configured, configured)
            return "unrestricted"
    return configured

def _new_backend(permission_mode: str) -> ComputerUseBackend:
    backend_name = os.environ.get("HERMES_COMPUTER_USE_BACKEND", "cua").lower()
    if backend_name in {"cua", "cua-driver", ""}:
        from tools.computer_use.cua_backend import CuaDriverBackend
        return CuaDriverBackend(permission_mode=permission_mode)
    if backend_name != "noop":
        raise RuntimeError(f"Unknown HERMES_COMPUTER_USE_BACKEND={backend_name!r}")
    return _NoopBackend()  # pragma: no cover

def _install_backend(sid: str, backend: ComputerUseBackend, permission_mode: str) -> ComputerUseBackend:
    """Record a backend in the session caches (the empty session also mirrors it onto the ``_backend`` hook).
    Caller holds ``_backend_lock``."""
    global _backend
    _backends[sid], _backend_permission_modes[sid] = backend, permission_mode
    _backend_call_locks[sid] = threading.RLock()
    _backend = backend if sid == "" else _backend
    return backend

def _get_backend(session_id: str = "") -> ComputerUseBackend:
    bare_sid, sid = str(session_id or ""), _scoped_sid(session_id)
    while True:
        with _backend_lock:
            # Mode resolved under the cache lock; YOLO mutation never holds the approval lock while releasing it.
            permission_mode = _cua_permission_mode(bare_sid)  # approval state is keyed by the Hermes session id
            if sid == "" and _backend is not None and sid not in _backends:
                _install_backend(sid, _backend, permission_mode)  # fold the injection hook into the cache
            if (cached := _backends.get(sid)) is None:
                backend = _new_backend(permission_mode)
                backend.start()  # under the cache lock: one backend per session; a concurrent toggle releases it
                return _install_backend(sid, backend, permission_mode)
            if _backend_permission_modes.get(sid, "standard") == permission_mode:
                return cached
            # Cua's mode is immutable after daemon startup: a /yolo toggle replaces only this session's backend.
            _, stale_lock = _detach_locked(sid)  # stopped outside the cache lock; the loop re-reads the mode first
        _stop_backend(cached, stale_lock, lambda e: None)

def release_computer_use_session(session_id: str) -> bool:
    """Release one session-owned backend (lifecycle seam for hosts/plugins); idempotent, True iff one was released.
    Cache entries are removed BEFORE stopping so new lookups cannot retain the stale target/ref namespace. Approval
    grants are not touched here: they live in the shared store and die with ``tools.approval.clear_session``."""
    sid = _scoped_sid(session_id)
    _reset_screenshot_dedup(sid)  # the next capture of a re-created session must deliver pixels
    with _backend_lock:
        backend, call_lock = _detach_locked(sid)
    if backend is None:
        return False
    _stop_backend(backend, call_lock,
                  lambda e: logger.debug("computer_use backend release failed for session %s", sid, exc_info=True))
    return True

@atexit.register
def _shutdown_backend_atexit() -> None:
    """Stop all cached backends so cua-driver subprocesses don't outlive us. atexit only, no signal handlers: a
    ``SystemExit`` from a prompt_toolkit key binding corrupts its coroutine state and makes the process unkillable.
    Never raises. Drops the global lock before stop(): teardown budgets 5s and must not block spawns.

    Each session backend holds a long-lived ``cua-driver`` subprocess, so without this a driver can survive
    the Hermes process that spawned it (#28152 item 3). #69903 kept the orphan from burning a core by
    disabling the cursor overlay; the process itself still lingered.
    """
    global _backend
    with _backend_lock:
        unique = {id(b): (b, _backend_call_locks.get(sid)) for sid, b in _backends.items()}
        if _backend is not None:
            unique.setdefault(id(_backend), (_backend, _backend_call_locks.get("")))
        _backend = None
        _backends.clear(), _backend_call_locks.clear(), _backend_permission_modes.clear()
    with _approval_lock:
        _escalation_warned.clear()
    for backend, call_lock in unique.values():
        _stop_backend(backend, call_lock, lambda e: logger.debug("cua-driver atexit teardown failed: %s", e))

def reset_backend_for_tests() -> None:  # pragma: no cover — tear down the cached backend and per-session state
    _shutdown_backend_atexit()
    _AUX_VISION_ROUTE_CACHE.clear()
    _reset_screenshot_dedup()

def _noop_stub(name: str, *params: str, result: Any = None):
    # Recording stub: positional args are folded in under *params* (declared params default to None). ``result`` may
    # be a factory of the recorded kwargs; None -> a trivial ok ActionResult.
    def method(self, *pos, **kw):
        self.calls.append((name, call := {**dict.fromkeys(params), **dict(zip(params, pos)), **kw}))
        return result(call) if callable(result) else ActionResult(ok=True, action=name) if result is None else result
    return method

class _NoopBackend(ComputerUseBackend):  # pragma: no cover
    """Test/CI stub (HERMES_COMPUTER_USE_BACKEND=noop). Records ``(name, kwargs)`` calls; returns trivial results."""

    def __init__(self) -> None: self.calls: List[Tuple[str, Dict[str, Any]]] = []
    start = stop = lambda self: None
    def is_available(self) -> bool: return True

    capture = _noop_stub("capture", "mode", "app", "pid", "window_id", result=lambda kw: CaptureResult(
        mode=kw["mode"] or "som", width=1024, height=768, png_b64=None, elements=[], app=kw["app"] or "", window_title=""))
    click, drag, scroll = _noop_stub("click"), _noop_stub("drag"), _noop_stub("scroll")
    type_text, key, set_value = _noop_stub("type", "text"), _noop_stub("key", "keys"), _noop_stub("set_value", "value", "element")
    list_apps, list_windows = _noop_stub("list_apps", result=[]), _noop_stub("list_windows", result=[])
    focus_app = _noop_stub("focus_app", "app", "raise_window")

# ── Dispatch ────────────────────────────────────────────────────────────────
def handle_computer_use(args: Dict[str, Any], **kwargs) -> Any:
    """Main entry point (tools.registry): a JSON string (text-only) or a dict marked `_multimodal`. Order: hard
    blocks (_reject_unsafe) -> approval scopes (destructive action, then 'bring_to_front' — persistent focus is a
    separate visible side effect with its own scope) -> backend -> dispatch under the session call lock."""
    action = (args.get("action") or "").strip().lower()
    if not action:
        return json.dumps({"error": "missing `action`"})
    # Per-run key for approval-state and daemon-mode isolation across
    # concurrent sessions.
    session_id = str(kwargs.get("session_id") or "")

    # Safety: validate actions before approval prompt.
    if action == "type":
        text = args.get("text", "")
        pat = _is_blocked_type(text)
        if pat:
            return json.dumps({
                "error": f"blocked pattern in type text: {pat!r}",
                "hint": "Dangerous shell patterns cannot be typed via computer_use.",
            })

    if action == "key":
        keys = args.get("keys", "")
        combo = _canon_key_combo(keys)
        for blocked in _BLOCKED_KEY_COMBOS:
            if blocked.issubset(combo) and len(blocked) <= len(combo):
                return json.dumps({
                    "error": f"blocked key combo: {sorted(blocked)}",
                    "hint": "Destructive system shortcuts are hard-blocked.",
                })

    if args.get("bring_to_front") and args.get("delivery_mode") != "foreground":
        return json.dumps({
            "error": "bring_to_front requires delivery_mode='foreground'",
            "code": "bring_to_front_requires_foreground",
        })

    # Approval gate (destructive actions only).
    if action in _DESTRUCTIVE_ACTIONS:
        err = _request_approval(action, args, session_id)
        if err is not None:
            return err
    try:
        backend = _get_backend(session_id=session_id)
    except Exception as e:
        return json.dumps({"error": f"computer_use backend unavailable: {e}",
                           "hint": "If the cua-driver binary is missing, run `hermes computer-use install`. "
                                   "If a Python dependency is missing, the error above shows the exact install command."})
    try:
        with _backend_lock:
            call_lock = _backend_call_locks.setdefault(session_id, threading.RLock())
        with call_lock:
            return _dispatch(backend, action, args, session_id=session_id or None)
    except Exception as e:
        logger.exception("computer_use %s failed", action)
        return json.dumps({"error": f"{action} failed: {e}"})

def _request_approval(action: str, args: Dict[str, Any]) -> Optional[str]:
    """None if approved, else a JSON error string. The decision (yolo bypass, session/permanent grants, CLI prompt,
    gateway pending, cron/unattended policy, fail-closed with nobody to ask) is ``tools.approval``'s shared gate,
    so a computer_use grant is one store entry like any terminal pattern. Scope key ``cua:<action>:<mode>``:
    foreground delivery is a visible focus change, so a background ``session`` grant must NOT cover it (#67052).
    """
    from tools.approval import _run_approval_gate

    mode = "foreground" if args.get("delivery_mode") == "foreground" else "background"
    description = f"Allow computer_use to perform `{action}`?"
    result = _run_approval_gate(
        pattern_key=f"cua:{action}:{mode}", description=description,
        display_target=f"computer_use: {_summarize_action(action, args)}", approval_callback=_approval_callback,
        subject=f"computer_use `{action}` requires approval", noun="desktop actions",
        advice="Find an alternative approach that avoids driving the desktop.",
        autoapprove_log_prefix="computer_use action in non-interactive non-gateway context",
        fail_closed_when_no_human=True,
        no_human_block_message=(f"BLOCKED: computer_use `{action}` requires approval but no interactive user or "
                                "gateway is present to approve it."),
    )
    if result.get("approved"):
        return None
    return json.dumps({"error": result.get("message") or "denied by user", "action": action})

def _summarize_action(action: str, args: Dict[str, Any]) -> str:
    fg = " [FOREGROUND — briefly raises the window / changes focus]" if args.get("delivery_mode") == "foreground" else ""
    return _ACTIONS.get(action, _ActionSpec(None)).summarize(action, args, fg)

# --- handlers: (backend, action, args, **delivery) -> ActionResult (_dispatch applies the follow-up capture) or a
#     final str/dict result. `delivery` = delivery_mode + bring_to_front; only input actions use it.

def _xy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Click semantics: a coordinate only counts when its x is set (a bare y is not a point)."""
    return dict(x=coord[0], y=coord[1]) if (coord := args.get("coordinate")) and coord[0] is not None else dict(x=None, y=None)

    if action == "capture":
        mode = str(args.get("mode", "som"))
        if mode not in {"som", "vision", "ax"}:
            return json.dumps({"error": f"bad mode {mode!r}; use som|vision|ax"})
        capture_kwargs: Dict[str, Any] = {"mode": mode, "app": args.get("app")}
        if args.get("pid") is not None or args.get("window_id") is not None:
            capture_kwargs.update({
                "pid": args.get("pid"),
                "window_id": args.get("window_id"),
            })
        cap = backend.capture(**capture_kwargs)
        return _capture_response(cap)

def _do_click(backend, action, args, button=None, count=1, **delivery):
    return backend.click(element=args.get("element"), **_xy(args), button=button or args.get("button") or "left",
                         click_count=count, modifiers=args.get("modifiers"), **delivery)

def _do_drag(backend, action, args, **delivery):
    src, dst = args.get("from_coordinate"), args.get("to_coordinate")
    if (args.get("from_element") is None or args.get("to_element") is None) and not (src and dst):
        return json.dumps({"error": "drag requires from_coordinate/to_coordinate or from_element/to_element"})
    return backend.drag(from_element=args.get("from_element"), to_element=args.get("to_element"),
                        from_xy=tuple(src) if src else None, to_xy=tuple(dst) if dst else None,
                        button=args.get("button", "left"), modifiers=args.get("modifiers"), **delivery)

def _do_scroll(backend, action, args, **delivery):
    return backend.scroll(direction=args.get("direction", "down"), amount=int(args.get("amount", 3)),
                          element=args.get("element"), **_scroll_xy(args), modifiers=args.get("modifiers"), **delivery)

def _do_capture(backend, action, args, session_id=None, **_):
    if (mode := str(args.get("mode", "som"))) not in {"som", "vision", "ax"}:
        return json.dumps({"error": f"bad mode {mode!r}; use som|vision|ax"})
    # pid/window_id forwarded only when given so older backends keep their defaults.
    return _capture_response(backend.capture(mode=mode, app=args.get("app"),
                                             **{k: args[k] for k in ("pid", "window_id") if args.get(k) is not None}),
                             session_id=session_id)

    # delivery_mode / bring_to_front thread through every input action so the
    # model can escalate background → foreground per cua-driver's ladder.
    delivery_mode = args.get("delivery_mode")
    bring_to_front = bool(args.get("bring_to_front"))

# Unknown actions are never aliased (no repairing bad model output), but the nearest real action is named as guidance.
_ACTION_SUGGESTIONS = {
    "hotkey": "key", "press_key": "key", "keypress": "key", "key_combo": "key", "shortcut": "key", "type_text": "type",
    "input_text": "type", "screenshot": "capture", "get_window_state": "capture", "left_click": "click", "mouse_click": "click",
}

    if action in {"click", "double_click", "right_click", "middle_click"}:
        button = args.get("button")
        click_count = 1
        if action == "double_click":
            click_count = 2
        elif action == "right_click":
            button = "right"
        elif action == "middle_click":
            button = "middle"
        else:
            button = button or "left"
        element = args.get("element")
        coord = args.get("coordinate") or (None, None)
        x, y = (coord[0], coord[1]) if coord and coord[0] is not None else (None, None)
        res = backend.click(
            element=element if element is not None else None,
            x=x, y=y, button=button or "left", click_count=click_count,
            modifiers=args.get("modifiers"),
            delivery_mode=delivery_mode, bring_to_front=bring_to_front,
        )
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "drag":
        has_elements = args.get("from_element") is not None and args.get("to_element") is not None
        has_coords = args.get("from_coordinate") and args.get("to_coordinate")
        if not has_elements and not has_coords:
            return json.dumps({
                "error": "drag requires from_coordinate/to_coordinate or from_element/to_element",
            })
        res = backend.drag(
            from_element=args.get("from_element"),
            to_element=args.get("to_element"),
            from_xy=tuple(args["from_coordinate"]) if args.get("from_coordinate") else None,
            to_xy=tuple(args["to_coordinate"]) if args.get("to_coordinate") else None,
            button=args.get("button", "left"),
            modifiers=args.get("modifiers"),
            delivery_mode=delivery_mode, bring_to_front=bring_to_front,
        )
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "scroll":
        coord = args.get("coordinate") or (None, None)
        res = backend.scroll(
            direction=args.get("direction", "down"),
            amount=int(args.get("amount", 3)),
            element=args.get("element"),
            x=coord[0] if coord and coord[0] is not None else None,
            y=coord[1] if coord and coord[1] is not None else None,
            modifiers=args.get("modifiers"),
            delivery_mode=delivery_mode, bring_to_front=bring_to_front,
        )
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "type":
        res = backend.type_text(args.get("text", ""),
                                delivery_mode=delivery_mode, bring_to_front=bring_to_front)
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "key":
        res = backend.key(args.get("keys", ""),
                          delivery_mode=delivery_mode, bring_to_front=bring_to_front)
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "set_value":
        value = args.get("value")
        if value is None:
            return json.dumps({"error": "set_value requires `value`"})
        res = backend.set_value(value=str(value), element=args.get("element"))
        return _maybe_follow_capture(backend, res, capture_after)

    # Do NOT alias unknown actions (we never repair bad model output), but
    # name the nearest real action: live QA showed a model emitting
    # "hotkey"/"press_key" and getting zero guidance from the bare error.
    _suggestions = {
        "hotkey": "key", "press_key": "key", "keypress": "key",
        "key_combo": "key", "shortcut": "key",
        "type_text": "type", "input_text": "type",
        "screenshot": "capture", "get_window_state": "capture",
        "left_click": "click", "mouse_click": "click",
    }
    hint = _suggestions.get(str(action))
    if hint:
        return json.dumps({
            "error": (
                f"unknown action {action!r} — did you mean {hint!r}? "
                "See the action enum in the tool schema."
            )
        })
    return json.dumps({"error": f"unknown action {action!r}"})


# ---------------------------------------------------------------------------
# Response shaping
# ---------------------------------------------------------------------------

def _classify_action_result(res: ActionResult) -> Dict[str, Any]:
    """Next ladder step from semantic evidence, in precedence order. Escalation is advisory: it never overrides
    a confirmed effect nor licenses repeating input."""
    if res.effect == "confirmed" or res.verified is True:
        return {"decision": "done"}
    if res.effect == "unverifiable":
        return {
            "decision": "verify_fresh_state",
            "hint": (
                "Input was delivered but not confirmed. Re-capture and check "
                "the result BEFORE any retry — do not repeat the input on an "
                "escalation recommendation alone."
            ),
        }
    if res.effect == "suspected_noop" or not res.ok or res.code is not None:
        decision: Dict[str, Any] = {"decision": "escalate"}
        if isinstance(res.escalation, dict):
            decision["recommended"] = res.escalation.get("recommended")
        decision["hint"] = (
            "The input likely did not land. Climb one rung following "
            "`recommended`: 'px' → re-issue by coordinate; 'foreground' (or a "
            "failed pixel click) → re-issue with delivery_mode='foreground' "
            "(separate approval). Do not predict the rung from the app being "
            "Electron/Chromium — react to this signal."
        )
        return decision
    # Transport success without semantic proof is not proof of effect.
    return {
        "decision": "verify_fresh_state",
        "hint": (
            "Transport succeeded but the effect is unproven. Re-capture and "
            "confirm before continuing."
        ),
    }

def _present(**fields: Any) -> Dict[str, Any]:
    return {k: v for k, v in fields.items() if v}  # only the truthy optional fields, in the given order

def _action_payload(res: ActionResult) -> Dict[str, Any]:
    # cua-driver's structured verdict fields only when returned (None = old driver). ok is transport success;
    # effect/escalation are the semantic verdict.
    return {"ok": res.ok, "action": res.action, **_present(message=res.message),
            **{k: v for k in ("verified", "effect", "escalation", "path", "degraded", "delivery_mode", "code")
               if (v := getattr(res, k)) is not None}, **_present(meta=res.meta),
            "verdict": _classify_action_result(res)}

def _text_response(res: ActionResult) -> str:
    return json.dumps(_action_payload(res))


def _enrich_escalation(res: ActionResult) -> Optional[Dict[str, Any]]:
    """Return the driver's escalation dict unchanged."""
    return res.escalation


# Fixed cap for the AX `elements` array surfaced in a capture response. Dense
# UIs (Electron apps, Obsidian, JetBrains IDEs) can publish 500+ AX nodes,
# which would exhaust session context after a single capture. The full,
# untruncated tree is always written to an `elements_file` spill (see
# _capture_lost_detail) so nothing is lost — read_file/search_files it when the
# target isn't in the surfaced window.
_DEFAULT_MAX_ELEMENTS = 100
_MIN_PROVIDER_IMAGE_DIMENSION = 8
# Some AX trees (Discord/Slack via UIA, Electron chat clients) expose ENTIRE message bodies as labels; uncapped
# they blew the tool-result budget and leaked private chat text. Labels identify a control, not text extraction.
_MAX_ELEMENT_LABEL_CHARS = 120
# Bounded cache trails: every dense capture can spill, and CLI-only sessions never run the gateway's media cleanup.
_MAX_SPILL_FILES = _MAX_CAPTURE_FILES = 20

def _capture_image_format(cap: CaptureResult) -> Tuple[str, str]:
    # (MIME, file extension): cua-driver's explicit MIME type, else sniff the base64 prefix (JPEG starts with /9j/,
    # PNG with iVBOR). The extension matches the on-disk bytes for MIME sniffing.
    mime = cap.image_mime_type or ("image/jpeg" if (cap.png_b64 or "").startswith("/9j/") else "image/png")
    return mime, (".jpg" if mime.lower() == "image/jpeg" else ".png")

def _bounds_unknown(bounds) -> bool:
    # No real geometry: KDE/Qt apps report [0, 0, 0, 0] for elements clickable by index; serializing that as a
    # rect invites coordinate=[0, 0] clicks.
    with contextlib.suppress(TypeError, ValueError):
        return all(int(v) == 0 for v in bounds)
    return False

def _element_to_dict(e: UIElement) -> Dict[str, Any]:
    # A zero rect is "geometry unknown", not a position — null it so no coordinate= is ever derived from it (the index still works).
    return {"index": e.index, "role": e.role, "label": e.label[:_MAX_ELEMENT_LABEL_CHARS],
            "bounds": None if _bounds_unknown(e.bounds) else list(e.bounds), "app": e.app,
            **({"label_truncated": True} if len(e.label) > _MAX_ELEMENT_LABEL_CHARS else {})}

def _format_elements(elements: List[UIElement], max_lines: int = 40) -> List[str]:
    out = [f"  #{e.index} {e.role} {e.label.replace(chr(10), ' ')[:60]!r} "
           + ("@ bounds-unknown (click by element index)" if _bounds_unknown(e.bounds) else f"@ {e.bounds}")
           + (f" [{e.app}]" if e.app else "") for e in elements[:max_lines]]
    return out + ([f"  ... +{len(elements) - max_lines} more (call capture with app= to narrow)"] if len(elements) > max_lines else [])

def _bounds_hints(elements: List[UIElement], image_width: int, image_height: int) -> Tuple[Optional[float], Optional[str]]:
    """(scale, note) when element bounds live in a different coordinate space than the screenshot, else (None, None).
    On HiDPI displays AX bounds are native while the screenshot is downscaled, so coordinate= clicks read off the
    screenshot miss by the scale factor. 5% slack: window chrome can hang a few px past the captured frame without
    implying a different space. Scale heuristic: larger axis ratio wins."""
    if not elements or image_width <= 0 or image_height <= 0:
        return None, None
    max_x = max_y = 0
    for e in elements:
        try:
            x, y, w, h = e.bounds
        except (TypeError, ValueError):
            continue
        max_x, max_y = max(max_x, int(x) + int(w)), max(max_y, int(y) + int(h))
    if max_x <= image_width * 1.05 and max_y <= image_height * 1.05:
        return None, None
    note = (f"element bounds are in native desktop coordinates (extend to ~{max_x}x{max_y}), "
            f"NOT screenshot pixels ({image_width}x{image_height}). coordinate= clicks expect the native "
            "space — derive click points from element bounds, or scale screenshot positions up accordingly")
    return round(max(max_x / image_width, max_y / image_height), 2), note

_bounds_scale = lambda elements, image_width, image_height: _bounds_hints(elements, image_width, image_height)[0]  # noqa: E731
_bounds_space_note = lambda elements, image_width, image_height: _bounds_hints(elements, image_width, image_height)[1]  # noqa: E731

def _capture_view(cap: CaptureResult, max_elements: int) -> SimpleNamespace:
    """One capture's derived facts, computed once for every response branch: ``visible`` is the capped element list,
    ``dims_omitted`` an image below the provider minimum."""
    visible, dims = cap.elements[:max_elements], None
    with contextlib.suppress(Exception):  # (width, height) of the inline PNG/JPEG screenshot, else the backend's
        dims = image_dimensions_from_bytes(base64.b64decode(cap.png_b64, validate=False)) if cap.png_b64 else None
    width, height = dims or (cap.width, cap.height)
    scale, note = _bounds_hints(visible, width, height)
    # Capped labels / capped element array: spill the complete tree for on-demand reads.
    lost_detail = len(cap.elements) > len(visible) or any(len(e.label) > _MAX_ELEMENT_LABEL_CHARS for e in visible)
    too_small = bool(dims) and min(dims) < _MIN_PROVIDER_IMAGE_DIMENSION
    has_image = bool(cap.png_b64) and cap.mode != "ax" and not too_small
    return SimpleNamespace(cap=cap, visible=visible, total=len(cap.elements), width=width, height=height,
                           truncated=len(cap.elements) - len(visible), bounds_scale=scale, bounds_note=note,
                           elements_file=_spill_elements_to_file(cap) if lost_detail else None,
                           screenshot_path=_persist_capture_image(cap) if has_image else None,
                           dims_omitted=dims if too_small else None, has_image=has_image)

def _capture_response(cap: CaptureResult, max_elements: int = _DEFAULT_MAX_ELEMENTS) -> Any:
    total_elements = len(cap.elements)
    visible_elements = cap.elements[:max_elements]
    truncated_elements = max(0, total_elements - len(visible_elements))
    image_dimensions = _image_dimensions_from_b64(cap.png_b64 or "") if cap.png_b64 else None
    response_width = image_dimensions[0] if image_dimensions else cap.width
    response_height = image_dimensions[1] if image_dimensions else cap.height
    bounds_note = _bounds_space_note(visible_elements, response_width, response_height)
    bounds_scale = _bounds_scale(visible_elements, response_width, response_height)
    if bounds_note and bounds_scale:
        bounds_note += (
            f"; estimated scale ~{bounds_scale}x (screenshot position x "
            f"{bounds_scale} ≈ native coordinate)"
        )
    # When the in-context response drops detail (capped labels / capped element
    # array), spill the complete tree to a cache file so the model can read or
    # grep the full text on demand instead of losing it entirely.
    elements_file = (
        _spill_elements_to_file(cap)
        if _capture_lost_detail(cap, visible_elements, truncated_elements)
        else None
    )
    image_too_small = bool(
        image_dimensions
        and (
            image_dimensions[0] < _MIN_PROVIDER_IMAGE_DIMENSION
            or image_dimensions[1] < _MIN_PROVIDER_IMAGE_DIMENSION
        )
    )
    screenshot_path = (
        _persist_capture_image(cap)
        if cap.png_b64 and cap.mode != "ax" and not image_too_small
        else None
    )

    # Index only what's actually surfaced in the response — otherwise the
    # human-readable summary references element indices the model cannot
    # find in the JSON `elements` array (the surfaced window is capped at
    # _DEFAULT_MAX_ELEMENTS; the full tree spills to elements_file).
    element_index = _format_elements(visible_elements)
    summary_lines = [
        f"capture mode={cap.mode} {response_width}x{response_height}"
        + (f" app={cap.app}" if cap.app else "")
        + (f" window={cap.window_title!r}" if cap.window_title else ""),
        f"{total_elements} interactable element(s):",
    ]
    if bounds_note:
        summary_lines.append(f"  ({bounds_note})")
    if screenshot_path:
        summary_lines.append(
            f"  (shareable screenshot saved to {screenshot_path})"
        )
    if cap.note:
        summary_lines.append(f"  ({cap.note})")
    if elements_file:
        summary_lines.append(
            f"  (full element tree with untruncated labels saved to "
            f"{elements_file} — read_file/search_files it if you need "
            "dropped label text or elements beyond the cap)"
        )
    if element_index:
        summary_lines.extend(element_index)
    # Multimodal and AX paths both reference `summary`; build it once up-front
    # so the aux-vision routing branch (which fires before either path is
    # selected) has a valid value to hand to _route_capture_through_aux_vision.
    # The AX path appends the "truncated to N of M" note to summary_lines
    # below and rebuilds; the multimodal path keeps this version untouched.
    if image_too_small:
        summary_lines.append(
            f"  (screenshot omitted: {image_dimensions[0]}x{image_dimensions[1]} "
            f"is below the {_MIN_PROVIDER_IMAGE_DIMENSION}x{_MIN_PROVIDER_IMAGE_DIMENSION} "
            "provider minimum)"
        )
    summary = "\n".join(summary_lines)

    if cap.png_b64 and cap.mode != "ax" and not image_too_small:
        # Decide whether to hand the screenshot to the auxiliary.vision
        # pipeline (text-only result) or keep the multimodal envelope (main
        # model handles vision natively). Issue #24015: previously the
        # multimodal envelope was returned unconditionally, so non-vision
        # main models tripped HTTP 404 / 400 at the provider boundary even
        # when auxiliary.vision was explicitly configured to handle this.
        if _should_route_through_aux_vision():
            routed = _route_capture_through_aux_vision(
                cap, summary,
                visible_elements=visible_elements,
                truncated_elements=truncated_elements,
                elements_file=elements_file,
                screenshot_path=screenshot_path,
            )
            if routed is not None:
                return routed
            # Aux routing was requested but failed (vision node down, aux call
            # raised, empty analysis, etc.). Routing being requested means the
            # main model may not be able to consume images; falling through to
            # the multimodal envelope can break the capture with a provider
            # error. Degrade to the AX/SOM text payload instead so element
            # indices remain usable while vision is unavailable.
            summary_lines.append(
                "  (vision unavailable: the auxiliary vision model could not "
                "be reached; screenshot omitted. Element-index actions still "
                "work — drive via the element list above.)"
            )
            if truncated_elements:
                summary_lines.append(
                    f"  (response truncated to {len(visible_elements)} of "
                    f"{total_elements} elements; the full tree is in "
                    "elements_file — read_file/search_files it, or pass app= "
                    "to narrow scope)"
                )
            payload = {
                "mode": cap.mode,
                "width": response_width,
                "height": response_height,
                "app": cap.app,
                "window_title": cap.window_title,
                "elements": [_element_to_dict(e) for e in visible_elements],
                "total_elements": total_elements,
                "summary": "\n".join(summary_lines),
                "vision_unavailable": True,
            }
            if truncated_elements:
                payload["truncated_elements"] = truncated_elements
            if elements_file:
                payload["elements_file"] = elements_file
            if screenshot_path:
                payload["screenshot_path"] = screenshot_path
            if bounds_scale:
                payload["bounds_scale"] = bounds_scale
            return json.dumps(payload)

        # Prefer the explicit MIME type cua-driver attaches to its image
        # parts (Surface 7 of NousResearch/hermes-agent#47072 — trycua/cua#1961
        # made `mimeType` part of every MCP image-part response). Fall back
        # to base64-prefix sniffing for older cua-driver builds that didn't
        # carry the field. JPEG base64 starts with /9j/; PNG with iVBOR.
        _mime = cap.image_mime_type
        if not _mime:
            _b64_prefix = cap.png_b64[:8]
            _mime = "image/jpeg" if _b64_prefix.startswith("/9j/") else "image/png"
        # The multimodal response carries the screenshot, not the AX
        # elements array, so a "response truncated to N of M elements"
        # note would be inaccurate — skip it on this branch.
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": summary},
                {"type": "image_url",
                 "image_url": {"url": f"data:{_mime};base64,{cap.png_b64}"}},
            ],
            "text_summary": summary,
            "meta": {"mode": cap.mode, "width": response_width, "height": response_height,
                      "elements": total_elements, "png_bytes": cap.png_bytes_len,
                      **({"screenshot_path": screenshot_path} if screenshot_path else {}),
                      **({"elements_file": elements_file} if elements_file else {}),
                      **({"bounds_scale": bounds_scale} if bounds_scale else {})},
        }
    # AX-only (or image-missing fallback): text path actually carries the
    # `elements` array, so the truncation note applies here.
    if truncated_elements:
        summary_lines.append(
            f"  (response truncated to {len(visible_elements)} of {total_elements} elements; "
            "the full tree is in elements_file — read_file/search_files it, or pass app= to narrow scope)"
        )
    summary = "\n".join(summary_lines)
    payload: Dict[str, Any] = {
        "mode": cap.mode,
        "width": response_width,
        "height": response_height,
        "app": cap.app,
        "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in visible_elements],
        "total_elements": total_elements,
        "summary": summary,
    }
    if truncated_elements:
        payload["truncated_elements"] = truncated_elements
    if elements_file:
        payload["elements_file"] = elements_file
    if bounds_scale:
        payload["bounds_scale"] = bounds_scale
    return json.dumps(payload)

# ── Cache files (screenshots, element spills, vision temps) ─────────────────
def _cache_file(subdir: str, legacy: str, name: str, pattern: str = "", cap: int = 0):
    """Path for a new file under ``$HERMES_HOME/<subdir>`` (dir created). With ``pattern``/``cap``, first unlinks the
    oldest matching files so at most ``cap - 1`` remain (best-effort)."""
    from hermes_constants import get_hermes_dir  # lazy so tests can patch get_hermes_dir
    cache_dir = get_hermes_dir(subdir, legacy)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        files = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime) if pattern else []
        for stale in files[: max(0, len(files) - (cap - 1))]:
            stale.unlink(missing_ok=True)
    return cache_dir / name

def _write_cache_file(what: str, subdir: str, legacy: str, name: str, pattern: str, cap: int,
                      write: Callable[[Any], None]) -> Optional[str]:
    """Bounded cache write via ``write(path)``; the path, or None on any failure — an unwritable cache must never
    break control (a capture keeps working without its spill/screenshot copy)."""
    try:
        write(path := _cache_file(subdir, legacy, name, pattern, cap))
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: %s failed: %s", what, exc)
        return None

def _persist_capture_image(cap: CaptureResult) -> Optional[str]:
    """Copy of the capture in Hermes' media cache so attachment surfaces can deliver it (None without an image)."""
    return _write_cache_file(
        "screenshot persistence", "cache/images", "image_cache", f"computer_use_{uuid.uuid4().hex}{_capture_image_format(cap)[1]}",
        "computer_use_*.*", _MAX_CAPTURE_FILES, lambda p: p.write_bytes(base64.b64decode(cap.png_b64, validate=False)),
    ) if cap.png_b64 else None

def _spill_elements_to_file(cap: CaptureResult) -> Optional[str]:
    """FULL element tree (untruncated labels) in a cache file — the read_file/search_files escape hatch for capped text."""
    payload = {"app": cap.app, "window_title": cap.window_title, "total_elements": len(cap.elements),
               "elements": [{"index": e.index, "role": e.role, "label": e.label, "bounds": list(e.bounds), "app": e.app}
                            for e in cap.elements]}
    return _write_cache_file("element spill", "cache/computer_use", "computer_use_cache", f"elements_{uuid.uuid4().hex}.json",
                             "elements_*.json", _MAX_SPILL_FILES,
                             lambda p: p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"))

# ── auxiliary.vision routing for captured screenshots ───────────────────────
_MAX_VISION_DIM = 1456  # longest image side handed to the aux vision model: full-resolution desktop captures tokenize heavily
# and can overflow small local-model context windows; ~1456px keeps SOM badges legible while cutting vision latency.

def _shrink_capture_for_vision(raw: bytes, ext: str, max_dim: int = _MAX_VISION_DIM) -> tuple[bytes, Optional[str]]:
    """Downscale encoded image bytes so the longest side is <= max_dim -> ``(bytes, scale_note)``. note is None when
    unchanged (fits, or Pillow unavailable/failed), else it tells the vision model the factor so reported
    coordinates map back to the real screen instead of being silently wrong."""
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(raw))
        if max(img.size) <= max_dim:
            return raw, None
        (orig_w, orig_h), out = img.size, BytesIO()
        img.thumbnail((max_dim, max_dim))
        new_w, new_h = img.size
        img.save(out, format="JPEG" if ext == ".jpg" else "PNG")
        fx, fy = (orig_w / new_w if new_w else 1.0), (orig_h / new_h if new_h else 1.0)
        factor_clause = (f"multiply any coordinates you report by {fx:.2f} to map back to the real screen." if f"{fx:.2f}" == f"{fy:.2f}"
                         else f"multiply any x coordinates you report by {fx:.2f} and any y coordinates by {fy:.2f} to map back to the real screen.")
        return out.getvalue(), f"Screenshot downscaled from {orig_w}x{orig_h} to {new_w}x{new_h} for vision; {factor_clause}"
    except Exception as exc:
        logger.debug("computer_use: vision downscale skipped: %s", exc)
        return raw, None

def _should_route_through_aux_vision() -> bool:
    """True when ``_capture_response`` should hand the PNG to aux vision. Any failure returns False (fail open) so a
    broken config never silently drops the screenshot for vision-capable main models."""
    stage = "import"
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from hermes_cli.config import load_config
        from hermes_constants import hermes_home_key
        from tools.computer_use.vision_routing import should_route_capture_to_aux_vision
        stage = "config read"
        provider, model = _read_main_provider() or "", _read_main_model() or ""
        if (cached := _AUX_VISION_ROUTE_CACHE.get(key := (hermes_home_key(), str(provider), str(model)))) is not None:
            return cached
        stage = "decision"
        _AUX_VISION_ROUTE_CACHE[key] = decision = bool(should_route_capture_to_aux_vision(provider, model, load_config()))
        return decision
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing %s failed: %s", stage, exc)
        return False

def _capture_after_mode() -> str:
    """Mode for ``capture_after`` follow-ups. Default ``som`` (screenshot)."""
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        mode = str(((load_config() or {}).get("computer_use") or {}).get("capture_after_mode", "som") or "som")
        return mode if (mode := mode.strip().lower()) in {"som", "vision", "ax"} else "som"
    return "som"

_VISION_PROMPT = ("Describe what is visible in this desktop application screenshot in concise but specific terms. Mention "
                  "the app name and window title if visible, the overall layout, any labelled buttons, menus or text fields, "
                  "and any prominent text content the user would need to know about. Do not invent details that are not "
                  "actually visible.\n\nAX/SOM index for cross-reference:\n")


def _route_capture_through_aux_vision(
    cap: CaptureResult,
    summary: str,
    *,
    visible_elements: Optional[List[UIElement]] = None,
    truncated_elements: int = 0,
    elements_file: Optional[str] = None,
    screenshot_path: Optional[str] = None,
) -> Optional[str]:
    """Pre-analyse the captured PNG via ``vision_analyze`` and return a text result.

    The captured base64 PNG is materialised to ``$HERMES_HOME/cache/vision/``
    and handed to ``vision_analyze_tool`` with a generic describe prompt.
    The resulting text description is merged into the existing AX/SOM
    summary so the main model receives a single text payload that mentions
    every interactable element AND a description of what the screenshot
    looked like.

    Returns:
      A JSON-encoded text response on success.
      ``None`` on failure (caller falls back to the multimodal envelope).
    """
    if not cap.png_b64:
        return None
    problem, temp_image_path = "aux-vision import failed", None
    try:
        from model_tools import _run_async
        from tools.vision_tools import vision_analyze_tool
        problem = "failed to decode capture base64"
        raw = base64.b64decode(cap.png_b64, validate=False)
        problem = None  # from here on failures are loud (warning)
        ext = _capture_image_format(cap)[1]
        temp_image_path = _cache_file("cache/vision", "temp_vision_images", f"computer_use_{uuid.uuid4().hex}{ext}")
        raw, scale_note = _shrink_capture_for_vision(raw, ext)
        temp_image_path.write_bytes(raw)
        prompt = _VISION_PROMPT + summary + (f"\n\nNote: {scale_note}" if scale_note else "")
        result_json = _run_async(vision_analyze_tool(str(temp_image_path), prompt))
    except Exception as exc:
        if problem:
            logger.debug("computer_use: %s: %s", problem, exc)
        else:
            logger.warning("computer_use: auxiliary.vision pre-analysis failed (%s); "
                           "returning to caller without aux analysis", exc)
        return None
    finally:
        if temp_image_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(str(temp_image_path))
    # The ``analysis`` field of vision_analyze_tool's JSON result; raw text when it isn't JSON; empty -> no merge.
    analysis_text = result_json.strip() if isinstance(result_json, str) else ""
    with contextlib.suppress(TypeError, json.JSONDecodeError):
        parsed = json.loads(analysis_text)
        analysis_text = str(parsed.get("analysis") or "").strip() if isinstance(parsed, dict) else ""
    if not analysis_text:
        return None
    # Same element cap as every other capture branch; dumping cap.elements in full would bypass max_elements
    # exactly for non-vision main models. Dimensions are the backend's on this branch.
    view = SimpleNamespace(cap=cap, visible=cap.elements if visible_elements is None else visible_elements,
                           total=len(cap.elements), width=cap.width, height=cap.height, truncated=truncated_elements,
                           elements_file=elements_file, screenshot_path=screenshot_path, bounds_scale=None)
    return _text_capture_payload(view, summary, {"vision_analysis": analysis_text,
                                                 "vision_analysis_routed_via": "auxiliary.vision"})

    # Respect the same element cap as every other capture branch. Before this,
    # the aux-vision path dumped cap.elements in full — silently bypassing
    # max_elements exactly when a non-vision main model was configured, so a
    # dense Electron UI (Discord, Slack, IDEs) could blow the response budget
    # on this branch alone.
    elements_out = cap.elements if visible_elements is None else visible_elements
    payload: Dict[str, Any] = {
        "mode": cap.mode,
        "width": cap.width,
        "height": cap.height,
        "app": cap.app,
        "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in elements_out],
        "total_elements": len(cap.elements),
        "summary": summary,
        "vision_analysis": analysis_text,
        "vision_analysis_routed_via": "auxiliary.vision",
    }
    if truncated_elements:
        payload["truncated_elements"] = truncated_elements
    if elements_file:
        payload["elements_file"] = elements_file
    if screenshot_path:
        payload["screenshot_path"] = screenshot_path
    return json.dumps(payload)


def _maybe_follow_capture(
    backend: ComputerUseBackend, res: ActionResult, do_capture: bool,
) -> Any:
    if not do_capture:
        return _text_response(res)
    # Skip the follow-up capture when the action itself failed: showing a
    # normal-looking screenshot after a failure misleads the model into thinking
    # the action succeeded. Return the error text instead.
    if not res.ok:
        return _text_response(res)
    try:
        # Preserve the exact selected window when possible. Linux may expose a
        # generic app name for several unrelated windows, so app-only recapture
        # can silently switch targets after a successful action.
        target = getattr(backend, "_last_target", None) or {}
        pid = target.get("pid")
        window_id = target.get("window_id")
        mode = _capture_after_mode()
        if pid is not None and window_id is not None:
            cap = backend.capture(mode=mode, pid=pid, window_id=window_id)
        else:
            cap = backend.capture(mode=mode, app=getattr(backend, "_last_app", None))
    except Exception as e:
        logger.warning("follow-up capture failed: %s", e)
        return _text_response(res)
    # Combine action summary with the capture.
    resp = _capture_response(cap)
    if isinstance(resp, dict) and resp.get("_multimodal"):
        # Keep the complete evidence/verdict contract visible when an image is
        # attached; otherwise capture_after would accidentally discard the
        # very signal that governs whether repeating input is allowed.
        prefix = json.dumps(_action_payload(res))
        resp["content"][0]["text"] = prefix + "\n\n" + resp["content"][0]["text"]
        resp["text_summary"] = prefix + "\n\n" + resp["text_summary"]
        resp["action_result"] = _action_payload(res)
        return resp
    # Fallback: action + text capture merged.
    try:
        data = json.loads(resp)
    except (TypeError, json.JSONDecodeError):
        data = {"capture": resp}
    data.update(_action_payload(res))
    return json.dumps(data)


def _bounds_unknown(bounds) -> bool:
    """True when the AX tree reported no real geometry for an element.

    KDE/Qt apps commonly report ``[0, 0, 0, 0]`` for elements that are
    perfectly clickable by index (live QA, Aug 2026: all of kcalc's radio
    buttons). Serializing that as a plausible-looking rect invites a model
    to derive ``coordinate=[0, 0]`` from it and click the screen corner.
    """
    try:
        return all(int(v) == 0 for v in bounds)
    except (TypeError, ValueError):
        return False


def _format_elements(elements: List[UIElement], max_lines: int = 40) -> List[str]:
    out: List[str] = []
    for e in elements[:max_lines]:
        label = e.label.replace("\n", " ")[:60]
        where = "@ bounds-unknown (click by element index)" if _bounds_unknown(e.bounds) else f"@ {e.bounds}"
        out.append(f"  #{e.index} {e.role} {label!r} {where}"
                   + (f" [{e.app}]" if e.app else ""))
    if len(elements) > max_lines:
        out.append(f"  ... +{len(elements) - max_lines} more (call capture with app= to narrow)")
    return out


# Element labels come straight from the platform accessibility tree, which on
# some apps (Discord/Slack via UIA, Electron chat clients generally) exposes
# ENTIRE message bodies / document text as the accessible name of a node.
# 100 elements x multi-KB labels made single capture responses exceed 170KB —
# blowing the tool-result budget so the model never saw the elements it needed,
# and leaking full private chat text into context. The summary line has always
# truncated to 60 chars; this applies a (more generous) cap to the JSON
# `elements` array too. Labels are for identifying a control, not for reading
# page content — captures are not a text-extraction surface.
_MAX_ELEMENT_LABEL_CHARS = 120

# Keep at most this many spilled element-tree files in the cache dir. Each
# capture of a dense UI can spill; without pruning the cache grows unbounded.
_MAX_SPILL_FILES = 20

# Keep user-shareable capture files bounded independently from the gateway's
# periodic media-cache cleanup. CLI-only sessions may never start the gateway,
# and capture_after can otherwise leave an unbounded screenshot trail.
_MAX_CAPTURE_FILES = 20


def _persist_capture_image(cap: CaptureResult) -> Optional[str]:
    """Save a capture in Hermes' media cache and return its absolute path.

    Captures are normally embedded only in the model's tool context. Persisting
    a bounded copy gives attachment-capable surfaces a real file to deliver
    when the user explicitly asks for the screenshot. This is best-effort: an
    unwritable cache must never break computer control.
    """
    if not cap.png_b64:
        return None
    try:
        import uuid as _uuid

        from hermes_constants import get_hermes_dir

        raw = base64.b64decode(cap.png_b64, validate=False)
        mime = str(cap.image_mime_type or "").lower()
        ext = ".jpg" if mime == "image/jpeg" or (
            not mime and cap.png_b64[:8].startswith("/9j/")
        ) else ".png"

        cache_dir = get_hermes_dir("cache/images", "image_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            captures = sorted(
                cache_dir.glob("computer_use_*.*"),
                key=lambda path: path.stat().st_mtime,
            )
            keep_before_write = max(0, _MAX_CAPTURE_FILES - 1)
            for stale in captures[: max(0, len(captures) - keep_before_write)]:
                stale.unlink(missing_ok=True)
        except Exception:
            pass

        path = cache_dir / f"computer_use_{_uuid.uuid4().hex}{ext}"
        path.write_bytes(raw)
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: screenshot persistence failed: %s", exc)
        return None


def _spill_elements_to_file(cap: CaptureResult) -> Optional[str]:
    """Write the FULL element tree (untruncated labels) to a cache file.

    The in-context response caps labels at ``_MAX_ELEMENT_LABEL_CHARS`` and
    the array at ``max_elements`` to protect the tool-result budget, but the
    dropped text is sometimes exactly what the task needs (reading a chat
    transcript or document text exposed through the AX tree). Spilling the
    complete tree to disk gives the model an escape hatch — read_file /
    search_files against the returned path — without paying the full tree
    into context on every capture.

    Returns the absolute path, or None on any failure (spilling is an
    enhancement; a capture must never fail because the cache dir is
    unwritable).
    """
    try:
        import uuid as _uuid

        from hermes_constants import get_hermes_dir

        cache_dir = get_hermes_dir("cache/computer_use", "computer_use_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Prune oldest spills beyond the cap (best-effort).
        try:
            spills = sorted(
                cache_dir.glob("elements_*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            for stale in spills[: max(0, len(spills) - (_MAX_SPILL_FILES - 1))]:
                stale.unlink(missing_ok=True)
        except Exception:
            pass
        path = cache_dir / f"elements_{_uuid.uuid4().hex}.json"
        payload = {
            "app": cap.app,
            "window_title": cap.window_title,
            "total_elements": len(cap.elements),
            "elements": [
                {
                    "index": e.index,
                    "role": e.role,
                    "label": e.label,  # full, untruncated
                    "bounds": list(e.bounds),
                    "app": e.app,
                }
                for e in cap.elements
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: element spill failed: %s", exc)
        return None


def _capture_lost_detail(
    cap: CaptureResult, visible_elements: List[UIElement], truncated_elements: int,
) -> bool:
    """True when the in-context response drops information the full tree has."""
    if truncated_elements:
        return True
    return any(
        len(e.label) > _MAX_ELEMENT_LABEL_CHARS for e in visible_elements
    )


def _bounds_scale(
    elements: List[UIElement], image_width: int, image_height: int,
) -> Optional[float]:
    """Estimated native-bounds → screenshot-pixel scale factor, or None.

    Only meaningful when the two spaces diverge (same condition as
    ``_bounds_space_note``). Uses the larger of the two axis ratios so the
    estimate is driven by the axis with real extent data. Rounded to 2
    decimals — this is a heuristic for mapping screenshot positions to
    native coordinates, not display-metrics ground truth.
    """
    if not elements or image_width <= 0 or image_height <= 0:
        return None
    max_x = 0
    max_y = 0
    for e in elements:
        try:
            x, y, w, h = e.bounds
        except (TypeError, ValueError):
            continue
        max_x = max(max_x, int(x) + int(w))
        max_y = max(max_y, int(y) + int(h))
    if max_x <= image_width * 1.05 and max_y <= image_height * 1.05:
        return None
    return round(max(max_x / image_width, max_y / image_height), 2)


def _bounds_space_note(
    elements: List[UIElement], image_width: int, image_height: int,
) -> Optional[str]:
    """Warn when element bounds live in a different coordinate space.

    On HiDPI/scaled displays (common on Windows + macOS retina), cua-driver
    reports AX element bounds in native desktop coordinates while the
    screenshot is captured/downscaled to a smaller pixel grid. Nothing in the
    response related the two, so models reading a position off the screenshot
    and clicking by coordinate= missed by the scale factor (e.g. 2.6x on a
    4K display with a 1455px-wide screenshot). Element bounds are what
    click(coordinate=...) expects; the note makes that explicit whenever the
    two spaces visibly diverge.
    """
    if not elements or image_width <= 0 or image_height <= 0:
        return None
    max_x = 0
    max_y = 0
    for e in elements:
        try:
            x, y, w, h = e.bounds
        except (TypeError, ValueError):
            continue
        max_x = max(max_x, int(x) + int(w))
        max_y = max(max_y, int(y) + int(h))
    if max_x <= 0 and max_y <= 0:
        return None
    # 5% slack: window chrome can hang a few px past the captured frame
    # without implying a different coordinate space.
    if max_x <= image_width * 1.05 and max_y <= image_height * 1.05:
        return None
    return (
        f"element bounds are in native desktop coordinates (extend to "
        f"~{max_x}x{max_y}), NOT screenshot pixels ({image_width}x"
        f"{image_height}). coordinate= clicks expect the native space — "
        "derive click points from element bounds, or scale screenshot "
        "positions up accordingly"
    )


def _element_to_dict(e: UIElement) -> Dict[str, Any]:
    label = e.label
    truncated = len(label) > _MAX_ELEMENT_LABEL_CHARS
    if truncated:
        label = label[:_MAX_ELEMENT_LABEL_CHARS]
    out: Dict[str, Any] = {
        "index": e.index,
        "role": e.role,
        "label": label,
        # A zero rect is "geometry unknown", not a position — null it so no
        # coordinate= is ever derived from it. The element index still works.
        "bounds": None if _bounds_unknown(e.bounds) else list(e.bounds),
        "app": e.app,
    }
    if truncated:
        out["label_truncated"] = True
    return out


# ---------------------------------------------------------------------------
# Availability check (used by the tool registry check_fn)
# ---------------------------------------------------------------------------

def check_computer_use_requirements() -> bool:
    """macOS/Windows/Linux + cua-driver binary (or env override). `hermes computer-use doctor` names blocked checks."""
    if sys.platform not in ("darwin", "win32", "linux"):
        return False
    from tools.computer_use.cua_backend_driver import cua_driver_binary_available
    return cua_driver_binary_available()

def get_computer_use_schema() -> Dict[str, Any]:
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA
    return COMPUTER_USE_SCHEMA


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import struct  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
