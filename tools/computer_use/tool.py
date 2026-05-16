"""Entry point for the `computer_use` tool.

Universal (any-model) macOS desktop control via cua-driver's background
computer-use primitive. Replaces #4562's Anthropic-native `computer_20251124`
approach — the schema here is standard OpenAI function-calling so every
tool-capable model can drive it.

Return contract
---------------
For text-only results (wait, key, list_apps, focus_app, failures, etc.):
  JSON string.

For captures / actions with `capture_after=True`:
  A dict wrapped as the OpenAI-style multi-part tool-message content:

      {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "<human-readable summary + SOM index>"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,<b64>"}},
        ],
        "text_summary": "<text used for fallback string content>",
      }

  run_agent.py's tool-message builder inspects `_multimodal` and emits a
  list-shaped `content` for OpenAI-compatible providers. The Anthropic
  adapter splices the base64 image into a `tool_result` block (see
  `agent/anthropic_adapter.py`). Every provider that supports multi-part
  tool content gets the image; text-only providers see the summary only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approval & safety
# ---------------------------------------------------------------------------

_approval_callback = None


def set_approval_callback(cb) -> None:
    """Register a callback for computer_use approval prompts (used by CLI).

    Matches the terminal_tool._approval_callback pattern. The callback
    receives (action, args, summary) and returns one of:
      "approve_once" | "approve_session" | "always_approve" | "deny".
    """
    global _approval_callback
    _approval_callback = cb


# Actions that read, not mutate. Always allowed.
_SAFE_ACTIONS = frozenset({"capture", "wait", "list_apps"})

# Actions that mutate user-visible state. Go through approval.
_DESTRUCTIVE_ACTIONS = frozenset({
    "click", "double_click", "right_click", "middle_click",
    "drag", "scroll", "type", "key", "set_value", "focus_app",
})

# Hard-blocked key combinations. Mirrored from #4562 — these are destructive
# regardless of approval level (e.g. logout kills the session Hermes runs in).
_BLOCKED_KEY_COMBOS = {
    frozenset({"cmd", "shift", "backspace"}),   # empty trash
    frozenset({"cmd", "option", "backspace"}),   # force delete
    frozenset({"cmd", "ctrl", "q"}),             # lock screen
    frozenset({"cmd", "shift", "q"}),            # log out
    frozenset({"cmd", "option", "shift", "q"}),  # force log out
}

_KEY_ALIASES = {"command": "cmd", "control": "ctrl", "alt": "option", "⌘": "cmd", "⌥": "option"}


def _canon_key_combo(keys: str) -> frozenset:
    parts = [p.strip().lower() for p in re.split(r"\s*\+\s*", keys) if p.strip()]
    parts = [_KEY_ALIASES.get(p, p) for p in parts]
    return frozenset(parts)


# Dangerous text patterns for the `type` action. Same list as #4562.
_BLOCKED_TYPE_PATTERNS = [
    re.compile(r"curl\s+[^|]*\|\s*bash", re.IGNORECASE),
    re.compile(r"curl\s+[^|]*\|\s*sh", re.IGNORECASE),
    re.compile(r"wget\s+[^|]*\|\s*bash", re.IGNORECASE),
    re.compile(r"\bsudo\s+rm\s+-[rf]", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}", re.IGNORECASE),  # fork bomb
]


def _is_blocked_type(text: str) -> Optional[str]:
    for pat in _BLOCKED_TYPE_PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None


# ---------------------------------------------------------------------------
# Backend selection — env-swappable for tests
# ---------------------------------------------------------------------------

# Per-process cached backend; lazily instantiated on first call.
_backend_lock = threading.Lock()
_backend: Optional[ComputerUseBackend] = None
# Session-scoped approval state.
_session_auto_approve = False
_always_allow: set = set()  # action names the user unlocked for the session


def _get_backend() -> ComputerUseBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            backend_name = os.environ.get("HERMES_COMPUTER_USE_BACKEND", "cua").lower()
            if backend_name in {"cua", "cua-driver", ""}:
                from tools.computer_use.cua_backend import CuaDriverBackend
                _backend = CuaDriverBackend()
            elif backend_name == "noop":  # pragma: no cover
                _backend = _NoopBackend()
            else:
                raise RuntimeError(f"Unknown HERMES_COMPUTER_USE_BACKEND={backend_name!r}")
            _backend.start()
        return _backend


def reset_backend_for_tests() -> None:  # pragma: no cover
    """Test helper — tear down the cached backend."""
    global _backend, _session_auto_approve, _always_allow
    with _backend_lock:
        if _backend is not None:
            try:
                _backend.stop()
            except Exception:
                pass
        _backend = None
    _session_auto_approve = False
    _always_allow = set()


class _NoopBackend(ComputerUseBackend):  # pragma: no cover
    """Test/CI stub. Records calls; returns trivial results."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._started = False

    def start(self) -> None: self._started = True
    def stop(self) -> None: self._started = False
    def is_available(self) -> bool: return True

    def capture(self, mode: str = "som", app: Optional[str] = None) -> CaptureResult:
        self.calls.append(("capture", {"mode": mode, "app": app}))
        return CaptureResult(mode=mode, width=1024, height=768, png_b64=None,
                             elements=[], app=app or "", window_title="")

    def click(self, **kw) -> ActionResult:
        self.calls.append(("click", kw))
        return ActionResult(ok=True, action="click")

    def drag(self, **kw) -> ActionResult:
        self.calls.append(("drag", kw))
        return ActionResult(ok=True, action="drag")

    def scroll(self, **kw) -> ActionResult:
        self.calls.append(("scroll", kw))
        return ActionResult(ok=True, action="scroll")

    def type_text(self, text: str) -> ActionResult:
        self.calls.append(("type", {"text": text}))
        return ActionResult(ok=True, action="type")

    def key(self, keys: str) -> ActionResult:
        self.calls.append(("key", {"keys": keys}))
        return ActionResult(ok=True, action="key")

    def list_apps(self) -> List[Dict[str, Any]]:
        self.calls.append(("list_apps", {}))
        return []

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        self.calls.append(("focus_app", {"app": app, "raise": raise_window}))
        return ActionResult(ok=True, action="focus_app")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def handle_computer_use(args: Dict[str, Any], **kwargs) -> Any:
    """Main entry point — dispatched by tools.registry.

    Returns either a JSON string (text-only) or a dict marked `_multimodal`
    (image + summary) which run_agent.py wraps into the tool message.
    """
    action = (args.get("action") or "").strip().lower()
    if not action:
        return json.dumps({"error": "missing `action`"})

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

    # Approval gate (destructive actions only).
    if action in _DESTRUCTIVE_ACTIONS:
        err = _request_approval(action, args)
        if err is not None:
            return err

    # Dispatch to backend.
    try:
        backend = _get_backend()
    except Exception as e:
        return json.dumps({
            "error": f"computer_use backend unavailable: {e}",
            "hint": "Run `hermes tools` and enable Computer Use to install cua-driver.",
        })

    try:
        return _dispatch(backend, action, args)
    except Exception as e:
        logger.exception("computer_use %s failed", action)
        return json.dumps({"error": f"{action} failed: {e}"})


def _request_approval(action: str, args: Dict[str, Any]) -> Optional[str]:
    """Return None if approved, or a JSON error string if denied."""
    global _session_auto_approve, _always_allow
    if _session_auto_approve:
        return None
    if action in _always_allow:
        return None
    cb = _approval_callback
    if cb is None:
        # No CLI approval wired — default allow. Gateway approval is handled
        # one layer out via the normal tool-approval infra.
        return None
    summary = _summarize_action(action, args)
    try:
        verdict = cb(action, args, summary)
    except Exception as e:
        logger.warning("approval callback failed: %s", e)
        verdict = "deny"
    if verdict == "approve_once":
        return None
    if verdict == "approve_session" or verdict == "always_approve":
        _always_allow.add(action)
        if verdict == "always_approve":
            _session_auto_approve = True
        return None
    return json.dumps({"error": "denied by user", "action": action})


def _summarize_action(action: str, args: Dict[str, Any]) -> str:
    if action in {"click", "double_click", "right_click", "middle_click"}:
        if args.get("element") is not None:
            return f"{action} element #{args['element']}"
        coord = args.get("coordinate")
        if coord:
            return f"{action} at {tuple(coord)}"
        return action
    if action == "drag":
        src = args.get("from_element") or args.get("from_coordinate")
        dst = args.get("to_element") or args.get("to_coordinate")
        return f"drag {src} → {dst}"
    if action == "scroll":
        return f"scroll {args.get('direction', '?')} x{args.get('amount', 3)}"
    if action == "type":
        text = args.get("text", "")
        return f"type {text[:60]!r}" + ("..." if len(text) > 60 else "")
    if action == "key":
        return f"key {args.get('keys', '')!r}"
    if action == "focus_app":
        return f"focus {args.get('app', '')!r}" + (" (raise)" if args.get("raise_window") else "")
    return action


def _dispatch(backend: ComputerUseBackend, action: str, args: Dict[str, Any]) -> Any:
    capture_after = bool(args.get("capture_after"))

    if action == "capture":
        mode = str(args.get("mode", "som"))
        if mode not in {"som", "vision", "ax"}:
            return json.dumps({"error": f"bad mode {mode!r}; use som|vision|ax"})
        cap = backend.capture(mode=mode, app=args.get("app"))
        return _capture_response(cap)

    if action == "wait":
        seconds = float(args.get("seconds", 1.0))
        res = backend.wait(seconds)
        return _text_response(res)

    if action == "list_apps":
        apps = backend.list_apps()
        return json.dumps({"apps": apps, "count": len(apps)})

    if action == "focus_app":
        app = args.get("app")
        if not app:
            return json.dumps({"error": "focus_app requires `app`"})
        res = backend.focus_app(app, raise_window=bool(args.get("raise_window")))
        return _maybe_follow_capture(backend, res, capture_after)

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
        )
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "drag":
        res = backend.drag(
            from_element=args.get("from_element"),
            to_element=args.get("to_element"),
            from_xy=tuple(args["from_coordinate"]) if args.get("from_coordinate") else None,
            to_xy=tuple(args["to_coordinate"]) if args.get("to_coordinate") else None,
            button=args.get("button", "left"),
            modifiers=args.get("modifiers"),
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
        )
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "type":
        res = backend.type_text(args.get("text", ""))
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "key":
        res = backend.key(args.get("keys", ""))
        return _maybe_follow_capture(backend, res, capture_after)

    if action == "set_value":
        value = args.get("value")
        if value is None:
            return json.dumps({"error": "set_value requires `value`"})
        res = backend.set_value(value=str(value), element=args.get("element"))
        return _maybe_follow_capture(backend, res, capture_after)

    return json.dumps({"error": f"unknown action {action!r}"})


# ---------------------------------------------------------------------------
# Response shaping
# ---------------------------------------------------------------------------

def _text_response(res: ActionResult) -> str:
    payload: Dict[str, Any] = {"ok": res.ok, "action": res.action}
    if res.message:
        payload["message"] = res.message
    if res.meta:
        payload["meta"] = res.meta
    return json.dumps(payload)


def _capture_response(cap: CaptureResult) -> Any:
    element_index = _format_elements(cap.elements)
    summary_lines = [
        f"capture mode={cap.mode} {cap.width}x{cap.height}"
        + (f" app={cap.app}" if cap.app else "")
        + (f" window={cap.window_title!r}" if cap.window_title else ""),
        f"{len(cap.elements)} interactable element(s):",
    ]
    if element_index:
        summary_lines.extend(element_index)
    summary = "\n".join(summary_lines)

    if cap.png_b64 and cap.mode != "ax":
        # Detect actual image format from base64 magic bytes so the MIME type
        # matches what the data contains (cua-driver may return JPEG or PNG).
        # JPEG: base64 starts with /9j/   PNG: starts with iVBOR
        _b64_prefix = cap.png_b64[:8]
        _mime = "image/jpeg" if _b64_prefix.startswith("/9j/") else "image/png"
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": summary},
                {"type": "image_url",
                 "image_url": {"url": f"data:{_mime};base64,{cap.png_b64}"}},
            ],
            "text_summary": summary,
            "meta": {"mode": cap.mode, "width": cap.width, "height": cap.height,
                     "elements": len(cap.elements), "png_bytes": cap.png_bytes_len},
        }
    # AX-only (or image missing): text path.
    return json.dumps({
        "mode": cap.mode,
        "width": cap.width,
        "height": cap.height,
        "app": cap.app,
        "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in cap.elements],
        "summary": summary,
    })


def _maybe_follow_capture(
    backend: ComputerUseBackend, res: ActionResult, do_capture: bool,
) -> Any:
    if not do_capture:
        return _text_response(res)
    try:
        cap = backend.capture(mode="som")
    except Exception as e:
        logger.warning("follow-up capture failed: %s", e)
        return _text_response(res)
    # Combine action summary with the capture.
    resp = _capture_response(cap)
    if isinstance(resp, dict) and resp.get("_multimodal"):
        prefix = f"[{res.action}] ok={res.ok}" + (f" — {res.message}" if res.message else "")
        resp["content"][0]["text"] = prefix + "\n\n" + resp["content"][0]["text"]
        resp["text_summary"] = prefix + "\n\n" + resp["text_summary"]
        return resp
    # Fallback: action + text capture merged.
    try:
        data = json.loads(resp)
    except (TypeError, json.JSONDecodeError):
        data = {"capture": resp}
    data["action"] = res.action
    data["ok"] = res.ok
    if res.message:
        data["message"] = res.message
    return json.dumps(data)


def _format_elements(elements: List[UIElement], max_lines: int = 40) -> List[str]:
    out: List[str] = []
    for e in elements[:max_lines]:
        label = e.label.replace("\n", " ")[:60]
        out.append(f"  #{e.index} {e.role} {label!r} @ {e.bounds}"
                   + (f" [{e.app}]" if e.app else ""))
    if len(elements) > max_lines:
        out.append(f"  ... +{len(elements) - max_lines} more (call capture with app= to narrow)")
    return out


def _element_to_dict(e: UIElement) -> Dict[str, Any]:
    return {
        "index": e.index,
        "role": e.role,
        "label": e.label,
        "bounds": list(e.bounds),
        "app": e.app,
    }


# ---------------------------------------------------------------------------
# Availability check (used by the tool registry check_fn)
# ---------------------------------------------------------------------------

def check_computer_use_requirements() -> bool:
    """Return True iff computer_use can run on this host.

    Conditions: macOS + cua-driver binary installed (or override via env).
    """
    if sys.platform != "darwin":
        return False
    from tools.computer_use.cua_backend import cua_driver_binary_available
    return cua_driver_binary_available()


def get_computer_use_schema() -> Dict[str, Any]:
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA
    return COMPUTER_USE_SCHEMA
