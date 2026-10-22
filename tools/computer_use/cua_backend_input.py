"""Input side of the cua-driver backend: delivery-mode handling and the pointer / keyboard /
value-setter methods (mixed into ``CuaDriverBackend``)."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_parse import _parse_key_combo

_NO_TARGET_MSG = "No active window — call capture() first."
_WINDOWS = sys.platform == "win32"
_IL_NAMES = {0x0000: "Untrusted", 0x1000: "Low", 0x2000: "Medium", 0x3000: "High", 0x4000: "System",
             0x5000: "Protected"}
_IL_BLOCKED_PREFIX = "Target window's process is unreachable by synthetic input on Windows: "
_IL_BLOCKED_SUFFIX = (" Windows UIPI would silently discard the input while the driver still reports an "
                      "unverifiable success. No input was sent; capture still works, but input cannot "
                      "reach this window.")
_BTF_UNSUPPORTED_MSG = "The connected cua-driver does not advertise the standalone bring_to_front tool."
_FOREGROUND_UNSUPPORTED_MSG = ("The connected cua-driver action schema does not accept delivery_mode, so foreground "
                               "delivery is unavailable. Use another verified rung without assuming the reported "
                               "package version describes the live schema.")
# (what, extra args) pointer addressing form; ``extra`` is None when the caller did not supply that form and
# may be a callable when computing it has side effects (capability probes) that must follow the refusal checks.
_Variant = tuple[str, None | dict[str, Any] | Callable[[], dict[str, Any]]]

def _refuse(action: str, message: str, **fields: Any) -> ActionResult:
    return ActionResult(ok=False, action=action, message=message, **fields)


class _IntegrityRids(NamedTuple):
    """Outcome of the read-only integrity probe, with the ambiguity made explicit."""
    own: Optional[int]      # mandatory label of this process, None when unreadable
    target: Optional[int]   # mandatory label of the target, None when unreadable
    probed: bool            # False when the target process could not be opened at all


def _il_name(rid: int) -> str:
    return f"0x{rid:x} ({_IL_NAMES.get(rid, 'unknown')})"


def _il_decision(own_rid: int | None, target_rid: int | None) -> str | None:
    """Refusal reason when the integrity comparison proves synthetic input cannot reach the target,
    else ``None``. A readable target RID is a determinate comparison — a caller holding
    ``SeDebugPrivilege`` can open even a System token, so token readability alone is not the verdict.
    A target token that cannot be opened while ours can stays a conservative block; an unreadable own
    token means the probe itself is permission-blind and must not refuse
    (NousResearch/hermes-agent#142661)."""
    if own_rid is None:
        return None
    if target_rid is None:
        return "the target's process token cannot be opened by this process (protected or a different " \
               "security context)"
    if target_rid > own_rid:
        return f"the target process runs at integrity {_il_name(target_rid)}, above this process at " \
               f"{_il_name(own_rid)}"
    return None


def _win_integrity_rids(pid: int) -> "_IntegrityRids":
    """Read ``(own RID, target RID)`` for the integrity comparison. Read-only; never raises.

    Split out from :func:`_win_input_blocked_by_integrity` so a test can assert the probe really parses
    a RID. A probe that silently answers ``None`` for both sides makes every verdict "not blocked", and
    the wiring tests — which stub this level — cannot see that.
    """
    if not _WINDOWS:
        return _IntegrityRids(None, None, False)
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        TOKEN_QUERY, TokenIntegrityLevel, PROCESS_QUERY_LIMITED_INFORMATION = 0x0008, 25, 0x1000
        # Handles are pointer-sized: without an explicit restype they truncate to c_int, and a
        # handle with the high bit set turns negative — a false "denied" verdict.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        advapi32.OpenProcessToken.restype = wintypes.BOOL
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                              ctypes.POINTER(wintypes.HANDLE)]
        advapi32.GetTokenInformation.restype = wintypes.BOOL
        advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_uint, ctypes.c_void_p,
                                                 wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
        advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
        advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
        advapi32.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]

        class _TokenLabel(ctypes.Structure):
            # TOKEN_MANDATORY_LABEL: SID_AND_ATTRIBUTES {PSID Sid; DWORD Attributes;} comes first.
            _fields_ = [("sid", ctypes.c_void_p), ("attributes", wintypes.DWORD)]

        def _il_rid(process) -> int | None:
            """``TokenIntegrityLevel`` RID — 0x1000 Low / 0x2000 Medium / 0x3000 High / 0x4000 System —
            or ``None`` when the token cannot be opened or parsed."""
            token = wintypes.HANDLE()
            if not advapi32.OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
                return None  # ACCESS_DENIED here is the higher-integrity / protected signal
            try:
                needed = wintypes.DWORD()
                # The sizing call always fails with ERROR_INSUFFICIENT_BUFFER and puts the required length
                # in `needed` — checking its return value here makes the probe answer None for every
                # caller, which silently disables the gate.
                advapi32.GetTokenInformation(token, TokenIntegrityLevel, None, 0, ctypes.byref(needed))
                if not needed.value:
                    return None
                buf = ctypes.create_string_buffer(needed.value)
                if not advapi32.GetTokenInformation(token, TokenIntegrityLevel, buf, needed.value,
                                                    ctypes.byref(needed)):
                    return None
                label = ctypes.cast(buf, ctypes.POINTER(_TokenLabel)).contents
                if not label.sid:
                    return None
                count = advapi32.GetSidSubAuthorityCount(label.sid)[0]
                if not count:
                    return None
                return advapi32.GetSidSubAuthority(label.sid, count - 1)[0]
            finally:
                kernel32.CloseHandle(token)

        own = _il_rid(kernel32.GetCurrentProcess())
        proc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not proc:
            # ERROR_INVALID_PARAMETER means the pid is gone: nothing to compare, let the driver report it.
            # Access denied means the process is unreachable even for a query handle, which is as
            # determinate as an unreadable token.
            denied = ctypes.get_last_error() == 5
            return _IntegrityRids(own, None, denied)
        try:
            return _IntegrityRids(own, _il_rid(proc), True)
        finally:
            kernel32.CloseHandle(proc)
    except Exception:
        return _IntegrityRids(None, None, False)


def _win_input_blocked_by_integrity(pid: int) -> str | None:
    """Refusal reason when *pid* is provably unreachable by synthetic input on Windows, else ``None``.
    Compares ``TokenIntegrityLevel`` RIDs of the target process and this process (determinate even
    under ``SeDebugPrivilege``, where token readability stops discriminating); a target token that
    cannot be opened while ours can stays a conservative block. Read-only probes; never raises. On
    any ctypes failure or non-Windows host returns ``None`` so callers keep today's ``unverifiable``
    behavior (NousResearch/hermes-agent#142661)."""
    rids = _win_integrity_rids(pid)
    if not rids.probed:
        return None  # process gone or unopenable for an unrelated reason: not determinate
    return _il_decision(rids.own, rids.target)


class _InputMixin:
    """Pointer / keyboard / value-setter actions against the sticky target."""

    def _target_args(self, action: str, *, need_window: bool = False) -> tuple[Optional[ActionResult], dict[str, Any]]:
        """``(refusal, base args)`` for an input action against the sticky target."""
        if self._active_pid is None or (need_window and self._active_window_id is None):
            return _refuse(action, _NO_TARGET_MSG), {}
        return None, {"pid": self._active_pid, **({"window_id": self._active_window_id} if need_window else {})}

    def _pointer_args(self, tool: str, args: dict[str, Any], variants: Sequence[_Variant],
                      missing_msg: Optional[str]) -> Optional[ActionResult]:
        """Fill *args* from the first supplied addressing variant (element or coordinates) plus ``window_id``; refuse
        when the target has a pid but no window_id yet. No variant -> refuse with *missing_msg* (None = bare window)."""
        for what, extra in variants:
            if extra is not None:
                if self._active_window_id is None:
                    return _refuse(tool, f"No active window_id for {what}.")
                args.update(extra() if callable(extra) else extra, window_id=self._active_window_id)
                return None
        return _refuse(tool, missing_msg) if missing_msg else None

    def _apply_delivery(self, action: str, args: dict[str, Any], delivery_mode: Optional[str]) -> Optional[ActionResult]:
        """Attach delivery_mode to an input-action args dict. Background is the default and needs no flag.
        Foreground is only sent when the live action schema accepts it; on an older driver we refuse with
        ``foreground_unsupported`` instead of silently downgrading to background (which would land input
        where the model didn't expect).

        Returns an ActionResult to short-circuit on refusal, or None to proceed. See
        NousResearch/hermes-agent#67052 phase B.
        """
        if not delivery_mode or delivery_mode == "background":
            return None
        if delivery_mode != "foreground":
            return _refuse(action, f"unknown delivery_mode {delivery_mode!r} — use background|foreground.",
                           code="bad_delivery_mode")
        if not self._session.supports_input_property(action, "delivery_mode"):
            return _refuse(action, _FOREGROUND_UNSUPPORTED_MSG, code="foreground_unsupported", delivery_mode="foreground")
        args["delivery_mode"] = "foreground"
        return None

    def _integrity_refusal(self, action: str) -> Optional[ActionResult]:
        """Windows UIPI pre-check: before injecting into the sticky target, refuse determinately when the
        target's process sits above this process on the integrity ladder — Windows silently discards the
        synthetic input while the verdict reads as an unverifiable success, so the verify → escalate ladder
        never terminates (NousResearch/hermes-agent#142661). Probe results are cached per pid; when the probe
        is inconclusive we proceed and keep today's semantics. ``set_value`` deliberately stays outside this
        gate (it goes through the driver's UIA path, not synthetic input) — but UIA value-sets cross UIPI
        too, so a blocked target will fail there as well, silently, just with a different result shape."""
        pid = self._active_pid
        if pid is None or not _WINDOWS:
            return None
        cache = getattr(self, "_il_probe_cache", None)
        if cache is None:
            cache = self._il_probe_cache = {}
        if pid not in cache:
            cache[pid] = _win_input_blocked_by_integrity(pid)
        reason = cache[pid]
        if reason:
            # No `escalation`: this rung has no next step, the ladder must stop here.
            return _refuse(action, _IL_BLOCKED_PREFIX + reason + "." + _IL_BLOCKED_SUFFIX,
                           code="input_blocked_il", effect="suspected_noop", verified=False)
        return None

    def _run_input_action(self, action: str, args: dict[str, Any], delivery_mode: Optional[str],
                          bring_to_front: bool) -> ActionResult:
        """Apply one delivery rung, optionally focusing via its own tool. ``bring_to_front`` is never an
        input-action property: when requested, the separately approved standalone focus action runs first,
        then the original foreground input runs unchanged."""
        refusal = self._integrity_refusal(action)
        if refusal is not None:
            return refusal
        refusal = self._apply_delivery(action, args, delivery_mode)
        if refusal is not None:
            return refusal
        if bring_to_front:
            if delivery_mode != "foreground":
                return _refuse(action, "bring_to_front requires delivery_mode='foreground'.",
                               code="bring_to_front_requires_foreground")
            if not self._session._has_tool("bring_to_front"):
                return _refuse(action, _BTF_UNSUPPORTED_MSG, code="bring_to_front_unsupported", delivery_mode="foreground")
            if self._active_pid is None or self._active_window_id is None:
                return _refuse(action, "Capture an exact target before requesting persistent foreground focus.",
                               code="bring_to_front_target_required", delivery_mode="foreground")
            focused = self.bring_to_front(pid=self._active_pid, window_id=self._active_window_id)
            if not focused.ok:
                return focused
        result = self._action(action, args)
        if bring_to_front:
            result.meta["foreground_focus"] = {"invoked": True, "tool": "bring_to_front"}
        return result

    def click(self, *, element: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None,
              button: str = "left", click_count: int = 1, modifiers: Optional[list[str]] = None,
              delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("click")
        if refusal is not None:
            return refusal
        # Tool is chosen by click_count only; `button` goes through click's enum (the driver rejects unknown
        # buttons). `right_click` / `middle_click` MCP tools are deprecated aliases and never invoked here.
        # Choose tool by click_count only — single-vs-double — and pass the button through to `click`'s
        # `button` enum (Surface 5 of NousResearch/hermes-agent#47072). cua-driver-rs gained an explicit
        # `button: "left"|"right"|"middle"` arg on `click` in trycua/cua#1961 which rejects unknown buttons;
        # before that, `middle` was silently mapped to a left-click via name-routing through `right_click`.
        button_norm = (button or "left").lower()
        if button_norm not in {"left", "right", "middle"}:
            return _refuse("click", f"unknown button {button!r} — expected left, right, middle.")
        tool, args["button"] = ("double_click" if click_count == 2 else "click"), button_norm
        refusal = self._pointer_args(tool, args, (
            ("element_index click", {"element_index": element} if element is not None else None),
            ("coordinate click", {"x": x, "y": y} if x is not None and y is not None else None),
        ), "click requires element= or x/y.")
        if modifiers:
            args["modifier"] = modifiers
        return refusal if refusal is not None else self._run_input_action(tool, args, delivery_mode, bring_to_front)

    def drag(self, *, from_element: Optional[int] = None, to_element: Optional[int] = None,
             from_xy: Optional[tuple[int, int]] = None, to_xy: Optional[tuple[int, int]] = None,
             button: str = "left", modifiers: Optional[list[str]] = None,
             delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("drag")
        if refusal is None:
            refusal = self._pointer_args("drag", args, (
                ("element-based drag", {"from_element": from_element, "to_element": to_element}
                 if from_element is not None and to_element is not None else None),
                ("coordinate drag", {"from_x": int(from_xy[0]), "from_y": int(from_xy[1]),
                                     "to_x": int(to_xy[0]), "to_y": int(to_xy[1])}
                 if from_xy is not None and to_xy is not None else None),
            ), "drag requires from_element/to_element or from_coordinate/to_coordinate.")
        return refusal if refusal is not None else self._run_input_action("drag", args, delivery_mode, bring_to_front)

    def scroll(self, *, direction: str, amount: int = 3, element: Optional[int] = None,
               x: Optional[int] = None, y: Optional[int] = None, modifiers: Optional[list[str]] = None,
               delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("scroll")
        if refusal is not None:
            return refusal
        args.update(direction=direction, amount=max(1, min(50, amount)))
        # An element without a known window_id is not an addressing form here; scrolling then falls through
        # to the coordinate form or the bare window. Some driver schemas reject x/y on scroll: only send
        # coordinates when the driver advertises support; otherwise it scrolls the targeted window
        # (window_id is still sent for routing).
        xy = lambda: ({"x": x, "y": y}
                      if self._session.supports_capability("input.scroll.coordinates", tool="scroll") else {})
        refusal = self._pointer_args("scroll", args, (
            ("element scroll", {"element_index": element}
             if element is not None and self._active_window_id is not None else None),
            ("coordinate scroll", xy if x is not None and y is not None else None),
        ), None)
        return refusal if refusal is not None else self._run_input_action("scroll", args, delivery_mode, bring_to_front)

    def type_text(self, text: str, *, delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("type_text", need_window=True)
        return refusal if refusal is not None else self._run_input_action("type_text", {**args, "text": text},
                                                                          delivery_mode, bring_to_front)

    def key(self, keys: str, *, delivery_mode: Optional[str] = None, bring_to_front: bool = False) -> ActionResult:
        refusal, args = self._target_args("key", need_window=True)
        if refusal is not None:
            return refusal
        key_name, modifiers = _parse_key_combo(keys)
        if not key_name:
            return _refuse("key", f"Could not parse key from '{keys}'.")
        if modifiers:  # hotkey requires at least one modifier + one key
            return self._run_input_action("hotkey", {**args, "keys": modifiers + [key_name]}, delivery_mode, bring_to_front)
        return self._run_input_action("press_key", {**args, "key": key_name}, delivery_mode, bring_to_front)

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        """Set a value on an element. Handles AXPopUpButton selects natively."""
        refusal, args = self._target_args("set_value", need_window=True)
        if refusal is not None:
            return refusal
        if element is None:
            return _refuse("set_value", "set_value requires element= (element index).")
        return self._action("set_value", {**args, "element_index": element, "value": value})
