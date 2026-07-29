"""Session-scoped typed-browser routing for cua-driver.

The public model surface remains the single ``computer_use`` tool.  This
module owns the stateful adapter between its namespaced ``cua_browser_*``
actions and cua-driver's raw ``get_browser_state`` / ``browser_*`` tools.

The adapter is deliberately stricter than the transport:

* native binding must be exact before mutation;
* the driver session id is injected by the adapter, never accepted from the
  model;
* refs are usable only from the latest snapshot in this Hermes session;
* every mutation invalidates refs and requires a fresh state read; and
* changing from trusted input to ``dom_event`` is always explicit.

Browser preparation remains a separate approved action.  Existing-profile
attachment is not performed here because it needs cua-driver's documented
interactive grant, not ordinary tool approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Set


ToolCaller = Callable[[str, Dict[str, Any]], Dict[str, Any]]
ToolProbe = Callable[[str], bool]


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _tool_payload(out: Dict[str, Any]) -> Dict[str, Any]:
    """Return the structured driver payload without discarding refusals."""
    structured = out.get("structuredContent")
    data = out.get("data")
    payload: Dict[str, Any] = {}
    if isinstance(data, dict):
        payload.update(data)
    elif isinstance(data, str) and data:
        payload["message"] = data
    if isinstance(structured, dict):
        payload.update(structured)
    if out.get("isError") is True:
        payload.setdefault("isError", True)
    return payload


def _ref_map(payload: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Normalize semantic-v2 action refs to ``ref -> actions``.

    cua-driver has emitted both mapping and list representations while the
    semantic snapshot contract evolved.  Accept both without weakening the
    capability rule: a ref with no declared action remains readable only.
    """
    normalized: Dict[str, Set[str]] = {}
    snapshot = payload.get("snapshot")
    # semantic_v2 carries the authoritative action-bearing entries in
    # ``content_refs``; some transitional builds also emitted a ``refs`` list
    # or map. Prefer the richer live shape, then accept both older forms.
    raw = payload.get("content_refs")
    if not raw:
        raw = payload.get("refs")
    if raw is None and isinstance(snapshot, dict):
        raw = snapshot.get("refs")
    if isinstance(raw, dict):
        entries: Iterable[tuple[Optional[str], Any]] = raw.items()
    elif isinstance(raw, list):
        entries = ((None, item) for item in raw)
    else:
        entries = ()

    for key, value in entries:
        if isinstance(value, dict):
            ref = value.get("ref") or key
            actions = value.get("actions")
        else:
            ref = key
            actions = None
        if not isinstance(ref, str) or not ref:
            continue
        normalized[ref] = {
            action for action in (actions or []) if isinstance(action, str)
        }
    return normalized


def _continuation(payload: Dict[str, Any]) -> Optional[str]:
    direct = payload.get("continuation")
    if isinstance(direct, str) and direct:
        return direct
    snapshot = payload.get("snapshot")
    if isinstance(snapshot, dict):
        nested = snapshot.get("continuation")
        if isinstance(nested, str) and nested:
            return nested
    return None


def _tab_ids(payload: Dict[str, Any]) -> Set[str]:
    result: Set[str] = set()
    for tab in payload.get("tabs") or []:
        if not isinstance(tab, dict):
            continue
        tab_id = tab.get("tab_id") or tab.get("id")
        if isinstance(tab_id, str) and tab_id:
            result.add(tab_id)
    return result


def _refusal_code(payload: Dict[str, Any]) -> Optional[str]:
    code = payload.get("code")
    if isinstance(code, str):
        return code
    refusal = payload.get("refusal")
    if isinstance(refusal, dict) and isinstance(refusal.get("code"), str):
        return refusal["code"]
    return None


def _refusal(
    code: str,
    message: str,
    *,
    native_fallback: bool = False,
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "status": "refused",
        "code": code,
        "message": message,
    }
    if native_fallback:
        payload["native_fallback_required"] = True
    payload.update(extra)
    return payload


@dataclass
class BrowserRouteState:
    """Capabilities minted for one explicit cua-driver session."""

    pid: Optional[int] = None
    window_id: Optional[int] = None
    target_id: Optional[str] = None
    tab_ids: Set[str] = field(default_factory=set)
    tab_id: Optional[str] = None
    binding_quality: Optional[str] = None
    mutation_allowed: bool = False
    refs: Dict[str, Set[str]] = field(default_factory=dict)
    continuation: Optional[str] = None
    verification_required: bool = False

    def clear_refs(self) -> None:
        self.refs.clear()
        self.continuation = None

    def clear(self) -> None:
        self.pid = None
        self.window_id = None
        self.target_id = None
        self.tab_ids.clear()
        self.tab_id = None
        self.binding_quality = None
        self.mutation_allowed = False
        self.clear_refs()
        self.verification_required = False


class CuaTypedBrowserRoute:
    """Exact-bind typed-browser adapter for a single driver session."""

    def __init__(
        self,
        *,
        session_id: str,
        call_tool: ToolCaller,
        has_tool: ToolProbe,
    ) -> None:
        self._session_id = session_id
        self._call_tool = call_tool
        self._has_tool = has_tool
        self.state = BrowserRouteState()

    def _call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(args)
        # The wrapper owns the session capability.  Never let a model-provided
        # id replace it or address another run's target/ref namespace.
        payload["session"] = self._session_id
        return _tool_payload(self._call_tool(name, payload))

    def _require_tool(self, name: str) -> Optional[Dict[str, Any]]:
        if self._has_tool(name):
            return None
        return _refusal(
            "typed_browser_unavailable",
            f"The connected cua-driver does not advertise {name}; use the native AX/PX/foreground ladder.",
            native_fallback=True,
        )

    def observe(
        self,
        *,
        pid: Any = None,
        window_id: Any = None,
        tab_id: Optional[str] = None,
        snapshot_format: str = "semantic_v2",
        query: Optional[str] = None,
        scope_ref: Optional[str] = None,
        continuation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bind an exact native window or snapshot a bound tab."""
        missing = self._require_tool("get_browser_state")
        if missing is not None:
            return missing

        binding_request = pid is not None or window_id is not None
        if binding_request:
            exact_pid = _positive_int(pid)
            exact_window = _positive_int(window_id)
            self.state.clear()
            if exact_pid is None or exact_window is None:
                return _refusal(
                    "browser_exact_target_required",
                    "Typed browser binding requires an exact positive pid and window_id pair.",
                    native_fallback=True,
                )
            payload = self._call(
                "get_browser_state",
                {"pid": exact_pid, "window_id": exact_window},
            )
            if payload.get("status") != "ok":
                code = _refusal_code(payload)
                payload.setdefault("ok", False)
                payload["native_fallback_available"] = True
                if code == "browser_requires_setup":
                    payload["setup_required"] = True
                return payload

            target_id = payload.get("target_id")
            quality = payload.get("binding_quality")
            mutation_allowed = payload.get("mutation_allowed") is True
            if not isinstance(target_id, str) or not target_id:
                return _refusal(
                    "browser_binding_unproven",
                    "Browser bind returned no opaque target capability; use native control.",
                    native_fallback=True,
                )

            self.state.pid = exact_pid
            self.state.window_id = exact_window
            self.state.target_id = target_id
            self.state.tab_ids = _tab_ids(payload)
            self.state.binding_quality = quality if isinstance(quality, str) else None
            self.state.mutation_allowed = mutation_allowed
            # Binding mints the target/tab capabilities but is not a page
            # snapshot. Require one fresh tab read before any mutation.
            self.state.verification_required = True
            payload["exact_binding"] = quality == "exact"
            if quality != "exact" or not mutation_allowed:
                payload["native_fallback_required"] = True
            return payload

        target_id = self.state.target_id
        if not target_id or self.state.binding_quality != "exact":
            return _refusal(
                "browser_exact_binding_required",
                "Bind the exact native pid/window_id before reading a browser tab.",
                native_fallback=True,
            )
        selected_tab = tab_id or self.state.tab_id
        if not isinstance(selected_tab, str) or not selected_tab:
            return _refusal(
                "browser_tab_required",
                "Choose an opaque tab_id returned by the exact bind.",
            )
        if selected_tab not in self.state.tab_ids:
            return _refusal(
                "browser_tab_unbound",
                "The requested tab_id was not minted by this session's exact bind.",
            )
        if continuation is not None and continuation != self.state.continuation:
            return _refusal(
                "browser_continuation_stale",
                "The continuation is not current for this session/tab; take a fresh snapshot.",
            )
        if scope_ref is not None and scope_ref not in self.state.refs:
            return _refusal(
                "browser_ref_stale",
                "scope_ref must come from this session's latest browser snapshot.",
            )

        args: Dict[str, Any] = {
            "target_id": target_id,
            "tab_id": selected_tab,
            "snapshot_format": snapshot_format,
        }
        if query:
            args["query"] = query
        if scope_ref:
            args["scope_ref"] = scope_ref
        if continuation:
            args["continuation"] = continuation

        continuing = continuation is not None
        if not continuing:
            # A new snapshot supersedes every prior ref before the transport
            # call.  Failure therefore cannot leave a stale ref usable.
            self.state.clear_refs()
        payload = self._call("get_browser_state", args)
        if payload.get("status") not in (None, "ok") or payload.get("isError") is True:
            self.state.clear_refs()
            self.state.verification_required = True
            payload.setdefault("ok", False)
            return payload

        discovered = _ref_map(payload)
        if continuing:
            self.state.refs.update(discovered)
        else:
            self.state.refs = discovered
        self.state.continuation = _continuation(payload)
        self.state.tab_id = selected_tab
        self.state.verification_required = False
        payload["fresh_state"] = True
        payload["refs_current"] = len(self.state.refs)
        return payload

    def prepare(
        self,
        *,
        pid: Any,
        window_id: Any = None,
        profile_mode: str,
        profile_name: Optional[str] = None,
        allow_launch: bool = False,
    ) -> Dict[str, Any]:
        """Run explicit isolated setup; refuse existing-profile attachment."""
        missing = self._require_tool("browser_prepare")
        if missing is not None:
            return missing
        exact_pid = _positive_int(pid)
        if exact_pid is None:
            return _refusal(
                "browser_pid_required", "browser_prepare requires a positive pid."
            )
        if profile_mode == "existing_profile":
            return _refusal(
                "browser_consent_required",
                "Existing-profile attachment requires cua-driver's interactive browser-approve grant bound to the exact pid, window, and session; ordinary tool approval is insufficient.",
                interactive_grant_required=True,
            )
        if profile_mode not in {"isolated_new", "isolated_named"}:
            return _refusal(
                "browser_profile_mode_invalid",
                "Use isolated_new, isolated_named, or existing_profile.",
            )
        if not allow_launch:
            return _refusal(
                "browser_launch_not_approved",
                "Driver-owned isolated setup requires explicit allow_launch=true.",
            )
        profile: Dict[str, Any] = {"mode": profile_mode}
        if profile_mode == "isolated_named":
            if not isinstance(profile_name, str) or not profile_name:
                return _refusal(
                    "browser_profile_name_required",
                    "isolated_named requires a non-empty profile name.",
                )
            profile["name"] = profile_name
        args: Dict[str, Any] = {
            "pid": exact_pid,
            "allow_launch": True,
            "profile": profile,
        }
        exact_window = _positive_int(window_id)
        if exact_window is not None:
            args["window_id"] = exact_window
        # Preparation/reconnect may have side effects even if its transport
        # fails. Invalidate old capabilities before crossing that boundary.
        self.state.clear()
        return self._call("browser_prepare", args)

    def _require_mutation(
        self,
        *,
        tool: str,
        tab_id: Optional[str],
        allow_without_snapshot: bool = False,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        missing = self._require_tool(tool)
        if missing is not None:
            return None, missing
        if (
            not self.state.target_id
            or self.state.binding_quality != "exact"
            or not self.state.mutation_allowed
        ):
            return None, _refusal(
                "browser_mutation_unproven",
                "Typed browser mutation requires status=ok, binding_quality=exact, and mutation_allowed=true; use native control otherwise.",
                native_fallback=True,
            )
        selected_tab = tab_id or self.state.tab_id
        if not isinstance(selected_tab, str) or not selected_tab:
            return None, _refusal(
                "browser_tab_required", "Choose a bound tab_id first."
            )
        if selected_tab not in self.state.tab_ids:
            return None, _refusal(
                "browser_tab_unbound",
                "The requested tab_id was not minted by this session's exact bind.",
            )
        if self.state.verification_required and not allow_without_snapshot:
            return None, _refusal(
                "browser_verification_required",
                "Take a fresh cua_browser_state snapshot before another browser mutation.",
            )
        return selected_tab, None

    def _require_ref(
        self,
        ref: Any,
        *,
        actions: Set[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(ref, str) or ref not in self.state.refs:
            return _refusal(
                "browser_ref_stale",
                "Use a current ref from the latest cua_browser_state snapshot.",
            )
        declared = self.state.refs[ref]
        if actions and not declared.intersection(actions):
            return _refusal(
                "browser_action_unavailable",
                "The current ref does not declare the requested browser action.",
            )
        return None

    def mutate(
        self,
        tool: str,
        *,
        tab_id: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke one typed browser tool against current capabilities."""
        call_args = dict(args or {})
        dialog_inspect = (
            tool == "browser_dialog" and call_args.get("action") == "inspect"
        )
        selected_tab, refusal = self._require_mutation(
            tool=tool,
            tab_id=tab_id,
            allow_without_snapshot=dialog_inspect,
        )
        if refusal is not None:
            return refusal
        assert selected_tab is not None and self.state.target_id is not None

        ref = call_args.get("ref")
        supports_trust_choice = tool in {"browser_click", "browser_pointer"}
        requested_route = call_args.get("input_route")
        if requested_route is not None and not supports_trust_choice:
            return _refusal(
                "browser_input_route_unsupported",
                f"{tool} does not expose a trust-route choice in the live 0.9 schema.",
            )
        route = requested_route or "trusted"
        if route not in {"trusted", "dom_event"}:
            return _refusal(
                "browser_input_route_invalid",
                "Use input_route=trusted or explicitly request dom_event.",
            )
        if route == "dom_event" and not ref:
            return _refusal(
                "browser_dom_event_ref_required",
                "The dom_event trust class requires a current semantic ref.",
            )

        required_actions: Set[str] = set()
        if tool == "browser_click" and ref:
            required_actions = {"click", "pointer"}
        elif tool == "browser_type":
            required_actions = {"type", "edit", "input"}
        elif tool == "browser_pointer" and ref:
            pointer_action = call_args.get("action")
            required_actions = (
                {"scroll", "pointer"} if pointer_action == "scroll" else {"pointer"}
            )
        elif tool == "browser_set_input_files":
            required_actions = {"set_input_files", "upload", "files"}
        elif tool == "browser_download":
            required_actions = {"download", "click"}

        if required_actions:
            invalid_ref = self._require_ref(ref, actions=required_actions)
            if invalid_ref is not None:
                return invalid_ref
        destination_ref = call_args.get("destination_ref")
        if destination_ref is not None:
            invalid_destination = self._require_ref(
                destination_ref, actions={"pointer", "drag", "drop"}
            )
            if invalid_destination is not None:
                return invalid_destination

        call_args["target_id"] = self.state.target_id
        call_args["tab_id"] = selected_tab
        if not dialog_inspect:
            # A lost/refused response does not prove the action was a no-op.
            # Disarm refs before transport so callers must observe fresh state
            # before any retry, trust downgrade, or different mutation.
            self.state.tab_id = selected_tab
            self.state.clear_refs()
            self.state.verification_required = True
        payload = self._call(tool, call_args)
        code = _refusal_code(payload)
        refused = (
            payload.get("isError") is True
            or payload.get("status") not in (None, "ok")
            or code is not None
        )
        if supports_trust_choice:
            payload["input_trust"] = route
            if route == "dom_event":
                payload["trust_downgrade_explicit"] = True

        if refused:
            payload["native_fallback_available"] = True
            if dialog_inspect and code in {
                "browser_ref_stale",
                "browser_binding_ambiguous",
            }:
                self.state.clear_refs()
                self.state.verification_required = True
            if code == "browser_input_trust_unavailable":
                payload["trust_change_requires_explicit_choice"] = True
                payload["native_fallback_available"] = True
            return payload

        if dialog_inspect:
            payload["fresh_dialog_state"] = True
            return payload

        # Never chain mutations from remembered state.  Navigation and a fresh
        # snapshot both invalidate refs in the driver; applying the same rule to
        # all mutations guarantees fresh-state verification before another act.
        payload["verification_required"] = True
        payload["next_step"] = "fresh_browser_state"
        return payload
