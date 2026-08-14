"""Control Room JSON-RPC handlers for the TUI gateway.

Registers ``control.room.snapshot`` (read-only home/section data) and
``control.room.action`` (safe action dispatch through the router). Both are
thin adapters over the shared ``control_room`` package — no business state
lives here, and every action goes through the existing authoritative
executors with the confirmation/stale/cross-profile guards intact.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("control.room.snapshot")
def _(rid, params: dict) -> dict:
    """Return a ControlRoomSnapshot as a JSON-safe dict for the current scope.

    Params:
        profile: optional profile label (defaults to the gateway scope).
        refresh: optional bool to bypass the bounded cache.
        context: optional live surface state (running agents etc).
    """
    try:
        from control_room.service import ControlRoomService

        profile = str(params.get("profile") or "default")
        refresh = bool(params.get("refresh", False))
        raw_context = params.get("context") or {}

        # Gateway live state is supplied by the caller (the TUI/desktop does
        # not push running-agent dicts over RPC; this stays read-only and
        # bounded). The service degrades missing providers to typed
        # unavailable rather than pretending zero.
        service = ControlRoomService()
        snapshot = service.build_snapshot(
            profile=profile,
            context=raw_context,
            refresh=refresh,
        )
        return _ok(rid, snapshot.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"control.room.snapshot failed: {exc}")


@method("control.room.action")
def _(rid, params: dict) -> dict:
    """Dispatch a ControlRoomAction envelope through the router.

    Params:
        action: the full ControlRoomAction dict (id, target, parameters,
            confirmation, expected_revision).
        confirmed: bool — must be true for confirmation-required actions.
    """
    try:
        from control_room.actions import ControlRoomActionRouter
        from control_room.contract import ControlRoomAction
        from control_room.executors import default_executors

        action_dict = params.get("action")
        if not isinstance(action_dict, dict):
            return _err(rid, 5001, "control.room.action requires an 'action' object")
        confirmed = bool(params.get("confirmed", False))
        profile = str(params.get("profile") or "default")

        action = ControlRoomAction(**action_dict)
        router = ControlRoomActionRouter(
            executors=default_executors(),
            scope_profile=profile,
        )
        result = router.dispatch(action, confirmed=confirmed)
        return _ok(rid, result.model_dump(mode="json"))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5002, f"control.room.action failed: {exc}")


# NOTE: _ok/_err are rebound onto the server's globals at install time (see
# method_ctx.py). They exist as module-level references only for type checkers.
def _ok(rid, payload: dict) -> dict:  # pragma: no cover - rebound at install
    return {"jsonrpc": "2.0", "id": rid, "result": payload}


def _err(rid, code: int, message: str) -> dict:  # pragma: no cover - rebound at install
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def register(server) -> None:
    """Bind this module's handlers onto ``server``'s globals and registry."""
    _registry.install(server)
