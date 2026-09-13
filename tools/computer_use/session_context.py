"""Desktop-only routing contract for session-owned Cua transports.

Deliberately restrict enumeration rather than filtering host PIDs heuristically.
This contract only narrows tools; existing approval/delivery policy still applies.
"""
from hermes_cli.session_execution import SessionExecutionError

DESKTOP_TARGET = {"kind": "desktop", "display_id": "primary"}
_INPUT = frozenset({"click", "double_click", "drag", "scroll", "type_text", "press_key", "hotkey"})
_READS = frozenset({"get_desktop_state", "get_config"})
_CONFIG = frozenset({"max_image_dimension", "capture_scope"})


def check_desktop_call(execution, name, args):
    if execution is None:
        return
    # Teardown remains allowed after revocation, but can never start a transport.
    if name == "end_session":
        return
    execution.check()
    launch = execution.context.computer_use
    if launch and launch.allow_input is not None and name not in _READS | {"start_session", "set_config", "set_agent_cursor_enabled"}:
        try:
            allowed = launch.allow_input() is True
        except Exception as exc:
            raise SessionExecutionError("computer-use input paused: input policy unavailable") from exc
        if not allowed:
            raise SessionExecutionError("computer-use input paused by session owner")
    if not launch or not launch.desktop_only:
        return
    if any(args.get(key) is not None for key in ("pid", "window_id", "element_index", "element_token", "from_element", "to_element")):
        raise SessionExecutionError("desktop-only context refuses explicit window/PID/element targets")
    if "target" in args and args["target"] != DESKTOP_TARGET:
        raise SessionExecutionError("desktop-only context refuses foreign targets")
    if name in _INPUT:
        if args.get("target") != DESKTOP_TARGET:
            raise SessionExecutionError("desktop-only input requires an explicit desktop target")
        return
    if name in _READS or name in {"start_session", "set_agent_cursor_enabled"}:
        return
    if name == "set_config":
        keys = set(args) - {"session"}
        if keys <= _CONFIG or (keys <= {"key", "value"} and args.get("key") in _CONFIG):
            return
    raise SessionExecutionError(f"desktop-only context does not expose {name}")


def check_desktop_request(backend, args):
    """Reject explicit foreign selectors before the generic dispatcher drops them."""
    if not getattr(backend, "desktop_only", False):
        return
    if (args.get("app") not in (None, "", "screen")
            or any(args.get(k) is not None for k in ("pid", "window_id", "element", "from_element", "to_element"))
            or args.get("raise_window") or args.get("bring_to_front")):
        backend._clear_active_target()
        raise SessionExecutionError("desktop-only context requires app='screen'; explicit app/PID/window/AX targeting is unavailable")
