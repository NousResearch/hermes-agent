"""Desktop-only routing contract for session-owned Cua transports.

Deliberately restrict enumeration rather than filtering host PIDs heuristically.
This contract only narrows tools; existing approval/delivery policy still applies.
"""
from contextlib import contextmanager
from contextvars import ContextVar

from hermes_cli.session_execution import SessionExecutionError

# Copied into the async bridge's send task; concurrent callers never share fences.
_access_fence = ContextVar("cua_access_fence", default=None)


def read_access_epoch(execution):
    if execution is None:
        return None
    execution.check()
    launch = execution.context.computer_use
    if launch is None or launch.access_epoch is None:
        return None
    try:
        epoch = launch.access_epoch()
    except SessionExecutionError:
        raise
    except Exception as exc:
        raise SessionExecutionError("computer-use access policy unavailable") from exc
    if type(epoch) is not int or epoch < 0:
        raise SessionExecutionError("computer-use access epoch must be a nonnegative integer")
    return epoch


def check_access_epoch(execution):
    fence = _access_fence.get()
    if fence is not None and fence[0] is execution and fence[2] is not None:
        fence[2]()
    epoch = read_access_epoch(execution)
    if fence is not None and fence[0] is execution and fence[1] != epoch:
        raise SessionExecutionError("computer-use control epoch changed during operation")
    return epoch


@contextmanager
def desktop_access(execution, *, check=None):
    previous = _access_fence.get()
    if check is None and previous is not None and previous[0] is execution:
        check = previous[2]
    if check is not None:
        check()
    epoch = check_access_epoch(execution)
    token = _access_fence.set((execution, epoch, check))
    try:
        yield
    finally:
        _access_fence.reset(token)


def capture_checked(backend, **kwargs):
    """Fence pixels before response shaping can persist them or call vision."""
    execution = getattr(backend, "execution_context", None)
    with desktop_access(execution):
        try:
            result = backend.capture(**kwargs)
            check_access_epoch(execution)
            return result
        except SessionExecutionError:
            clear = getattr(backend, "_clear_active_target", None)
            if clear is not None:
                clear()
            raise

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
    check_access_epoch(execution)
    launch = execution.context.computer_use
    if launch and launch.desktop_attestor is not None and args.get("screenshot_out_file") is not None:
        raise SessionExecutionError("remote desktop capture requires inline pixels")
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
