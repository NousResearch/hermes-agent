"""Launch-profile-only telemetry RPCs. Registry/auth live in the backend, never renderer input."""
from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


def valid_request_id(rid):
    return (rid is None or (type(rid) is int and -(2**63) <= rid < 2**63)
            or (isinstance(rid, str) and len(rid) <= 128))


def _usage_method(name):
    def decorate(fn):
        def handler(rid, params):
            from tui_gateway.transport import current_transport
            from tui_gateway.ws import WSTransport
            transport = current_transport()
            # The inherited stdio pipe is the TUI capability. WebSockets must have
            # passed the existing upgrade gate; an arbitrary Transport is not auth.
            if transport is not _stdio_transport and not (
                isinstance(transport, WSTransport) and getattr(transport, "authenticated", False) is True
            ):
                return _err(rid, 4403, "authenticated telemetry transport required")
            if params:
                return _err(rid, -32602, "telemetry methods accept no parameters")
            if transport is not _stdio_transport and getattr(transport, "local_telemetry", False) is not True:
                return _err(rid, 4403, "remote telemetry unsupported")
            try:
                import json
                result = fn(_hermes_home)
                if len(json.dumps(result, allow_nan=False).encode("utf-8")) > 32768:
                    return _err(rid, 5033, "telemetry payload unavailable")
                return _ok(rid, result)
            except Exception:
                # Never let dispatch's generic exception formatter expose paths or credentials.
                return _err(rid, 5033, "telemetry unavailable")
        return method(name)(handler)
    return decorate


@_usage_method("usage.codex_quota")
def _usage_quota(home):
    from tui_gateway.usage_telemetry import quota
    return quota(home)


@_usage_method("usage.codex_timeline")
def _usage_timeline(home):
    from tui_gateway.usage_telemetry import timeline
    return timeline(home)


@_usage_method("usage.active_work")
def _usage_work(home):
    from tui_gateway.usage_active_work import active_work
    return active_work(home, _sessions, _sessions_lock)


def register(server):
    bind_module(globals(), server)
