import json

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


@method("session.start_chat")
@_profile_scoped
def _(rid, params: dict) -> dict:
    from tui_gateway.start_chat import start_chat
    return _ok(rid, json.loads(start_chat(params.get("args") or {}, str(params.get("session_id") or ""))))


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
