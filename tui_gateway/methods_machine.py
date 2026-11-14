"""``machine.facts``: the host summary the desktop first-run onboarding branches on.

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


# Unscoped on purpose: hardware and the OS account belong to the host, not to any profile home.
@method("machine.facts")
def _(rid, params: dict) -> dict:
    from hermes_platform.host.summary import summary
    # The account's name reaches the desktop only on the onboarding name card.
    return _ok(rid, {key: value for key, value in summary().items() if key != "full_name"})


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
