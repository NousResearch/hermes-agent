"""Group-relay JSON-RPC handlers — the gateway door for ``hermes group send``
into Desktop-coordinated Group Chats.

Two methods the Desktop's ``hermes-bots`` plugin calls on its connection:

- ``group_relay.outbox.drain`` — claim envelopes queued by the CLI
  (``tools/group_relay.enqueue``). The Desktop then calls its own
  ``sendToGroupChat`` on the user's behalf.
- ``group_relay.reply`` — append one progress line (``accepted`` / ``reply``
  / ``done`` / ``error``) for a claimed envelope; the CLI's ``--wait`` tails
  the file.

Storage lives in ``tools/group_relay.py``. Handlers are rebound onto
server.py's globals at install time (see method_ctx.py) and reference
``_ok``/``_err`` from there.
"""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("group_relay.outbox.drain")
def _(rid, params: dict) -> dict:
    """Claim every pending group-relay envelope on this gateway. Result: ``{envelopes}``."""
    try:
        from tools.group_relay import claim_pending, gateway_root

        return _ok(rid, {"envelopes": claim_pending(gateway_root())})
    except Exception as e:
        return _err(rid, 5095, str(e))


@method("group_relay.reply")
def _(rid, params: dict) -> dict:
    """Append one progress line for envelope ``id``. Params: ``id``, ``line`` (object)."""
    try:
        from tools.group_relay import GroupRelayError, append_reply_line, gateway_root

        line = params.get("line")
        if not isinstance(line, dict):
            return _err(rid, 4095, "line must be an object")
        try:
            append_reply_line(gateway_root(), params.get("id"), line)
        except GroupRelayError as exc:
            return _err(rid, 4096, str(exc))
        return _ok(rid, {"ok": True})
    except Exception as e:
        return _err(rid, 5096, str(e))


def register(server) -> None:
    _registry.install(server)
