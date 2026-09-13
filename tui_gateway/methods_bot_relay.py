"""Bot-relay JSON-RPC handlers — the gateway side of cross-connection A2A. Connections ARE the
peer set: the Desktop owns every gateway socket and relays between them via four doors on EACH
gateway: ``roster.sync`` (push OTHER connections' agents so ``message_agent`` resolves them),
``outbox.drain`` (collect envelopes queued here for other connections), ``deliver`` (one-turn Bot
Chat delivery on the TARGET gateway, returns the reply), ``reply`` (write the reply/error back on
the SENDER gateway for its waiter). Plumbing: ``tools/bot_relay.py``; handlers are rebound onto
server.py's globals (method_ctx.py) and reference ``_ok``/``_err`` bare."""

import os
from pathlib import Path

# Defined beside the sender-side waiter budget so the two Python sides cannot drift (#93911).
from tools.bot_relay import TURN_ATTEMPT_TIMEOUT_SECONDS

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


def _relay_root() -> Path:
    """Install root shared by every profile (relay state is install-wide)."""
    home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
    return home.parent.parent if home.parent.name == "profiles" else home


# Historical Desktop deadline mirrors; no subprocess retry is performed here.
# Remove with the renderer relay deadline/receipt migration.
TURN_MAX_ATTEMPTS = 2  # first attempt + the policy-gated re-run


@method("bot_relay.roster.sync")
def _(rid, params: dict, _root=_relay_root) -> dict:
    """Replace this gateway's view of agents on OTHER connections → ``{count}`` accepted rows
    (``agents`` rows ``{profile, handle, connection_id, ...}``; invalid rows are dropped)."""
    try:
        from tools.bot_relay import write_remote_roster
        return _ok(rid, {"count": write_remote_roster(_root(), params.get("agents"))})
    except Exception as e:
        return _err(rid, 5090, str(e))


@method("bot_relay.outbox.drain")
def _(rid, params: dict, _root=_relay_root) -> dict:
    """Claim every pending cross-connection envelope queued here → ``{envelopes}``; claimed
    envelopes move to ``claimed/`` atomically so concurrent drains can't double-deliver."""
    try:
        from tools.bot_relay import claim_pending_envelopes
        return _ok(rid, {"envelopes": claim_pending_envelopes(_root())})
    except Exception as e:
        return _err(rid, 5091, str(e))


@method("bot_relay.deliver")
def _(rid, params: dict, _root=_relay_root) -> dict:
    """Legacy transport bridge only: canonical admission or an explicit refusal."""
    from tools.bot_live_delivery import _delivery_id, authority_delivery
    from tools.bot_relay import _HANDLE_RE
    try:
        _delivery_id(params.get('id'))
        profile = params.get('profile')
        if not isinstance(profile, str) or not _HANDLE_RE.fullmatch(profile):
            raise ValueError('invalid profile')
    except ValueError:
        return _err(rid, 4090, 'invalid_params', data={'reason': 'invalid_params'})
    from tools.bot_relay import delivery_turn_author
    from tui_gateway.methods_browser_control import _is_authenticated_identity
    sender_fields = ("from_profile", "from_handle", "from_connection")
    if (any(params.get(key) for key in sender_fields)
            and _is_authenticated_identity(getattr(current_transport(), "auth_identity", None))):
        return _err(rid, 4095, "a logged-in client cannot name the sender of a relayed dm")
    author = delivery_turn_author(*(params.get(key) for key in sender_fields))
    forwarded = {key: value for key, value in params.items() if key not in sender_fields}
    if author:
        forwarded["author"] = author
    resolved = 'default' if profile.lower() == 'hermes' else profile
    root = _root()
    home = root if resolved == 'default' else root / 'profiles' / resolved
    try:
        return _ok(rid, authority_delivery(home, {**forwarded, 'profile': resolved}))
    except Exception as exc:
        return _err(rid, 5094, str(exc), data={'reason': 'runtime_unavailable'})


@method("bot_relay.reply")
def _(rid, params: dict, _root=_relay_root) -> dict:
    """Write a relayed ``reply`` and/or ``error`` (+ optional typed ``reason``, see
    ``tools.bot_failure_reasons``) for envelope ``id`` so the sender-side waiter picks it up."""
    envelope_id = str(params.get("id") or "").strip()
    if not envelope_id:
        return _err(rid, 4093, "id required")
    try:
        from tools.bot_relay import write_reply
        write_reply(_root(), envelope_id, reply=str(params.get("reply") or ""),
                    error=str(params.get("error") or ""), reason=str(params.get("reason") or ""))
        return _ok(rid, {"ok": True})
    except ValueError as e:
        return _err(rid, 4094, str(e))
    except Exception as e:
        return _err(rid, 5095, str(e))


def register(server) -> None:
    _registry.install(server)
    from . import methods_groups
    server._LONG_HANDLERS = server._LONG_HANDLERS | methods_groups.LONG_HANDLERS
    for name in (
        "get_hosted_room_service", "_WORKER_UNAVAILABLE", "_profile_name", "_requested_profile",
        "_api_server_key", "_room_link_run_storage_durable"):
        setattr(server, name, getattr(methods_groups, name))
    methods_groups.bind_server(server)
    methods_groups.register(server)
