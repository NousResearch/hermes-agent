"""Bot-relay JSON-RPC handlers for cross-connection A2A."""

import contextlib
import os
import subprocess
from pathlib import Path

# Defined beside the sender-side waiter budget so the two Python sides cannot drift (#93911).
from tools.bot_failure_reasons import delivery_failure_reason
from tools.bot_relay import TURN_ATTEMPT_TIMEOUT_SECONDS

from .contracts.groups_bot_relay import (
    BotRelayDeliverParams, BotRelayDeliverResult, BotRelayOutboxDrainParams,
    BotRelayOutboxDrainResult, BotRelayReplyParams, BotRelayRosterSyncParams,
    BotRelayRosterSyncResult,
)
from .contracts.prompt_voice import PromptSubmitParams
from .contracts.common import OkResult
from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


def _relay_root() -> Path:
    """Install root shared by every profile (relay state is install-wide)."""
    from tools.bot_mode_probe import _default_home, _hermes_root
    return _hermes_root(Path(_default_home()))


def _run_delivery(profile: str, tmp: str, env: dict | None = None) -> subprocess.CompletedProcess:
    from tools.bot_relay import local_delivery_command
    return subprocess.run(
        local_delivery_command(profile, tmp), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=TURN_ATTEMPT_TIMEOUT_SECONDS, env=env)


@method("bot_relay.roster.sync")
def _(rid, params: BotRelayRosterSyncParams, _root=_relay_root) -> BotRelayRosterSyncResult | dict:
    """Replace this gateway's view of agents on other connections."""
    try:
        from tools.bot_relay import write_remote_roster
        agents = [agent.model_dump(mode="json") for agent in params.agents] if params.agents else None
        return BotRelayRosterSyncResult(count=write_remote_roster(_root(), agents))
    except Exception as exc:
        return srv._err(rid, 5090, str(exc))


@method("bot_relay.outbox.drain")
def _(rid, params: BotRelayOutboxDrainParams, _root=_relay_root) -> BotRelayOutboxDrainResult | dict:
    """Claim every pending cross-connection envelope queued here."""
    try:
        from tools.bot_relay import claim_pending_envelopes
        return BotRelayOutboxDrainResult(envelopes=claim_pending_envelopes(_root()))
    except Exception as exc:
        return srv._err(rid, 5091, str(exc))


@method("bot_relay.deliver")
def _(rid, params: BotRelayDeliverParams, _root=_relay_root, _run=_run_delivery,
      _failure_reason=delivery_failure_reason) -> BotRelayDeliverResult | dict:
    """Deliver a relayed DM (``profile``, attribution-prefixed ``message``) into a Bot Chat ON THIS
    GATEWAY via the one-turn ``hermes -p <profile> chat -c "Bot Chat"`` transport local DMs use →
    ``{reply}``. Blocking by design (Desktop relay worker; the RPC pool keeps it off the reader)."""
    import tempfile

    profile = params.profile.strip()
    message = params.message.strip()
    if not profile or not message:
        return srv._err(rid, 4090, "profile and message required")
    try:
        from tools.bot_mode_dm import MESSAGE_MAX_CHARS
        from tools.bot_relay import DeliveryAuthor, acquire_turn_lock, delivery_env, delivery_turn_author
        if len(message) > MESSAGE_MAX_CHARS + 200:
            return srv._err(rid, 4091, "message too long")
        root = _root()
        from tools.bot_mode_probe import _roster
        known = {name for name, _ in _roster(root)}
        resolved = "default" if profile.lower() == "hermes" else profile
        if resolved not in known:
            return srv._err(rid, 4092, f"no profile '{profile}' on this gateway")

        from tools.bot_mode_probe import BOT_CHAT_TITLE
        live_home = srv._profile_home(resolved)
        want_home = str(live_home) if live_home is not None else None
        live_sid = next((
            sid for sid, record in list(srv._sessions.items())
            if isinstance(record, dict) and (record.get("profile_home") or None) == want_home
            and srv._session_live_title(record, srv._session_lookup_key(record, fallback=sid)) == BOT_CHAT_TITLE), "")
        sender_fields = (params.from_profile, params.from_handle, params.from_connection)
        from tui_gateway.methods_browser_control import _is_authenticated_identity
        if any(sender_fields) and _is_authenticated_identity(getattr(srv.current_transport(), "auth_identity", None)):
            return srv._err(rid, 4095, "a logged-in client cannot name the sender of a relayed dm")
        author = delivery_turn_author(*sender_fields)
        if live_sid:
            submitted = srv.invoke(
                "prompt.submit", PromptSubmitParams(session_id=live_sid, text=message, queued=True),
                _turn_author=DeliveryAuthor(author) if author else None,
            )
            if isinstance(submitted, dict):
                return submitted
            return BotRelayDeliverResult(
                reply=f"Delivered into @{resolved}'s open Bot Chat; the reply will appear there.")

        # This process's _sessions is not the ownership authority: the Desktop pools one backend per
        # (connection, profile) and an SSH source runs one remote dashboard per profile, so the
        # target's Bot Chat can be live in a sibling process on this host while the relay RPC lands
        # here. The subprocess transport would then be refused SESSION_NOT_OWNED by that owner's
        # lease (#113753). Hand the DM to the live owner through the same mailbox local DMs use
        # (tools/bot_mode_dm.py::_run_delivery); its poller admits it at the next idle boundary.
        from tools.bot_live_delivery import deliver_to_live_owner, find_canonical_live_owner
        owner_home = live_home if live_home is not None else Path(srv._hermes_home)
        owner = find_canonical_live_owner(owner_home)
        if owner is not None:
            deliver_to_live_owner(owner_home, owner, message, author=author)
            # The owner's poller admits the mailbox record at its next idle boundary; this
            # process only queued it, so say so (the in-process branch above really submitted).
            reply = f"Queued for @{resolved}'s open Bot Chat; it runs as that chat's next turn and the reply will appear there."
            return BotRelayDeliverResult(reply=reply)

        def _detail(p) -> str:
            from tools.bot_failure_reasons import turn_failure_text
            return turn_failure_text(p.stdout, p.stderr)

        turn_env = delivery_env(author, live_home)

        fd, tmp = tempfile.mkstemp(prefix="hermes-relay-dm-", suffix=".txt", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(message)
            with acquire_turn_lock(root, resolved):
                proc = _run(resolved, tmp, turn_env)
                if proc.returncode != 0:
                    from tools.bot_failure_reasons import RETRY_NONE, classify_agent_error, retry_action
                    if retry_action(classify_agent_error(_detail(proc))) != RETRY_NONE:
                        # The failed attempt already persisted the DM; the re-run resumes that row.
                        from tools.bot_relay import retry_turn_env
                        proc = _run(resolved, tmp, retry_turn_env(turn_env))
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        if proc.returncode != 0:
            from tools.bot_failure_reasons import classify_agent_error
            detail = _detail(proc)
            return srv._err(rid, 5092, f"delivery turn failed: {detail[-500:] or proc.returncode}",
                        data={"reason": classify_agent_error(detail)})
        # Use the same canonical whole-response predicate as live Bot Chat
        # completion.  A marker remains a successful turn, but is never sent
        # back to the relay caller as visible prose.
        from tui_gateway.prompt_turn import _bot_mode_delivery_text
        reply = _bot_mode_delivery_text((proc.stdout or "").strip(), successful=True)
        return BotRelayDeliverResult(reply=reply)
    except subprocess.TimeoutExpired:
        # Every classified refusal has to ride `data.reason`: the Desktop forwards only that field,
        # and the sender re-classifies from free text, which cannot name these. This branch is also
        # `delivery_timeout`'s only producer.
        from tools.bot_failure_reasons import DELIVERY_TIMEOUT
        return srv._err(rid, 5093, "delivery turn timed out", data={"reason": DELIVERY_TIMEOUT})
    except Exception as e:
        reason = _failure_reason(e)
        return srv._err(rid, 5096 if reason == "target_busy" else 5094, str(e), data={"reason": reason})


@method("bot_relay.reply")
def _(rid, params: BotRelayReplyParams, _root=_relay_root) -> OkResult | dict:
    """Write a relayed reply or typed error for an envelope."""
    envelope_id = params.id.strip()
    if not envelope_id:
        return srv._err(rid, 4093, "id required")
    try:
        from tools.bot_relay import write_reply
        write_reply(_root(), envelope_id, reply=params.reply or "", error=params.error or "", reason=params.reason or "")
        return OkResult(ok=True)
    except ValueError as exc:
        return srv._err(rid, 4094, str(exc))
    except Exception as exc:
        return srv._err(rid, 5095, str(exc))


def register(server) -> None:
    _registry.install(server, globals())
    from . import methods_groups
    server._LONG_HANDLERS = server._LONG_HANDLERS | methods_groups.LONG_HANDLERS
    for name in (
        "get_hosted_room_service", "_WORKER_UNAVAILABLE", "_profile_name", "_requested_profile",
        "_api_server_key", "_room_link_run_storage_durable"):
        setattr(server, name, getattr(methods_groups, name))
    methods_groups.bind_server(server)
    methods_groups.register(server)

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
