"""Bot-relay JSON-RPC handlers — the gateway side of cross-connection A2A. Connections ARE the
peer set: the Desktop owns every gateway socket and relays between them via four doors on EACH
gateway: ``roster.sync`` (push OTHER connections' agents so ``message_agent`` resolves them),
``outbox.drain`` (collect envelopes queued here for other connections), ``deliver`` (one-turn Bot
Chat delivery on the TARGET gateway, returns the reply), ``reply`` (write the reply/error back on
the SENDER gateway for its waiter). Plumbing: ``tools/bot_relay.py``; handlers are rebound onto
server.py's globals (method_ctx.py) and reference ``_ok``/``_err`` bare."""

import os
import subprocess
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


def _run_delivery(profile: str, tmp: str, env: dict | None = None) -> subprocess.CompletedProcess:
    from tools.bot_relay import local_delivery_command
    return subprocess.run(
        local_delivery_command(profile, tmp), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=TURN_ATTEMPT_TIMEOUT_SECONDS, env=env)


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
def _(rid, params: dict, _root=_relay_root, _run=_run_delivery) -> dict:
    """Deliver a relayed DM (``profile``, attribution-prefixed ``message``) into a Bot Chat ON THIS
    GATEWAY via the one-turn ``hermes -p <profile> chat -c "Bot Chat"`` transport local DMs use →
    ``{reply}``. Blocking by design (Desktop relay worker; the RPC pool keeps it off the reader)."""
    import tempfile
    profile = str(params.get("profile") or "").strip()
    message = str(params.get("message") or "").strip()
    if not profile or not message:
        return _err(rid, 4090, "profile and message required")
    try:
        from tools.bot_mode_dm import MESSAGE_MAX_CHARS
        from tools.bot_relay import acquire_turn_lock
        if len(message) > MESSAGE_MAX_CHARS + 200:  # + attribution headroom
            return _err(rid, 4091, "message too long")
        root = _root()
        known = {"default"}
        if (root / "profiles").is_dir():
            known.update(c.name for c in (root / "profiles").iterdir() if c.is_dir())
        resolved = "default" if profile.lower() == "hermes" else profile
        if resolved not in known:
            return _err(rid, 4092, f"no profile '{profile}' on this gateway")

        # When THIS gateway already hosts the target's Bot Chat live, the subprocess transport is
        # fenced out by the single-owner lease and the payload dropped. Land the DM in the live
        # session via prompt.submit — the composer's choke point, so role alternation, persistence
        # and streaming behave as a typed message would.
        # (Nested per method_ctx rebinding.) See #100523.
        from tools.bot_mode_probe import BOT_CHAT_TITLE
        live_home = _profile_home(resolved)
        want_home = str(live_home) if live_home is not None else None
        live_sid = next((
            live_sid for live_sid, record in list(_sessions.items())
            if isinstance(record, dict) and (record.get("profile_home") or None) == want_home
            and _session_live_title(
                record, _session_lookup_key(record, fallback=live_sid)) == BOT_CHAT_TITLE), "")
        # The sender fields are whatever the relaying client says. The author labels memory only and grants nothing.
        from tools.bot_relay import (
            DeliveryAuthor, delivery_env, delivery_turn_author, relaying_principal_author)
        from tui_gateway.methods_browser_control import _is_authenticated_identity, _principal_digest
        sender_fields = ("from_profile", "from_handle", "from_connection")
        identity = getattr(current_transport(), "auth_identity", None)
        if _is_authenticated_identity(identity):
            # A logged-in client's sender fields are NOT trusted — but the delivery is not refused
            # either: the Desktop is itself a logged-in client on every gateway that requires sign-in
            # (it mints a ws-ticket carrying the signed-in {user_id, provider} —
            # hermes_cli/dashboard_auth/routes.py), so refusing took cross-connection relay offline
            # for exactly the auth-gated gateways it serves; only ``?internal=`` callers are
            # identity-exempt and the Desktop cannot present one. Nor is the author dropped: an
            # unattributed turn is the HUMAN's to the recipient's memory (Honcho routes it into the
            # human session and allows conclusion / profile / mirror writes), so a bot DM must stay
            # bot-authored. The author is derived from the caller's minted identity instead — stable,
            # unspoofable, and ``is_bot`` — whether or not the client named a sender. The human-facing
            # "Message from 🤖 …" signature stays in the text the sender composed.
            author = relaying_principal_author(_principal_digest(identity))
        else:
            author = delivery_turn_author(*(params.get(k) for k in sender_fields))
        if live_sid:
            # queued=True: a teammate's DM runs as the NEXT turn and never interrupts or steers a
            # turn in flight (the default busy mode does); arrivals queue in order.
            submit_params: dict = {"session_id": live_sid, "text": message, "queued": True}
            if author:
                submit_params["_turn_author"] = DeliveryAuthor(author)
            submitted = _methods["prompt.submit"](rid, submit_params)
            if "error" in submitted:
                return submitted
            reply = f"Delivered into @{resolved}'s open Bot Chat; the reply will appear there."
            return _ok(rid, {"reply": reply})

        def _detail(p) -> str:
            return (p.stderr or p.stdout or "").strip()[-500:]

        turn_env = delivery_env(author)

        fd, tmp = tempfile.mkstemp(prefix="hermes-relay-dm-", suffix=".txt", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(message)
            # Per-profile turn lock serializes with any other delivery turn into this profile and
            # covers only the turn window. Worst-case hold is lock wait (bot_mode.turn_wait_seconds,
            # default 120s) + the 600s turn timeout, doubled on one retry — callers tolerate ~1320s.
            # Worst-case handler hold is lock wait (bot_mode.turn_wait_seconds, default 120s) + the 600s
            # turn timeout below — doubled when the retry policy grants one bounded re-run — so clients
            # calling bot_relay.deliver must tolerate ~1320s before assuming failure. See #93091.
            with acquire_turn_lock(root, resolved):
                proc = _run(resolved, tmp, turn_env)
                if proc.returncode != 0:
                    # Retry policy: transient classes re-run the SAME session once; context_overflow
                    # too — the retried turn's pre-API compaction pass compacts the over-threshold
                    # transcript first (no fresh session is minted). Auth/quota/config never retry.
                    # See #93091.
                    from tools.bot_failure_reasons import (
                        RETRY_NONE, classify_agent_error, retry_action)
                    if retry_action(classify_agent_error(_detail(proc))) != RETRY_NONE:
                        proc = _run(resolved, tmp, turn_env)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        if proc.returncode != 0:
            from tools.bot_failure_reasons import classify_agent_error
            detail = _detail(proc)
            return _err(rid, 5092, f"delivery turn failed: {detail or proc.returncode}",
                        data={"reason": classify_agent_error(detail)})
        return _ok(rid, {"reply": (proc.stdout or "").strip()})
    except subprocess.TimeoutExpired:
        return _err(rid, 5093, "delivery turn timed out")
    except Exception as e:
        # 'target_busy' extends the structured refusal enum.
        return _err(rid, 5096 if getattr(e, "reason", "") == "target_busy" else 5094, str(e))


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
