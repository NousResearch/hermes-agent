"""Bot-relay JSON-RPC handlers — the gateway side of cross-connection A2A. Connections ARE the
peer set: the Desktop owns every gateway socket and relays between them via four doors on EACH
gateway: ``roster.sync`` (push OTHER connections' agents so ``message_agent`` resolves them),
``outbox.drain`` (collect envelopes queued here for other connections), ``deliver`` (one-turn Bot
Chat delivery on the TARGET gateway, returns the reply), ``reply`` (write the reply/error back on
the SENDER gateway for its waiter). Plumbing: ``tools/bot_relay.py``; handlers are rebound onto
server.py's globals (method_ctx.py) and reference ``_ok``/``_err`` bare."""

import contextlib
import os
import subprocess
from pathlib import Path

# Defined beside the sender-side waiter budget so the two Python sides cannot drift (#93911).
from tools.bot_relay import TURN_ATTEMPT_TIMEOUT_SECONDS

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


def _relay_root() -> Path:
    """Install root shared by every profile (relay state is install-wide). Same formula as the
    writers (``tools/bot_relay``, ``tools/bot_mode_dm``): both ends of the mailbox must agree for
    every HERMES_HOME, including non-``profiles/`` subdirs of ``~/.hermes``."""
    from tools.bot_mode_probe import _default_home, _hermes_root
    return _hermes_root(Path(_default_home()))


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
    """Deliver a relayed DM into this gateway's Bot Chat.

    Identified deliveries keep an immutable target-side receipt while the turn
    runs. A caller that timed out can therefore retry the same id and recover
    the original reply without executing the Bot Chat twice.
    """
    import tempfile
    from tools.bot_relay import (
        DeliveryAuthor, _validate_envelope_id, _write_delivery_receipt_locked,
        acquire_turn_lock, delivery_env, delivery_request_digest,
        delivery_receipt_lock, delivery_turn_author, read_delivery_receipt,
    )

    profile = str(params.get("profile") or "").strip()
    message = str(params.get("message") or "").strip()
    if not profile or not message:
        return _err(rid, 4090, "profile and message required")
    from tools.bot_mode_dm import MESSAGE_MAX_CHARS
    if len(message) > MESSAGE_MAX_CHARS + 200:  # + attribution headroom
        return _err(rid, 4091, "message too long")

    root = _root()
    known = {"default"}
    if (root / "profiles").is_dir():
        known.update(c.name for c in (root / "profiles").iterdir() if c.is_dir())
    resolved = "default" if profile.lower() == "hermes" else profile
    if resolved not in known:
        return _err(rid, 4092, f"no profile '{profile}' on this gateway")

    # The sender fields are whatever the relaying client says. The author labels memory only and grants nothing.
    from tui_gateway.methods_browser_control import _is_authenticated_identity
    sender_fields = ("from_profile", "from_handle", "from_connection")
    # A logged-in browser never relays for another connection; only the Desktop and server-internal callers do.
    if (any(params.get(k) for k in sender_fields)
            and _is_authenticated_identity(getattr(current_transport(), "auth_identity", None))):
        return _err(rid, 4095, "a logged-in client cannot name the sender of a relayed dm")

    envelope_id = None
    if params.get("id") not in (None, ""):
        try:
            envelope_id = _validate_envelope_id(params.get("id"))
        except ValueError as exc:
            return _err(rid, 4096, str(exc), data={"reason": "invalid_params"})

    request_digest = (
        delivery_request_digest(
            resolved, message,
            from_profile=params.get("from_profile"),
            from_handle=params.get("from_handle"),
            from_connection=params.get("from_connection"),
        )
        if envelope_id
        else ""
    )

    def _receipt_response(receipt: dict) -> dict:
        if receipt.get("status") == "failed":
            reason = str(receipt.get("reason") or "").strip()
            data = {"reason": reason} if reason else None
            return _err(
                rid,
                int(receipt.get("code") or 5092),
                str(receipt.get("error") or "delivery failed"),
                data=data,
            )
        return _ok(rid, {"reply": str(receipt.get("reply") or "")})

    def _deliver_once() -> dict:
        try:
            # When THIS gateway already hosts the target's Bot Chat live, the subprocess transport is
            # fenced out by the single-owner lease and the payload dropped. Land the DM in the live
            # session via prompt.submit — the composer's choke point, so role alternation, persistence
            # and streaming behave as a typed message would. See #100523.
            from tools.bot_mode_probe import BOT_CHAT_TITLE
            live_home = _profile_home(resolved)
            want_home = str(live_home) if live_home is not None else None
            live_sid = next((
                live_sid for live_sid, record in list(_sessions.items())
                if isinstance(record, dict) and (record.get("profile_home") or None) == want_home
                and _session_live_title(
                    record, _session_lookup_key(record, fallback=live_sid)) == BOT_CHAT_TITLE), "")
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
            stream = None
            try:
                stream = os.fdopen(fd, "w", encoding="utf-8")
                with stream as f:
                    f.write(message)
                # The profile lock serializes delivery turns. Worst-case hold is the configured wait plus
                # the bounded attempt window, doubled only by the existing retry policy. See #93091.
                with acquire_turn_lock(root, resolved):
                    proc = _run(resolved, tmp, turn_env)
                    if proc.returncode != 0:
                        from tools.bot_failure_reasons import (
                            RETRY_NONE, classify_agent_error, retry_action)
                        if retry_action(classify_agent_error(_detail(proc))) != RETRY_NONE:
                            proc = _run(resolved, tmp, turn_env)
            finally:
                # os.fdopen normally owns and closes fd. Keep the explicit close as a
                # fallback for a wrapper that fails before taking ownership (and for
                # the gateway's injected writer doubles), before unlinking on Windows.
                if stream is not None:
                    with contextlib.suppress(AttributeError, OSError):
                        stream.close()
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
            if proc.returncode != 0:
                from tools.bot_failure_reasons import classify_agent_error
                detail = _detail(proc)
                return _err(rid, 5092, f"delivery turn failed: {detail or proc.returncode}",
                            data={"reason": classify_agent_error(detail)})
            return _ok(rid, {"reply": (proc.stdout or "").strip()})
        except subprocess.TimeoutExpired:
            return _err(rid, 5093, "delivery turn timed out", data={"reason": "delivery_timeout"})
        except Exception as exc:
            # 'target_busy' extends the structured refusal enum. It is not a terminal receipt: the
            # Desktop keeps the envelope claimed and retries it after the competing turn settles.
            reason = str(getattr(exc, "reason", "") or "").strip()
            return _err(
                rid,
                5096 if reason == "target_busy" else 5094,
                str(exc),
                data={"reason": reason} if reason else None,
            )

    if envelope_id is None:
        return _deliver_once()

    with delivery_receipt_lock(root, envelope_id, lock_key=resolved) as safe_id:
        receipt = read_delivery_receipt(root, safe_id)
        if receipt is not None:
            if receipt.get("request_digest") != request_digest:
                return _err(rid, 4097, "delivery id already belongs to a different payload",
                            data={"reason": "invalid_params"})
            return _receipt_response(receipt)

        response = _deliver_once()
        if "error" in response:
            error = response["error"] if isinstance(response.get("error"), dict) else {}
            reason = str((error.get("data") or {}).get("reason") or "").strip()
            if reason != "target_busy":
                _write_delivery_receipt_locked(
                    root,
                    safe_id,
                    request_digest=request_digest,
                    status="failed",
                    error=str(error.get("message") or "delivery failed"),
                    reason=reason,
                    code=int(error.get("code") or 5092),
                )
        else:
            result = response.get("result") if isinstance(response.get("result"), dict) else {}
            _write_delivery_receipt_locked(
                root,
                safe_id,
                request_digest=request_digest,
                status="settled",
                reply=str(result.get("reply") or ""),
            )
        return response


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
