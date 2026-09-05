"""Bot-relay JSON-RPC handlers — the gateway side of cross-connection A2A. Connections ARE the
peer set: the Desktop owns every gateway socket and relays between them via four doors on EACH
gateway: ``roster.sync`` (push OTHER connections' agents so ``message_agent`` resolves them),
``outbox.drain`` (collect envelopes queued here for other connections), ``deliver`` (one-turn Bot
Chat delivery on the TARGET gateway, returns the reply), ``receipt.read`` (read back the durable
target receipt), ``reply`` (write the reply/error back on
the SENDER gateway for its waiter). Plumbing: ``tools/bot_relay.py``; handlers are rebound onto
server.py's globals (method_ctx.py) and reference ``_ok``/``_err`` bare."""

import contextlib
import math
import os
import re
import subprocess
import time
from pathlib import Path

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


def _relay_root() -> Path:
    """Install root shared by every profile (relay state is install-wide)."""
    from tools.bot_relay import relay_install_root

    home = Path(os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes"))
    return relay_install_root(home)


# Per-attempt turn timeout and attempt ceiling for bot_relay.deliver. The Desktop client mirrors
# both (apps/desktop/src/plugins/hermes-bots/relay.ts: RELAY_TURN_ATTEMPT_MS / RELAY_TURN_MAX_ATTEMPTS)
# and its relay-deliver-budget test reads these two lines, so a change here must be deliberate (#93911).
TURN_ATTEMPT_TIMEOUT_SECONDS = 600
TURN_MAX_ATTEMPTS = 2  # first attempt + the policy-gated re-run
LIVE_TARGET_PERSIST_TIMEOUT_SECONDS = 30
LIVE_TARGET_PERSIST_POLL_SECONDS = 0.05


def _run_delivery(profile: str, tmp: str) -> subprocess.CompletedProcess:
    from tools.bot_relay import local_delivery_command
    return subprocess.run(
        local_delivery_command(profile, tmp), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=TURN_ATTEMPT_TIMEOUT_SECONDS)


class _DeliveryContextError(ValueError):
    """A validation failure with the legacy machine-readable relay reason."""

    def __init__(self, message: str, *, code: int = 4097, reason: str = "target_receipt_unverified"):
        super().__init__(message)
        self.code = code
        self.reason = reason


def _live_target_message_persisted(db, session_key: str, message_id: str, message: str) -> bool:
    """Prove the structured relay user row reached the target SessionDB.

    ``prompt.submit`` acknowledges after starting a background turn. The relay must not
    turn that sender ACK into a completed target receipt until the exact platform message
    id and sanitized content are visible through the durable read path.
    """
    if db is None or not session_key or not message_id:
        return False
    getter = getattr(db, "get_messages_as_conversation", None)
    if not callable(getter):
        return False
    try:
        history = getter(session_key, include_row_ids=True)
    except Exception:
        return False
    if not isinstance(history, list):
        return False
    expected = str(message or "")
    try:
        from hermes_cli.input_sanitize import sanitize_user_prompt_text
        expected = sanitize_user_prompt_text(expected)
    except Exception:
        pass
    try:
        from agent.memory_manager import sanitize_context
        expected = sanitize_context(expected).strip()
    except Exception:
        expected = expected.strip()
    for row in history:
        if not isinstance(row, dict) or row.get("role") != "user":
            continue
        if str(row.get("message_id") or "") != str(message_id):
            continue
        content = row.get("content")
        if not isinstance(content, str):
            continue
        try:
            from agent.memory_manager import sanitize_context
            content = sanitize_context(content)
        except Exception:
            pass
        if content.strip() == expected:
            return True
    return False


def _delivery_context(
    params: dict, *, profile: str, message: str, resolved: str, enforce_expiry: bool = True,
) -> dict:
    """Validate one delivery envelope and derive the exact readback inputs."""
    from tools.bot_relay import ENVELOPE_SCHEMA, MESSAGE_TYPES, delivery_fingerprint

    envelope_schema = str(params.get("envelope_schema") or "").strip()
    envelope = params.get("envelope") if isinstance(params.get("envelope"), dict) else {}
    message_id = str(params.get("message_id") or "").strip()
    idempotency_key = str(params.get("idempotency_key") or "").strip()
    if envelope_schema and envelope_schema != ENVELOPE_SCHEMA:
        raise ValueError(f"unsupported structured envelope schema: {envelope_schema}")
    structured = envelope_schema == ENVELOPE_SCHEMA
    if not structured:
        return {
            "structured": False,
            "schema": envelope_schema,
            "envelope": envelope,
            "message_id": message_id,
            "idempotency_key": idempotency_key,
            "target_connection": "",
            "target_profile": resolved,
            "target_handle": "",
            "fingerprint": delivery_fingerprint(
                envelope, target_profile=resolved, message=message, structured=False
            ),
        }

    required = (
        "schema", "message_id", "idempotency_key", "type", "from_agent", "to_agent",
        "scope", "expires_at", "target_connection", "target_profile", "message",
    )
    missing = [key for key in required if not envelope.get(key)]
    if missing:
        raise ValueError(f"structured envelope missing: {', '.join(missing)}")
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        raise ValueError("structured envelope schema mismatch")
    if not re.fullmatch(r"[0-9a-f]{32}", str(envelope.get("message_id") or "")):
        raise ValueError("structured envelope has invalid message_id")
    if str(envelope.get("message_id")) != message_id or str(envelope.get("idempotency_key")) != idempotency_key:
        raise ValueError("structured envelope identity mismatch")
    if str(envelope.get("message") or "") != message:
        raise ValueError("structured envelope message mismatch")
    target_connection = str(envelope.get("target_connection") or "").strip()
    target_profile = str(envelope.get("target_profile") or "").strip()
    target_handle = str(envelope.get("target_handle") or envelope.get("to_agent") or "").strip()
    if target_profile != resolved or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", target_connection):
        raise ValueError("structured envelope target identity mismatch")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", target_profile):
        raise ValueError("structured envelope has invalid target profile")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", target_handle):
        raise ValueError("structured envelope has invalid target handle")
    if envelope.get("target_handle") and str(envelope.get("target_handle")) != str(envelope.get("to_agent")):
        raise ValueError("structured envelope target handle mismatch")
    if str(envelope.get("type")) not in MESSAGE_TYPES:
        raise ValueError("structured envelope has invalid type")
    try:
        expires_at = float(envelope.get("expires_at") or 0)
    except (TypeError, ValueError):
        expires_at = 0
    if not math.isfinite(expires_at) or expires_at <= 0:
        raise ValueError("structured envelope has invalid expires_at")
    if enforce_expiry and expires_at <= time.time():
        raise _DeliveryContextError("structured envelope expired", code=4098, reason="queued_expired")
    scope = envelope.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("structured envelope has invalid scope")
    consequential = str(scope.get("mutation") or "none") != "none" or str(scope.get("production") or "none") != "none"
    if consequential and not (envelope.get("mission_id") and envelope.get("work_item_id")):
        raise _DeliveryContextError(
            "consequential message lacks mission/work binding",
            code=4099,
            reason="authority_missing",
        )
    if envelope.get("authority_effect") != "none":
        raise _DeliveryContextError(
            "bot messages cannot grant authority", code=4099, reason="authority_escalation"
        )
    return {
        "structured": True,
        "schema": ENVELOPE_SCHEMA,
        "envelope": envelope,
        "message_id": message_id,
        "idempotency_key": idempotency_key,
        "target_connection": target_connection,
        "target_profile": target_profile,
        "target_handle": target_handle,
        "fingerprint": delivery_fingerprint(
            envelope, target_profile=resolved, message=message, structured=True
        ),
    }


def _target_receipt_error(rid, outcome: dict, err) -> dict:
    reason = str(outcome.get("reason") or "target_receipt_unverified")
    code = {
        "target_receipt_missing": 4100,
        "target_receipt_pending": 4101,
        "target_receipt_mismatch": 4102,
    }.get(reason, 4102)
    return err(rid, code, reason, data={"reason": reason})


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
    """Deliver a relayed DM and return a durable target receipt for v2 envelopes."""
    import tempfile

    profile = str(params.get("profile") or "").strip()
    message = str(params.get("message") or "").strip()
    if not profile or not message:
        return _err(rid, 4090, "profile and message required")
    try:
        from tools.bot_mode_dm import MESSAGE_MAX_CHARS
        from tools.bot_relay import (
            acquire_turn_lock,
            begin_idempotent_delivery,
            cancel_idempotent_delivery,
            complete_idempotent_delivery,
        )

        if len(message) > MESSAGE_MAX_CHARS + 200:  # + attribution headroom
            return _err(rid, 4091, "message too long")
        root = _root()
        known = {"default"}
        if (root / "profiles").is_dir():
            known.update(c.name for c in (root / "profiles").iterdir() if c.is_dir())
        resolved = "default" if profile.lower() == "hermes" else profile
        if resolved not in known:
            return _err(rid, 4092, f"no profile '{profile}' on this gateway")
        try:
            context = _delivery_context(params, profile=profile, message=message, resolved=resolved)
        except ValueError as exc:
            return _err(
                rid,
                int(getattr(exc, "code", 4097)),
                str(exc),
                data={"reason": str(getattr(exc, "reason", "target_receipt_unverified"))},
            )

        structured = context["structured"]
        idempotency_key = context["idempotency_key"]
        message_id = context["message_id"]
        fingerprint = context["fingerprint"]

        def _admit_once(*, completion_reply=""):
            admission = begin_idempotent_delivery(
                root,
                idempotency_key,
                message_id,
                fingerprint,
                target_connection=context["target_connection"] if structured else "",
                target_profile=context["target_profile"] if structured else "",
                target_handle=context["target_handle"] if structured else "",
                completion_reply=completion_reply,
            )
            if admission.get("disposition") == "replay":
                result = {"reply": admission.get("reply", ""), "replayed": True}
                if structured:
                    result["target_receipt"] = admission.get("receipt")
                return _ok(rid, result)
            if admission.get("disposition") == "ambiguous":
                return _err(rid, 4095, "duplicate delivery has an ambiguous prior outcome", data={"reason": "duplicate_ambiguous"})
            if admission.get("disposition") == "conflict":
                return _err(rid, 4096, "idempotency key was already used for a different delivery", data={"reason": "idempotency_conflict"})
            return None

        # If THIS gateway already hosts the target Bot Chat live, prompt.submit is the
        # compositor choke point. Admission is still written before the queued turn.
        from tools.bot_mode_probe import BOT_CHAT_TITLE
        live_home = _profile_home(resolved)
        want_home = str(live_home) if live_home is not None else None
        live_sid, live_record = next((
            (candidate_sid, record) for candidate_sid, record in list(_sessions.items())
            if isinstance(record, dict) and (record.get("profile_home") or None) == want_home
            and _session_live_title(record, _session_lookup_key(record, fallback=candidate_sid)) == BOT_CHAT_TITLE
        ), ("", None))
        if live_sid:
            live_reply = f"Delivered into @{resolved}'s open Bot Chat; the reply will appear there."
            prior = _admit_once(completion_reply=live_reply)
            if prior is not None:
                return prior
            submit_params = {"session_id": live_sid, "text": message, "queued": True}
            if structured:
                # The existing restart-dedup field gives the background prompt a
                # non-secret correlation key. It lets the receipt wait for the exact
                # target row instead of accepting a same-text row from another turn.
                submit_params["_relay_message_id"] = message_id
            submitted = _methods["prompt.submit"](rid, submit_params)
            if "error" in submitted:
                cancel_idempotent_delivery(root, idempotency_key)
                return submitted
            if structured:
                persisted = False
                deadline = time.monotonic() + LIVE_TARGET_PERSIST_TIMEOUT_SECONDS
                try:
                    with _session_db(live_record) as db:
                        while True:
                            target_key = _session_lookup_key(live_record, fallback=live_sid)
                            if _live_target_message_persisted(db, target_key, message_id, message):
                                persisted = True
                                break
                            if time.monotonic() >= deadline:
                                break
                            time.sleep(min(LIVE_TARGET_PERSIST_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
                except Exception:
                    persisted = False
                if not persisted:
                    # The submit ACK proves only that this gateway accepted the
                    # background dispatch. Leave the started receipt in place: a
                    # later readback must remain pending rather than fabricating a
                    # completed target delivery.
                    return _target_receipt_error(
                        rid, {"reason": "target_receipt_pending"}, _err
                    )
            reply = live_reply
            receipt = complete_idempotent_delivery(
                root,
                idempotency_key,
                reply,
                target_connection=context["target_connection"] if structured else "",
                target_profile=context["target_profile"] if structured else "",
                target_handle=context["target_handle"] if structured else "",
            )
            result = {"reply": reply}
            if structured:
                result["target_receipt"] = receipt
            return _ok(rid, result)

        def _detail(proc) -> str:
            return (proc.stderr or proc.stdout or "").strip()[-500:]

        fd, tmp = tempfile.mkstemp(prefix="hermes-relay-dm-", suffix=".txt", text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(message)
            with acquire_turn_lock(root, resolved):
                prior = _admit_once()
                if prior is not None:
                    return prior
                proc = _run(resolved, tmp)
                if proc.returncode != 0:
                    from tools.bot_failure_reasons import RETRY_NONE, classify_agent_error, retry_action
                    if retry_action(classify_agent_error(_detail(proc))) != RETRY_NONE:
                        proc = _run(resolved, tmp)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        if proc.returncode != 0:
            from tools.bot_failure_reasons import classify_agent_error
            detail = _detail(proc)
            return _err(rid, 5092, f"delivery turn failed: {detail or proc.returncode}",
                        data={"reason": classify_agent_error(detail)})
        reply = (proc.stdout or "").strip()
        receipt = complete_idempotent_delivery(
            root,
            idempotency_key,
            reply,
            target_connection=context["target_connection"] if structured else "",
            target_profile=context["target_profile"] if structured else "",
            target_handle=context["target_handle"] if structured else "",
        )
        result = {"reply": reply}
        if structured:
            result["target_receipt"] = receipt
        return _ok(rid, result)
    except subprocess.TimeoutExpired:
        return _err(rid, 5093, "delivery turn timed out")
    except Exception as e:
        return _err(rid, 5096 if getattr(e, "reason", "") == "target_busy" else 5094, str(e))


@method("bot_relay.receipt.read")
def _(rid, params: dict, _root=_relay_root) -> dict:
    """Read back a completed target receipt without running another turn."""
    profile = str(params.get("profile") or "").strip()
    message = str(params.get("message") or "").strip()
    if not profile or not message:
        return _err(rid, 4090, "profile and message required")
    try:
        from tools.bot_relay import read_idempotent_delivery

        root = _root()
        known = {"default"}
        profiles_dir = root / "profiles"
        if profiles_dir.is_dir():
            known.update(c.name for c in profiles_dir.iterdir() if c.is_dir())
        resolved = "default" if profile.lower() == "hermes" else profile
        if resolved not in known:
            return _err(rid, 4092, f"no profile '{profile}' on this gateway")
        context = _delivery_context(
            params, profile=profile, message=message, resolved=resolved, enforce_expiry=False
        )
        if not context["structured"]:
            return _err(
                rid,
                4097,
                "structured envelope required for target receipt readback",
                data={"reason": "target_receipt_unverified"},
            )
        outcome = read_idempotent_delivery(
            root,
            context["idempotency_key"],
            message_id=context["message_id"],
            delivery_fingerprint=context["fingerprint"],
            target_connection=context["target_connection"],
            target_profile=context["target_profile"],
            target_handle=context["target_handle"],
        )
        if outcome.get("disposition") != "completed":
            return _target_receipt_error(rid, outcome, _err)
        return _ok(rid, {"receipt": outcome["receipt"]})
    except ValueError as exc:
        return _err(
            rid,
            int(getattr(exc, "code", 4097)),
            str(exc),
            data={"reason": str(getattr(exc, "reason", "target_receipt_unverified"))},
        )
    except Exception as exc:
        return _err(rid, 4102, str(exc), data={"reason": "target_receipt_unverified"})


@method("bot_relay.reply")
def _(rid, params: dict, _root=_relay_root) -> dict:
    """Write a relayed ``reply`` and/or ``error`` (+ optional typed ``reason``, see
    ``tools.bot_failure_reasons``) for envelope ``id`` so the sender-side waiter picks it up."""
    envelope_id = str(params.get("id") or "").strip()
    if not envelope_id:
        return _err(rid, 4093, "id required")
    try:
        from tools.bot_relay import write_reply
        write_reply(
            _root(),
            envelope_id,
            reply=str(params.get("reply") or ""),
            error=str(params.get("error") or ""),
            reason=str(params.get("reason") or ""),
            target_receipt=params.get("target_receipt"),
        )
        return _ok(rid, {"ok": True})
    except ValueError as e:
        return _err(rid, 4094, str(e))
    except Exception as e:
        return _err(rid, 5095, str(e))


def register(server) -> None:
    # HandlerRegistry rebinds handler globals onto server.py. Publish shared
    # helpers explicitly so the rebound handlers do not call an unbound
    # module-global name.
    server._delivery_context = _delivery_context
    server._target_receipt_error = _target_receipt_error
    server._live_target_message_persisted = _live_target_message_persisted
    server.LIVE_TARGET_PERSIST_TIMEOUT_SECONDS = LIVE_TARGET_PERSIST_TIMEOUT_SECONDS
    server.LIVE_TARGET_PERSIST_POLL_SECONDS = LIVE_TARGET_PERSIST_POLL_SECONDS
    _registry.install(server)
    from . import methods_groups
    server._LONG_HANDLERS = server._LONG_HANDLERS | methods_groups.LONG_HANDLERS
    for name in (
        "get_hosted_room_service", "_WORKER_UNAVAILABLE", "_profile_name", "_requested_profile",
        "_api_server_key", "_room_link_run_storage_durable"):
        setattr(server, name, getattr(methods_groups, name))
    methods_groups.bind_server(server)
    methods_groups.register(server)
