"""CLI fallback tickets in the existing Bot Chat mailbox, not a second queue.

Only proven pre-turn refusals may return to queued. A durable claim is never
leased/reclaimed: a crash between claim and target receipt is uncertain, not
permission to repeat a potentially mutating turn. Local same-user filesystem
trust applies; this is not a peer/remote execution API.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

from tools import bot_live_delivery as mailbox

log = logging.getLogger(__name__)
TICKET_ENV = "HERMES_BOT_CLI_TICKET"
MAX_PENDING = 128
MAX_BUSY_ATTEMPTS = 12
MAX_TTL_SECONDS = 86400
RUNNER_WAIT_SECONDS = 120


def payload_digest(home, sender_home, target_profile, message, author):
    payload = dict(profile_home=str(Path(home).resolve()), sender_home=str(Path(sender_home).resolve()),
                   target_profile=target_profile, message=message, author=author)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def admit(home, *, sender_home, target_profile, message, author, delivery_id, dm_file):
    """Durably admit before spawning; immutable id binds both identities and input."""
    from tools.bot_relay import _HANDLE_RE, _envelope_ttl_seconds

    key = mailbox._delivery_id(delivery_id)
    if not _HANDLE_RE.fullmatch(target_profile) or not isinstance(message, str) or len(message) > 16000 + 256:
        raise ValueError("invalid local delivery payload")
    if not isinstance(author, dict) or author.get("is_bot") is not True or not author.get("id"):
        raise ValueError("local delivery requires a bound bot author")
    digest = payload_digest(home, sender_home, target_profile, message, author)
    with mailbox._locked(home) as root:
        path = root / f"{key}.json"
        existing = mailbox._read(path)
        if existing is not None:
            if existing.get("transport") != "cli" or existing.get("payload_digest") != digest:
                raise ValueError("delivery id already belongs to a different payload")
            return existing
        records = [mailbox._read(p) for p in root.glob("*.json")]
        if sum(r is not None and r.get("transport") == "cli" and r.get("status") in ("queued", "claimed", "ambiguous")
               for r in records) >= MAX_PENDING:
            raise ValueError("local delivery mailbox is full")
        ttl = min(MAX_TTL_SECONDS, max(1, _envelope_ttl_seconds()))
        now = time.time()
        record = dict(id=key, delivery_id=key, transport="cli", status="queued",
                      profile_home=str(Path(home).resolve()), sender_home=str(Path(sender_home).resolve()),
                      target_profile=target_profile, author=dict(author), message=message,
                      payload_digest=digest, dm_file=str(Path(dm_file).resolve()),
                      created_at=time.time_ns(), sequence=mailbox._next_sequence(root),
                      expires_at=now + ttl, retry_after=now, attempts=0)
        mailbox._write(path, record)
        return record


def _read_ticket(root, key):
    record = mailbox._read(root / f"{mailbox._delivery_id(key)}.json")
    if record is None or record.get("transport") != "cli":
        raise ValueError("local delivery ticket not found")
    return record


def _finish(root, record, status, *, reason, error="", reply="", session_id=""):
    record.update(status=status, reason=reason, error=error, reply=reply[:18000],
                  session_id=session_id, completed_at=time.time_ns(), message="")
    mailbox._write(root / f"{record['delivery_id']}.json", record)
    return record


def claim(home, key):
    """Caller holds the real profile turn lock. FIFO includes unresolved claims."""
    now = time.time()
    with mailbox._locked(home) as root:
        record = _read_ticket(root, key)
        if record["status"] != "queued":
            return record, False
        # Fail closed on corrupt mail, rather than skip an uninspectable older input.
        pending = [r for p in root.glob("*.json") if (r := mailbox._read(p)) is not None
                   and r.get("transport") == "cli" and r.get("status") in ("queued", "claimed", "ambiguous")]
        for row in pending:
            if row["status"] == "queued" and now >= row["expires_at"]:
                _finish(root, row, "cancelled", reason="queued_expired", error="Queued input expired before execution")
        pending = [r for r in pending if r["status"] in ("queued", "claimed", "ambiguous")]
        record = _read_ticket(root, key)
        if record["status"] != "queued":
            return record, False
        head = min(pending, key=lambda r: (r["sequence"], r["delivery_id"]))
        if head["delivery_id"] != key or now < record["retry_after"]:
            return record, False
        record.update(status="claimed", claimed_at=time.time_ns(), claim_token=uuid.uuid4().hex,
                      attempts=record["attempts"] + 1)
        mailbox._write(root / f"{key}.json", record)
        return record, True


def defer(home, key, token=None):
    """Return only a proven not-started attempt to queued, with bounded backoff."""
    with mailbox._locked(home) as root:
        record = _read_ticket(root, key)
        if token is None:
            if record["status"] != "queued":
                return record
        elif record["status"] != "claimed" or record.get("claim_token") != token:
            return record
        if record.get("target_started_at"):
            raise ValueError("admitted target cannot be requeued")
        attempts = record["attempts"] if token else record["attempts"] + 1
        if time.time() >= record["expires_at"] or attempts >= MAX_BUSY_ATTEMPTS:
            return _finish(root, record, "cancelled", reason="queued_expired", error="Busy delivery budget exhausted before execution")
        record.update(status="queued", reason="target_busy", attempts=attempts,
                      retry_after=time.time() + min(30, 2 ** min(attempts, 5)))
        record.pop("claim_token", None)
        mailbox._write(root / f"{key}.json", record)
        return record


def take_target_ticket(query, author, environ=None, *, session_id=""):
    """Consume the claim capability BEFORE tools run; bind to actual target/input."""
    from hermes_constants import get_hermes_home

    env = os.environ if environ is None else environ
    raw = env.pop(TICKET_ENV, None)
    if not raw:
        return None
    context = json.loads(raw)
    home = str(Path(get_hermes_home()).resolve())
    with mailbox._locked(home) as root:
        record = _read_ticket(root, context["delivery_id"])
        if (record["status"] != "claimed" or record.get("claim_token") != context.get("claim_token")
                or record["profile_home"] != home or record["message"] != query or record["author"] != author
                or record["payload_digest"] != payload_digest(home, record["sender_home"], record["target_profile"], query, author)):
            raise ValueError("local delivery claim does not match target turn")
        if record.get("target_started_at"):
            raise ValueError("target already admitted this input; no replay allowed")
        from hermes_state import SessionDB

        db = SessionDB(db_path=Path(home) / "state.db", read_only=True)
        try:
            canonical = db.get_session_by_title("Bot Chat")
            if not canonical or db.get_compression_tip(canonical["id"]) != session_id:
                raise ValueError("local delivery requires the actual canonical target session")
        finally:
            db.close()
        record.update(target_started_at=time.time_ns(), target_session_id=session_id, target_pid=os.getpid())
        mailbox._write(root / f"{record['delivery_id']}.json", record)
    return home, context


def complete_target(ticket, *, exit_code, session_id, reply="", error=""):
    """The target writes its own immutable receipt, not the dispatcher's stdout."""
    if ticket is None:
        return None
    if not session_id:
        raise ValueError("target receipt requires an actual session")
    home, context = ticket
    with mailbox._locked(home) as root:
        record = _read_ticket(root, context["delivery_id"])
        if record.get("claim_token") != context["claim_token"]:
            raise ValueError("target receipt claim mismatch")
        if record.get("target_pid") != os.getpid():
            raise ValueError("target receipt belongs to another consumer")
        if record["status"] in ("settled", "failed"):
            return record
        if record["status"] not in ("claimed", "ambiguous"):
            raise ValueError("target receipt requires a claim")
        from hermes_state import SessionDB
        from gateway.response_filters import is_intentional_silence_response

        db = SessionDB(db_path=Path(home) / "state.db", read_only=True)
        try:
            if db.get_compression_tip(record["target_session_id"]) != session_id or db.get_session(session_id) is None:
                raise ValueError("target receipt does not name its persisted session")
        finally:
            db.close()
        if exit_code == 0 and is_intentional_silence_response(reply):
            reply = ""
        # An uncertain dispatcher may learn a later authoritative outcome; never re-execute.
        record["receipt"] = dict(delivery_id=record["delivery_id"], payload_digest=record["payload_digest"],
                                 profile_home=record["profile_home"], session_id=session_id,
                                 claim_token=context["claim_token"], exit_code=int(exit_code))
        return _finish(root, record, "settled" if exit_code == 0 else "failed", reason="" if exit_code == 0 else "target_turn_failed",
                       reply=reply, error="Target turn failed; it will not be replayed" if exit_code else "", session_id=session_id)


def uncertain(home, key, token):
    with mailbox._locked(home) as root:
        record = _read_ticket(root, key)
        if record["status"] == "claimed" and record.get("claim_token") == token:
            # Retain the input as evidence, and block later inputs until an authoritative receipt arrives.
            record.update(status="ambiguous", reason="outcome_unknown", error="Target outcome is uncertain. Do not resend.")
            mailbox._write(root / f"{key}.json", record)
        return record


def verified_result(home, key, claimed):
    record = mailbox.read_delivery_result(home, key)
    receipt = (record or {}).get("receipt") or {}
    if (record is not None and record.get("status") in ("settled", "failed")
            and all(receipt.get(k) == claimed.get(k) for k in ("delivery_id", "payload_digest", "profile_home", "claim_token"))
            and receipt.get("session_id") == record.get("session_id") and receipt.get("session_id")
            and type(receipt.get("exit_code")) is int
            and (receipt["exit_code"] == 0) == (record["status"] == "settled")):
        return record
    return None


def public_result(record):
    """No payload, path, capability or provider error may escape in status/logging."""
    keys = ("status", "delivery_id", "sequence", "expires_at", "retry_after", "attempts", "reason", "reply", "error")
    result = {k: record[k] for k in keys if k in record}
    if record["status"] in ("queued", "claimed", "ambiguous"):
        result["detail"] = "Durable input retained. Do not resend. Uncertain claims are never automatically replayed."
    return result


def run_ticket(home, key, argv, dm_file, author):
    """One bounded drain attempt; restart/reinvoke inspects the same durable id."""
    from hermes_cli.quiet_single_query import run_reported_turn
    from tools.bot_mode_probe import _hermes_root
    from tools.bot_relay import BOT_CHAT_TURN_ARGS, TURN_ATTEMPT_TIMEOUT_SECONDS, TurnBusyError, acquire_turn_lock, delivery_env
    from utils import atomic_write_text

    record = mailbox.read_delivery_result(home, key)
    if record is None or record.get("transport") != "cli":
        raise ValueError("local ticket missing")
    expected = ["-p", record["target_profile"], *BOT_CHAT_TURN_ARGS]
    if (argv[1:] != expected or str(Path(home).resolve()) != record["profile_home"]
            or author != record["author"] or str(Path(dm_file).resolve()) != record["dm_file"]):
        raise ValueError("runner does not match pinned local delivery")
    if record["status"] in ("settled", "failed"):
        if verified_result(home, key, record) is None:
            raise ValueError("terminal local receipt is unverified; no replay allowed")
        return record
    if record["status"] != "queued" or time.time() < record["retry_after"]:
        return record
    claimed = None
    try:
        with acquire_turn_lock(_hermes_root(Path(home)), record["target_profile"], timeout_seconds=0):
            claimed, started = claim(home, key)
            if not started:
                return claimed
            # The temp payload may have been swept or the machine restarted. Rebuild only from durable input.
            atomic_write_text(dm_file, claimed["message"], mode=0o600, fsync_dir=True)
            env = delivery_env(author, home)
            env[TICKET_ENV] = json.dumps(dict(delivery_id=key, claim_token=claimed["claim_token"]))
            # No policy re-run: a failed model turn may already have mutated external state.
            proc = run_reported_turn([*argv, "--query-file", dm_file], env=env, report_path=dm_file + ".turn.json",
                                     timeout=TURN_ATTEMPT_TIMEOUT_SECONDS, encoding="utf-8",
                                     cwd=str(Path(__file__).resolve().parents[1]))
            receipt = verified_result(home, key, claimed)
            if receipt is not None:
                return receipt
            # A typed CLI refusal is emitted before turn admission; prose and exit code alone are not proof.
            reasons = [line.removeprefix("hermes-refusal-reason: ").strip() for line in (proc.stderr or "").splitlines()
                       if line.startswith("hermes-refusal-reason: ")]
            current = mailbox.read_delivery_result(home, key) or {}
            if proc.returncode != 0 and reasons == ["SESSION_NOT_OWNED"] and not current.get("target_started_at"):
                return defer(home, key, claimed["claim_token"])
            return uncertain(home, key, claimed["claim_token"])
    except TurnBusyError:
        return defer(home, key)
    except Exception:
        log.warning("local delivery attempt failed (%s); outcome retained", key)
        if claimed is not None and claimed.get("claim_token"):
            return uncertain(home, key, claimed["claim_token"])
        raise


def has_pending(home):
    """Keep later local DMs behind CLI input even when a live owner appears."""
    if not mailbox.has_mailbox(home):
        return False
    with mailbox._locked(home) as root:
        return any(r.get("transport") == "cli" and r.get("status") in ("queued", "claimed", "ambiguous")
                   for p in root.glob("*.json") if (r := mailbox._read(p)) is not None)


def drain(home, key, argv, dm_file, author):
    """Retry safely queued inputs within one bounded runner window; no new daemon.

    A later invocation of this SAME command/key can resume queued input after a
    runner exits. It may inspect, but never replay, an uncertain durable claim.
    """
    deadline = time.monotonic() + RUNNER_WAIT_SECONDS
    while True:
        record = run_ticket(home, key, argv, dm_file, author)
        if record["status"] != "queued" or time.monotonic() >= deadline:
            return record
        delay = max(0.1, min(1, record["retry_after"] - time.time()))
        time.sleep(min(delay, max(0, deadline - time.monotonic())))
