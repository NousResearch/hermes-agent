"""Bot Mode cross-connection relay — connections ARE the peer set.

Gateway-side half of the relay letting agents on ANY Desktop-connected gateway
message agents on ANY other. Plain file plumbing under ``<root>/bot_relay/`` —
no network; the Desktop owns every socket: ``roster.json`` (union roster of
agents on OTHER connections, pushed via ``bot_relay.roster.sync``), ``outbox/``
(envelopes queued by ``message_agent``, drained via ``bot_relay.outbox.drain``),
``replies/`` (one JSON per envelope via ``bot_relay.reply``; a waiter spawned at
send time watches it so the reply wakes the sender like a local DM).
Public helpers never raise, except ``enqueue_envelope`` → ``EnvelopeRefusedError``
when the target is definitively offline (fail fast instead of queueing a DM nobody will drain).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from tools.bot_mode_probe import _default_home, _hermes_root

logger = logging.getLogger(__name__)

RELAY_DIR_NAME = "bot_relay"
ROSTER_FILE = "roster.json"
OUTBOX_DIR = "outbox"
CLAIMED_DIR = "claimed"
REPLIES_DIR = "replies"
LOCKS_DIR = "locks"
DELIVERED_DIR = "delivered"

ENVELOPE_SCHEMA = "asm-hermes-a2a-envelope/v2"
TARGET_RECEIPT_SCHEMA = "asm-hermes-a2a-target-receipt/v1"
MESSAGE_TYPES = frozenset({"REQUEST", "RESPONSE", "HANDOFF", "BLOCKER", "REVIEW", "DECISION", "FYI"})

# Config fallbacks (real knobs: ``bot_mode.turn_wait_seconds`` / ``bot_mode.envelope_ttl_seconds``).
TURN_WAIT_SECONDS_FALLBACK = 120
DEFAULT_ENVELOPE_TTL_SECONDS = 900  # older envelopes are refused at drain with 'queued_expired'
DELIVERY_RECEIPT_RETENTION_SECONDS = 7 * 24 * 3600
# Waiter give-up budget: cross-connection turns can be slow — generous, but bounded.
REPLY_WAIT_SECONDS = 900
# Envelopes/replies older than this are stale artifacts (Desktop closed) and are swept.
STALE_AFTER_SECONDS = 6 * 3600
# Only a recent roster is authoritative for the fail-fast offline check: the
# Desktop re-pushes roster.sync on connection-state changes.
ROSTER_FRESH_SECONDS = 600


class EnvelopeRefusedError(RuntimeError):
    """``enqueue_envelope`` refused to queue (nothing written); ``reason`` is a stable machine code.

    ``reason`` is a stable machine code; ``str(exc)`` is the human text. 'runtime_offline' matches the
    #93091 item-1 failure-reason enum (plain literal here so the branches merge cleanly).
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# Profile names, handles and connection ids share one shape (also the local
# ``message_agent`` target grammar in ``tools/bot_mode_dm.py``).
_HANDLE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# One turn in a profile's canonical Bot Chat: ``hermes -p <profile> *BOT_CHAT_TURN_ARGS``.
# ``-c "Bot Chat"`` must match ``bot_mode_probe.BOT_CHAT_TITLE``.
BOT_CHAT_TURN_ARGS = ("chat", "--in", "~", "-c", "Bot Chat", "--create-if-missing", "-Q")


def relay_root(root: Path | str) -> Path:
    return Path(root) / RELAY_DIR_NAME


def relay_install_root(home: Path | str) -> Path:
    """Normalize a profile home to the install-wide relay state owner."""
    path = Path(home)
    return path.parent.parent if path.parent.name == "profiles" else path


def _ensure_dirs(root: Path | str) -> Path:
    base = relay_root(root)
    for sub in (OUTBOX_DIR, CLAIMED_DIR, REPLIES_DIR, DELIVERED_DIR):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def _atomic_write_json(target: Path, payload: Any, *, prefix: str, sort_keys: bool = False) -> None:
    """tempfile + os.replace so readers never see a partial file; tempfile removed on failure."""
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=sort_keys)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _bot_mode_cfg(key: str, *, loader: str) -> Any:
    """``bot_mode.<key>`` from config, read lazily (tools/ must not import CLI
    config at import time); None when absent or the config is unreadable."""
    try:
        import hermes_cli.config as cfgmod

        cfg = getattr(cfgmod, loader)() or {}
        return (cfg.get("bot_mode") or {}).get(key)
    except Exception:
        logger.debug("bot_mode.%s config read failed", key, exc_info=True)
        return None


def _normalize_roster_row(row: Any) -> Optional[dict]:
    """Validated, minimal roster row or None. Rows come from the Desktop over
    RPC — treat as untrusted input."""
    if not isinstance(row, dict):
        return None
    profile = str(row.get("profile") or "").strip()
    handle = str(row.get("handle") or "").strip().lstrip("@") or ("hermes" if profile == "default" else profile)
    connection_id = str(row.get("connection_id") or "").strip()
    if not profile or not connection_id or not all(_HANDLE_RE.match(v) for v in (handle, profile, connection_id)):
        return None
    out = {
        "profile": profile, "handle": handle, "connection_id": connection_id,
        "connection_label": str(row.get("connection_label") or "").strip()[:80],
        "title": str(row.get("title") or "").strip()[:120],
        "description": " ".join(str(row.get("description") or "").split())[:160],
    }
    # Liveness kept only when a real bool: absent == unknown == fail-open on enqueue.
    if isinstance(row.get("online"), bool):
        out["online"] = row["online"]
    return out


def write_remote_roster(root: Path | str, rows: Any) -> int:
    """Atomically persist the Desktop-pushed remote roster. Returns count."""
    base = _ensure_dirs(root)
    by_key: dict[tuple[str, str], dict] = {}
    for norm in filter(None, map(_normalize_roster_row, rows if isinstance(rows, list) else [])):
        by_key.setdefault((norm["connection_id"], norm["profile"]), norm)
    cleaned = [by_key[k] for k in sorted(by_key)]
    _atomic_write_json(base / ROSTER_FILE, {"updated_at": int(time.time()), "agents": cleaned},
                       prefix=".roster-", sort_keys=True)
    return len(cleaned)


def read_remote_roster(root: Path | str) -> list[dict]:
    """The current remote roster (possibly empty). Never raises."""
    try:
        data = json.loads((relay_root(root) / ROSTER_FILE).read_text(encoding="utf-8"))
        agents = data.get("agents") if isinstance(data, dict) else None
        return [r for r in map(_normalize_roster_row, agents) if r] if isinstance(agents, list) else []
    except FileNotFoundError:
        return []
    except Exception:
        logger.debug("bot_relay roster read failed", exc_info=True)
        return []


def resolve_remote_target(raw_target: str, roster: list[dict]) -> Any:
    """Matched row for a bare handle/profile (unique across connections) or
    ``<handle|profile>@<connection-id>``; ``"ambiguous"`` for a bare form on several connections; None otherwise."""
    want, at, conn = (p.strip() for p in str(raw_target or "").strip().lstrip("@").partition("@"))
    if not want or (at and not conn):
        return None
    matches = [row for row in roster if want.lower() in (row["handle"].lower(), row["profile"].lower())
               and (not conn or row["connection_id"].lower() == conn.lower())]
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else "ambiguous"


def remote_target_forms(roster: list[dict]) -> list[str]:
    """Target strings: bare handle when unique across connections, else
    ``handle@connection`` (mirrors ``resolve_remote_target``)."""
    handles = [row["handle"].lower() for row in roster]
    return [f"{row['handle']}@{row['connection_id']}" if handles.count(h) > 1 else row["handle"]
            for row, h in zip(roster, handles)]


def _envelope_ttl_seconds() -> int:
    """Configured drain TTL (``bot_mode.envelope_ttl_seconds``), read per-drain.
    ``0`` (or negative) disables expiry."""
    val = _bot_mode_cfg("envelope_ttl_seconds", loader="load_config_readonly")
    return DEFAULT_ENVELOPE_TTL_SECONDS if val is None else int(val)


def _target_liveness(root: Path | str, target: dict) -> Optional[bool]:
    """Tri-state liveness: True / False / None (unknown → callers fail open). Offline =
    explicit ``online: false`` or ABSENT from a *fresh* roster; a missing, unreadable,
    empty or stale roster proves nothing → None. Never raises."""
    try:
        try:
            age = time.time() - (relay_root(root) / ROSTER_FILE).stat().st_mtime
        except OSError:
            return None
        roster = read_remote_roster(root) if age <= ROSTER_FRESH_SECONDS else []
        if not roster:
            return None
        key = (str(target.get("connection_id") or ""), str(target.get("profile") or ""))
        row = next((r for r in roster if (r["connection_id"], r["profile"]) == key), None)
        if row is None:
            return False  # fresh roster no longer lists the target — offline
        return row["online"] if isinstance(row.get("online"), bool) else None
    except Exception:
        logger.debug("bot_relay liveness check failed", exc_info=True)
        return None


def enqueue_envelope(
    root: Path | str,
    *,
    target: dict,
    message: str,
    sender_profile: str,
    sender_handle: str,
    metadata: Optional[dict] = None,
) -> dict:
    """Queue a cross-connection DM for the Desktop relay; returns the envelope. Raises
    ``EnvelopeRefusedError`` ('runtime_offline') without writing when the target is
    definitively offline; unknown liveness enqueues (fail-open)."""
    if _target_liveness(root, target) is False:
        label = (f"@{target.get('handle') or target.get('profile') or '?'} on "
                 f"{target.get('connection_label') or target.get('connection_id') or '?'}")
        raise EnvelopeRefusedError("runtime_offline", f"{label} is offline right now — the message was NOT queued. "
                                   "Try again once that machine reconnects to the Desktop.")
    base = _ensure_dirs(root)
    now = int(time.time())
    envelope_id = uuid.uuid4().hex
    meta = metadata if isinstance(metadata, dict) else {}
    message_type = str(meta.get("type") or "REQUEST").upper()
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"invalid message type: {message_type}")
    mutation_scope = str(meta.get("mutation_scope") or "none").strip()[:120]
    production_scope = str(meta.get("production_scope") or "none").strip()[:120]
    mission_id = str(meta.get("mission_id") or "").strip()[:160]
    work_item_id = str(meta.get("work_item_id") or "").strip()[:160]
    if (mutation_scope != "none" or production_scope != "none") and not (mission_id and work_item_id):
        raise ValueError("mutation/production messages require mission_id and work_item_id")
    ttl = max(1, min(int(meta.get("ttl_seconds") or _envelope_ttl_seconds()), 86400))
    evidence_refs = meta.get("evidence_refs") if isinstance(meta.get("evidence_refs"), list) else []
    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "id": envelope_id,
        "message_id": envelope_id,
        "idempotency_key": str(meta.get("idempotency_key") or f"dm:{envelope_id}")[:240],
        "type": message_type,
        "created_at": now,
        "expires_at": now + ttl,
        "response_due": now + ttl,
        "from_profile": sender_profile,
        "from_handle": sender_handle,
        "from_agent": sender_handle,
        "target_connection": target["connection_id"],
        "target_profile": target["profile"],
        "target_handle": target["handle"],
        "to_agent": target["handle"],
        "mission_id": mission_id,
        "work_item_id": work_item_id,
        "subject": str(meta.get("subject") or str(message).splitlines()[0])[:160],
        "request": {
            "question": str(message),
            "required_output": [
                str(value)[:80]
                for value in (meta.get("required_output") or ["response", "evidence", "unknowns"])
            ][:12],
        },
        "scope": {"mutation": mutation_scope, "production": production_scope},
        "evidence_refs": [str(value)[:500] for value in evidence_refs][:20],
        "authority_effect": "none",
        "message": message,
    }
    _atomic_write_json(base / OUTBOX_DIR / f"{envelope['id']}.json", envelope, prefix=".env-")
    return envelope


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def delivery_fingerprint(
    envelope: Optional[dict], *, target_profile: str, message: str, structured: bool
) -> str:
    """Canonical semantic delivery input shared by admission and readback."""
    env = envelope if isinstance(envelope, dict) else {}
    semantic = {
        "target_profile": str(target_profile),
        "message": str(message),
        "type": env.get("type") if structured else "legacy",
        "from_agent": env.get("from_agent") if structured else "",
        "to_agent": env.get("to_agent") if structured else str(target_profile),
        "mission_id": env.get("mission_id") if structured else "",
        "work_item_id": env.get("work_item_id") if structured else "",
        "request": env.get("request") if structured else {},
        "scope": env.get("scope") if structured else {},
        "authority_effect": env.get("authority_effect") if structured else "none",
    }
    return _canonical_json(semantic)


def _target_identity(connection: str, profile: str, handle: str = "") -> dict[str, str]:
    return {
        "target_connection": str(connection or "").strip(),
        "target_profile": str(profile or "").strip(),
        "target_handle": str(handle or "").strip(),
    }


def _target_identity_sha256(identity: dict[str, str]) -> str:
    return _sha256_text(_canonical_json(identity))


def _public_target_receipt(payload: dict) -> dict:
    fields = (
        "schema", "status", "idempotency_sha256", "message_id", "delivery_sha256",
        "target_sha256", "target_connection", "target_profile", "target_handle",
        "started_at", "completed_at", "reply_sha256",
    )
    return {field: payload[field] for field in fields if field in payload}


def _receipt_shape_error(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "receipt is not an object"
    required = (
        "schema", "status", "idempotency_sha256", "message_id", "delivery_sha256",
        "target_sha256", "target_connection", "target_profile", "target_handle", "started_at",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        return f"receipt missing: {', '.join(missing)}"
    if payload.get("schema") != TARGET_RECEIPT_SCHEMA:
        return "receipt schema is unsupported"
    if payload.get("status") not in {"started", "completed"}:
        return "receipt status is invalid"
    if not all(isinstance(payload.get(field), str) for field in required):
        return "receipt identity fields are invalid"
    if any(not payload[field].strip() for field in required):
        return "receipt fields are incomplete"
    if not re.fullmatch(r"[0-9a-f]{64}", payload["idempotency_sha256"]):
        return "receipt idempotency hash is invalid"
    if not re.fullmatch(r"[0-9a-f]{64}", payload["delivery_sha256"]):
        return "receipt delivery hash is invalid"
    if not re.fullmatch(r"[0-9a-f]{64}", payload["target_sha256"]):
        return "receipt target hash is invalid"
    if not payload["target_connection"] or not payload["target_profile"]:
        return "receipt target identity is incomplete"
    identity = _target_identity(
        payload["target_connection"], payload["target_profile"], payload["target_handle"]
    )
    if payload["target_sha256"] != _target_identity_sha256(identity):
        return "receipt target hash does not match identity"
    if payload["status"] == "completed":
        for field in ("completed_at", "reply_sha256"):
            if not isinstance(payload.get(field), str) or not payload[field]:
                return f"completed receipt missing {field}"
        if not re.fullmatch(r"[0-9a-f]{64}", payload["reply_sha256"]):
            return "receipt reply hash is invalid"
    return ""


def delivery_receipt_path(root: Path | str, idempotency_key: str) -> Path:
    """Stable, non-secret filename for one target-side delivery decision."""
    digest = _sha256_text(str(idempotency_key).strip())
    return _ensure_dirs(root) / DELIVERED_DIR / f"{digest}.json"


def begin_idempotent_delivery(
    root: Path | str,
    idempotency_key: str,
    message_id: str,
    delivery_fingerprint: str = "",
    *,
    target_connection: str = "",
    target_profile: str = "",
    target_handle: str = "",
    completion_reply: str = "",
) -> dict:
    """Atomically admit one delivery or return its durable disposition."""
    key = str(idempotency_key or "").strip()
    if not key:
        return {"disposition": "untracked"}
    path = delivery_receipt_path(root, key)
    identity = _target_identity(target_connection, target_profile, target_handle)
    payload = {
        "schema": TARGET_RECEIPT_SCHEMA,
        "idempotency_sha256": path.stem,
        "message_id": str(message_id or "")[:160],
        "delivery_sha256": _sha256_text(delivery_fingerprint),
        **identity,
        "target_sha256": _target_identity_sha256(identity),
        "status": "started",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **({"_completion_reply": str(completion_reply)} if completion_reply else {}),
    }
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"disposition": "ambiguous", "status": "unreadable"}
        if prior.get("idempotency_sha256") not in (None, path.stem):
            return {"disposition": "ambiguous", "status": "invalid"}
        if prior.get("delivery_sha256") != payload["delivery_sha256"] or prior.get("message_id") != payload["message_id"]:
            return {"disposition": "conflict"}
        if any(identity.values()):
            if _receipt_shape_error(prior) or any(prior.get(k) != v for k, v in identity.items()):
                return {"disposition": "conflict"}
        if prior.get("status") == "completed":
            return {
                "disposition": "replay",
                "reply": "Duplicate suppressed: this message was already delivered.",
                "receipt": _public_target_receipt(prior),
            }
        return {"disposition": "ambiguous", "status": str(prior.get("status") or "unknown")}
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return {"disposition": "admitted", "path": str(path), "receipt": _public_target_receipt(payload)}


def complete_idempotent_delivery(
    root: Path | str,
    idempotency_key: str,
    reply: str,
    *,
    target_connection: str = "",
    target_profile: str = "",
    target_handle: str = "",
) -> dict:
    """Atomically mark an admitted delivery complete and return its receipt."""
    key = str(idempotency_key or "").strip()
    if not key:
        return {}
    path = delivery_receipt_path(root, key)

    return _complete_idempotent_delivery_path(
        path,
        reply,
        target_connection=target_connection,
        target_profile=target_profile,
        target_handle=target_handle,
    )


def _complete_idempotent_delivery_path(
    path: Path,
    reply: str,
    *,
    target_connection: str = "",
    target_profile: str = "",
    target_handle: str = "",
) -> dict:
    """Complete one receipt when its hashed idempotency path is already known."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = _target_identity(target_connection, target_profile, target_handle)
    if _receipt_shape_error(payload) and any(identity.values()):
        raise ValueError("cannot complete an unverified target receipt")
    if any(identity.values()) and any(payload.get(k) != v for k, v in identity.items()):
        raise ValueError("target receipt identity changed before completion")
    if payload.get("status") == "completed":
        return _public_target_receipt(payload)
    if any(identity.values()) and payload.get("status") != "started":
        raise ValueError("cannot complete a target receipt in its current state")
    payload.update({
        "schema": TARGET_RECEIPT_SCHEMA,
        **({k: v for k, v in identity.items()} if any(identity.values()) else {}),
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reply_sha256": _sha256_text(str(reply or "")),
    })
    _atomic_write_json(path, payload, prefix=".delivery-", sort_keys=True)
    return _public_target_receipt(payload)


def complete_pending_deliveries_for_message(root: Path | str, message_id: str) -> list[dict]:
    """Complete delayed live receipts after their exact user row is durable.

    A live ``prompt.submit`` can acknowledge before its queued turn reaches the
    SessionDB. The turn-start persistence path calls this helper with the
    platform message id, so a timeout does not strand the receipt in ``started``
    forever. The durable row is the proof; only receipts explicitly admitted
    with a completion reply are eligible.
    """
    wanted = str(message_id or "").strip()
    if not wanted:
        return []
    directory = relay_root(root) / DELIVERED_DIR
    if not directory.is_dir():
        return []
    completed: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "started":
            continue
        if str(payload.get("message_id") or "") != wanted:
            continue
        key = str(payload.get("idempotency_sha256") or "")
        reply = payload.get("_completion_reply")
        if key != path.stem or not isinstance(reply, str) or not reply:
            continue
        try:
            completed.append(
                _complete_idempotent_delivery_path(
                    path,
                    reply,
                    target_connection=str(payload.get("target_connection") or ""),
                    target_profile=str(payload.get("target_profile") or ""),
                    target_handle=str(payload.get("target_handle") or ""),
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            # A concurrent normal completion, malformed receipt, or teardown
            # race is safe to leave for the normal readback disposition.
            continue
    return completed


def cancel_idempotent_delivery(root: Path | str, idempotency_key: str) -> None:
    """Remove a started receipt only when the target turn definitely did not run."""
    key = str(idempotency_key or "").strip()
    if not key:
        return
    path = delivery_receipt_path(root, key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "started":
            path.unlink()
    except FileNotFoundError:
        pass


def read_idempotent_delivery(
    root: Path | str,
    idempotency_key: str,
    *,
    message_id: str = "",
    delivery_fingerprint: str = "",
    target_connection: str = "",
    target_profile: str = "",
    target_handle: str = "",
) -> dict:
    """Read and independently verify a completed target receipt."""
    key = str(idempotency_key or "").strip()
    if not key:
        return {"disposition": "invalid", "reason": "idempotency_key_missing"}
    path = delivery_receipt_path(root, key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"disposition": "missing", "reason": "target_receipt_missing"}
    except Exception:
        return {"disposition": "invalid", "reason": "target_receipt_unreadable"}
    shape_error = _receipt_shape_error(payload)
    if shape_error:
        return {"disposition": "invalid", "reason": "target_receipt_invalid", "detail": shape_error}
    if payload.get("idempotency_sha256") != path.stem:
        return {"disposition": "mismatch", "reason": "target_receipt_mismatch"}
    if payload.get("status") != "completed":
        return {"disposition": "pending", "reason": "target_receipt_pending"}
    if message_id and payload.get("message_id") != str(message_id):
        return {"disposition": "mismatch", "reason": "target_receipt_mismatch"}
    if delivery_fingerprint and payload.get("delivery_sha256") != _sha256_text(delivery_fingerprint):
        return {"disposition": "mismatch", "reason": "target_receipt_mismatch"}
    identity = _target_identity(target_connection, target_profile, target_handle)
    if any(identity.values()) and any(payload.get(k) != v for k, v in identity.items()):
        return {"disposition": "mismatch", "reason": "target_receipt_mismatch"}
    return {"disposition": "completed", "receipt": _public_target_receipt(payload)}


def _expire_if_stale(root: Path | str, path: Path, ttl: float, now: float) -> bool:
    """True when the outbox envelope is older than ``ttl``; writes the 'queued_expired'
    reply so the sender's waiter resolves (best effort). Unreadable envelopes are left for the claim."""
    try:
        env = json.loads(path.read_text(encoding="utf-8"))
        created = float(env.get("created_at") or path.stat().st_mtime)
        explicit_expiry = float(env.get("expires_at") or 0)
    except (OSError, ValueError):
        return False
    expired = (explicit_expiry > 0 and now >= explicit_expiry) or (ttl > 0 and now - created > ttl)
    if not expired:
        return False
    with contextlib.suppress(OSError, ValueError):
        write_reply(root, str(env.get("id") or ""), reason="queued_expired", error=(
            f"queued message to @{env.get('target_handle') or '?'} on {env.get('target_connection') or '?'} "
            f"expired after {ttl}s waiting for the Desktop to drain it — it was NOT delivered. "
            "Resend once the Desktop reconnects."))
    return True


def claim_pending_envelopes(root: Path | str) -> list[dict]:
    """Drain the outbox (rename → claimed/ so a second drain can't double-deliver).
    TTL-expired envelopes get a 'queued_expired' reply and are removed instead.

    Envelopes older than ``bot_mode.envelope_ttl_seconds`` are NOT delivered: each gets an error reply
    (reason ``'queued_expired'``) so the sender's waiter resolves, and its outbox file is removed (#93091
    item 2).
    """
    base = _ensure_dirs(root)
    _sweep_stale(base)
    ttl = _envelope_ttl_seconds()
    now = time.time()
    out: list[dict] = []
    for path in sorted((base / OUTBOX_DIR).glob("*.json")):
        if _expire_if_stale(root, path, ttl, now):
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        claimed = base / CLAIMED_DIR / path.name
        with contextlib.suppress(OSError, ValueError):
            os.replace(path, claimed)  # atomic claim
            out.append(json.loads(claimed.read_text(encoding="utf-8")))
    return out


def write_reply(
    root: Path | str,
    envelope_id: str,
    *,
    reply: str = "",
    error: str = "",
    reason: str = "",
    target_receipt: Optional[dict] = None,
) -> Path:
    """Persist the relayed reply (or delivery error) for the waiter. ``reason`` (typed
    code, ``tools.bot_failure_reasons``) is classified from ``error`` when omitted."""
    base = _ensure_dirs(root)
    safe = str(envelope_id or "").strip()
    if not re.match(r"^[0-9a-f]{32}$", safe):
        raise ValueError(f"invalid envelope id: {envelope_id!r}")
    err, code = str(error or ""), str(reason or "")
    if not code and err:
        from tools.bot_failure_reasons import classify_agent_error

        code = classify_agent_error(err)
    payload = {"id": safe, "at": int(time.time()), "reply": str(reply or ""), "error": err, "reason": code}
    if target_receipt is not None:
        shape_error = _receipt_shape_error(target_receipt)
        if shape_error or target_receipt.get("status") != "completed":
            raise ValueError(f"invalid target receipt: {shape_error or 'not completed'}")
        claimed = base / CLAIMED_DIR / f"{safe}.json"
        try:
            envelope = json.loads(claimed.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            envelope = None
        if not isinstance(envelope, dict) or envelope.get("schema") != ENVELOPE_SCHEMA:
            raise ValueError("target receipt requires its claimed structured envelope")
        expected_key = _sha256_text(str(envelope.get("idempotency_key") or "").strip())
        if target_receipt.get("idempotency_sha256") != expected_key:
            raise ValueError("target receipt idempotency does not match envelope")
        for field in ("message_id", "target_connection", "target_profile", "target_handle"):
            if str(target_receipt.get(field) or "") != str(envelope.get(field) or ""):
                raise ValueError(f"target receipt {field} does not match envelope")
        expected_delivery = _sha256_text(
            delivery_fingerprint(
                envelope,
                target_profile=str(envelope.get("target_profile") or ""),
                message=str(envelope.get("message") or ""),
                structured=True,
            )
        )
        if target_receipt.get("delivery_sha256") != expected_delivery:
            raise ValueError("target receipt delivery does not match envelope")
        payload["target_receipt"] = _public_target_receipt(target_receipt)
    path = base / REPLIES_DIR / f"{safe}.json"
    _atomic_write_json(path, payload, prefix=".rep-", sort_keys=True)
    return path


def unlink_files_older_than(directory: Path, pattern: str, cutoff: float) -> int:
    """Unlink regular files matching ``pattern`` with mtime before ``cutoff``; returns count. Never raises."""
    removed = 0
    with contextlib.suppress(OSError):
        for path in directory.glob(pattern):
            with contextlib.suppress(OSError):
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
    return removed


def _sweep_stale(base: Path, *, now: float | None = None) -> int:
    cutoff = (time.time() if now is None else now) - STALE_AFTER_SECONDS
    removed = sum(unlink_files_older_than(base / sub, "*.json", cutoff) for sub in (CLAIMED_DIR, REPLIES_DIR, OUTBOX_DIR))
    receipt_cutoff = (time.time() if now is None else now) - DELIVERY_RECEIPT_RETENTION_SECONDS
    return removed + unlink_files_older_than(base / DELIVERED_DIR, "*.json", receipt_cutoff)


def cleanup_bot_relay_artifacts(max_age_hours: float | None = None) -> int:
    """Hourly sweep of stale relay artifacts (DM plaintext; ``_sweep_stale`` otherwise runs
    only on Desktop drains). ``max_age_hours`` is for ``cleanup_*_cache`` signature parity only."""
    del max_age_hours
    try:
        base = relay_root(_hermes_root(Path(_default_home())))
        return _sweep_stale(base) if base.is_dir() else 0
    except Exception:
        logger.debug("bot_relay artifact sweep failed", exc_info=True)
        return 0


def waiter_command(root: Path | str, envelope: dict) -> str:
    """Shell command that blocks until the reply file appears, then prints it."""
    reply_path = str(relay_root(root) / REPLIES_DIR / f"{envelope['id']}.json")
    label = f"@{envelope.get('target_handle', '')} on {envelope.get('target_connection', '')}"
    return shlex.join([
        sys.executable or "python3", "-m", "tools.bot_relay", "wait", reply_path,
        label, str(REPLY_WAIT_SECONDS),
    ])


def wait_for_reply(reply_path: Path | str, label: str, timeout_seconds: float) -> int:
    """Print one relay reply for the process-completion notification path."""
    path = Path(reply_path)
    timeout = max(0.0, float(timeout_seconds))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("error"):
                reason = str(data.get("reason") or "").strip()
                tag = f" [reason: {reason}]" if reason else ""
                print(f"Delivery to {label} failed{tag}: {data['error']}")
                return 1
            print(f"Reply from {label}:")
            print(data.get("reply") or "(empty reply)")
            return 0
        time.sleep(0.25)
    print(
        f"No reply from {label} within {timeout:g}s. The message may still be delivered when the Desktop reconnects; "
        "do not resend blindly."
    )
    return 1


def _main(argv: list[str]) -> int:
    if len(argv) != 5 or argv[1] != "wait":
        return 2
    try:
        timeout = float(argv[4])
    except ValueError:
        return 2
    return wait_for_reply(argv[2], argv[3], timeout)


def _hermes_cli() -> str:
    """hermes CLI beside this interpreter, then ``shutil.which``, then the bare name
    (service contexts lack PATH, so a bare "hermes" died with ENOENT).

    The deliver RPC runs on the target gateway, whose process is the venv python — its bin/Scripts directory
    holds the matching ``hermes`` entrypoint. A bare ``"hermes"`` relies on PATH, which is exactly what
    service contexts (systemd units, desktop launchers, non-login SSH shells) do not provide, so delivery
    died with ENOENT there (#93590). When no sibling exists (e.g. running from a source tree without an
    installed script), a ``shutil.which`` lookup runs next — it honors whatever PATH the process does have —
    before falling back to the bare name, preserving today's behavior for interactive shells.
    """
    sibling = Path(sys.executable or "").parent / ("hermes.exe" if sys.platform == "win32" else "hermes")
    return str(sibling) if sibling.is_file() else shutil.which("hermes") or "hermes"


def local_delivery_command(profile: str, query_file: str) -> list[str]:
    """argv that delivers a DM into ``profile``'s Bot Chat on THIS gateway."""
    return [_hermes_cli(), "-p", profile, *BOT_CHAT_TURN_ARGS, "--query-file", query_file]


# Two deliveries into the SAME profile must never run Bot Chat turns concurrently.
# Deliveries are separate ``hermes`` subprocesses, so the lock is a per-profile
# lockfile under ``<root>/bot_relay/locks/`` held with ``fcntl.flock`` for exactly
# the turn window; the kernel releases it on fd close (incl. process death), so a
# crashed turn can never wedge the profile.


# ── per-profile turn lock (#93091) ─────────────────────────────────────────── Two deliveries into the SAME
# target profile must never run their Bot Chat turns concurrently: deliveries spawn separate ``hermes``
# subprocesses, so an in-memory mutex is useless — the lock is a per-profile lockfile under
# ``<root>/bot_relay/locks/`` held with ``fcntl.flock`` for exactly the turn execution window. flock is
# released by the kernel when the holder's fd closes (including process death), so a crashed turn can never
# wedge the profile. A queued delivery waits up to ``bot_mode.turn_wait_seconds`` and then fails with a
# structured 'target_busy' refusal instead of blocking forever.
class TurnBusyError(RuntimeError):
    """A delivery turn is already running for the target profile (``waited_seconds`` ≈ time queued).

    ``reason`` is 'target_busy' — extends the #93091 item-1 structured refusal enum. ``waited_seconds`` is
    roughly how long the caller queued behind the current turn before giving up.
    """

    reason = "target_busy"

    def __init__(self, profile: str, waited_seconds: float):
        self.profile, self.waited_seconds = profile, waited_seconds
        super().__init__(f"target_busy: another delivery turn is already running for profile '{profile}' — "
                         f"queued behind it for ~{int(round(waited_seconds))}s without it finishing. "
                         "The message was NOT delivered; retry shortly.")


def turn_wait_seconds() -> float:
    """Wait budget for a queued delivery turn (config, lazily read)."""
    val = _bot_mode_cfg("turn_wait_seconds", loader="load_config")
    return float(TURN_WAIT_SECONDS_FALLBACK) if val is None else max(0.0, float(val))


def turn_lock_path(root: Path | str, profile: str) -> Path:
    """Per-profile lockfile path (short — safe on macOS temp roots)."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(profile or ""))[:64] or "_"
    return relay_root(root) / LOCKS_DIR / f"{safe}.lock"


@contextlib.contextmanager
def acquire_turn_lock(root: Path | str, profile: str, timeout_seconds: float | None = None) -> Iterator[Path]:
    """Hold ``profile``'s cross-process turn lock for the ``with`` body: non-blocking
    flock probe + short-sleep retry up to the budget (``bot_mode.turn_wait_seconds``
    unless ``timeout_seconds``); raises :class:`TurnBusyError` when exhausted. No
    ordering among waiters, but every waiter is bounded. Without ``fcntl`` (Windows)
    the lock is a no-op — those installs never had this race path."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover — Windows
        logger.debug("bot turn lock disabled: fcntl unavailable on this platform")
        yield turn_lock_path(root, profile)
        return

    budget = turn_wait_seconds() if timeout_seconds is None else max(0.0, float(timeout_seconds))
    path = turn_lock_path(root, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        start = time.monotonic()
        deadline = start + budget
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                now = time.monotonic()
                if now >= deadline:
                    raise TurnBusyError(profile, now - start)
                time.sleep(min(0.1, max(0.005, deadline - now)))
        try:
            yield path
        finally:
            with contextlib.suppress(OSError):  # kernel releases on close anyway
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


if __name__ == "__main__":  # pragma: no cover - exercised by waiter process
    raise SystemExit(_main(sys.argv))
