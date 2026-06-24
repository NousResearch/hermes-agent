#!/usr/bin/env python3
"""Canonical Brain tools for free-Hermes operational persistence.

These tools are intentionally thin mechanical adapters. They do not decide
business meaning. The Hermes agent decides when durable operational state
exists, then calls these tools to persist canonical events or route-back state.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import uuid
from typing import Any, Dict, Optional

try:
    from hermes_cli.config import load_config
except Exception:  # pragma: no cover - import-safe for tool discovery
    load_config = None  # type: ignore[assignment]

from tools.registry import registry, tool_error

CANONICAL_BRAIN_ROOT = pathlib.Path("/opt/adventico-ai-platform/canonical-brain")
CLOUD_SQL_HELPER = CANONICAL_BRAIN_ROOT / "bin" / "cloud_sql_synthetic_write_gate.py"
EVENT_TABLE = "canonical_event_log"
ALLOWED_EVENT_TYPES = {
    "case.note",
    "handoff.created",
    "handoff.waiting",
    "resolver.reply.received",
    "route_back.required",
    "route_back.intent.created",
    "route_back.sent",
    "route_back.blocked",
    "handoff.closed",
    "operational.note.needs_review",
    "semantic_interpreter.failed",
    "semantic_interpreter.skipped",
    "semantic_event.drafted",
}
RECEIPT_REQUIRED_EVENT_TYPES = {"route_back.sent"}
SECRET_MARKERS = (
    "api_key=", "apikey=", "token=", "password=", "secret=",
    "authorization: bearer", "private_key", "BEGIN PRIVATE KEY",
)
SECRET_KEY_NAMES = {
    "token",
    "access_token",
    "password",
    "secret",
    "api_key",
    "apikey",
    "private_key",
    "authorization",
    "bearer",
    "credentials",
    "payment_credential",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8", errors="replace")).hexdigest()


def _event_uuid(idempotency_key: str, event_type: str = "") -> str:
    """Deterministic event UUID scoped by event type + lifecycle key.

    The lifecycle idempotency key can intentionally be shared across
    route_back.required -> route_back.sent/blocked transitions, so event_type is
    part of the event UUID while the raw key remains in payload for grouping.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"canonical-brain:{event_type}:{idempotency_key}"))


def _load_helper() -> Any:
    if not CLOUD_SQL_HELPER.exists():
        raise RuntimeError("canonical brain Cloud SQL helper missing")
    spec = importlib.util.spec_from_file_location("canonical_brain_cloud_sql_helper", CLOUD_SQL_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load canonical brain Cloud SQL helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _normalize_secret_key(key: Any) -> str:
    return str(key or "").strip().casefold().replace("-", "_")


def _contains_secret_like(value: Any) -> bool:
    """Return True for secret-looking values or structured secret keys.

    This is deliberately mechanical, not semantic. It recursively inspects
    dict/list payloads before any Cloud SQL helper load/connect so structured
    credentials such as {"token": "..."} are blocked even when the value does
    not contain marker strings like ``token=``.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if _normalize_secret_key(key) in SECRET_KEY_NAMES:
                return True
            if _contains_secret_like(nested):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_secret_like(item) for item in value)
    text = str(value or "").casefold()
    return any(marker.casefold() in text for marker in SECRET_MARKERS)


def _block_secret_like_fields(**fields: Any) -> None:
    """Fail closed before any Cloud SQL helper/connect on secret-like content.

    Hermes decides operational meaning, but the adapter must mechanically ensure
    no secret-looking values are written into source/actor/payload/receipt/status
    surfaces.  Keep this broad and field-oriented rather than business-semantic.
    """
    for name, value in fields.items():
        if _contains_secret_like(value):
            raise ValueError(f"secret_like_content_blocked:{name}")


def _normalize_dict(value: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _get_session_env(name: str, default: str = "") -> str:
    """Read gateway session context without making tool discovery depend on it."""
    try:
        from gateway.session_context import get_session_env

        return str(get_session_env(name, default) or default)
    except Exception:
        return default


def _augment_source_refs_from_session_context(source_refs: Dict[str, Any]) -> Dict[str, Any]:
    """Fill mechanical source refs from the current gateway session when omitted.

    The model should pass exact refs when it has them. In live gateway runs the
    runtime already carries platform/chat/thread/message context; use that as a
    deterministic fallback before validation so operational appends do not fail
    merely because the model forgot to copy boilerplate refs into the tool call.
    """
    refs = dict(source_refs)
    changed = False

    env_fields = {
        "platform": "HERMES_SESSION_PLATFORM",
        "chat_id": "HERMES_SESSION_CHAT_ID",
        "thread_id": "HERMES_SESSION_THREAD_ID",
        "session_id": "HERMES_SESSION_ID",
        "session_key": "HERMES_SESSION_KEY",
        "user_id": "HERMES_SESSION_USER_ID",
        "user_name": "HERMES_SESSION_USER_NAME",
    }
    for ref_key, env_key in env_fields.items():
        if refs.get(ref_key):
            continue
        value = _get_session_env(env_key, "").strip()
        if value:
            refs[ref_key] = value
            changed = True

    if not (refs.get("message_id") or refs.get("event_ref") or refs.get("manual_ref")):
        message_id = _get_session_env("HERMES_SESSION_MESSAGE_ID", "").strip()
        if message_id:
            refs["message_id"] = message_id
            changed = True
        else:
            manual_parts = [
                str(refs.get("platform") or "").strip(),
                str(refs.get("chat_id") or "").strip(),
                str(refs.get("thread_id") or "").strip(),
                str(refs.get("session_id") or refs.get("session_key") or "").strip(),
            ]
            manual_ref = ":".join(part for part in manual_parts if part)
            if manual_ref:
                refs["manual_ref"] = f"hermes_session:{manual_ref}"
                changed = True

    if changed:
        refs.setdefault("source_ref_source", "hermes_session_context")
    return refs


def _normalize_list(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _validate_append_request(
    *,
    event_type: str,
    case_id: str,
    summary: str,
    source_refs: Dict[str, Any],
    actors: Dict[str, Any],
    payload: Dict[str, Any],
    safety: Dict[str, Any],
) -> None:
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"event_type_not_allowed:{event_type}")
    if not case_id or not str(case_id).startswith("case:"):
        raise ValueError("case_id must be present and start with case:")
    if not source_refs.get("platform"):
        raise ValueError("source_refs.platform is required")
    if not (source_refs.get("message_id") or source_refs.get("event_ref") or source_refs.get("manual_ref")):
        raise ValueError("source_refs requires message_id, event_ref, or manual_ref")
    if bool(safety.get("contains_secret")) or bool(safety.get("contains_payment_credential")):
        raise ValueError("safety flags block append")
    _block_secret_like_fields(
        summary=summary,
        source_refs=source_refs,
        actors=actors,
        payload=payload,
        safety=safety,
    )
    if event_type in RECEIPT_REQUIRED_EVENT_TYPES:
        receipt = payload.get("receipt") if isinstance(payload, dict) else None
        if not isinstance(receipt, dict) or not receipt.get("message_id"):
            raise ValueError("route_back.sent requires payload.receipt.message_id")


def canonical_event_append_tool(
    event_type: str,
    case_id: str,
    summary: str,
    source_refs: Dict[str, Any],
    actors: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    safety: Optional[Dict[str, Any]] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Append one canonical operational event to Cloud SQL.

    The caller (Hermes) decides meaning. This function validates mechanics and
    writes a deterministic/idempotent event row.
    """
    try:
        source_refs = _normalize_dict(source_refs, "source_refs")
        source_refs = _augment_source_refs_from_session_context(source_refs)
        actors = _normalize_dict(actors, "actors")
        payload = _normalize_dict(payload, "payload")
        safety = _normalize_dict(safety, "safety")
        _validate_append_request(
            event_type=event_type,
            case_id=case_id,
            summary=summary,
            source_refs=source_refs,
            actors=actors,
            payload=payload,
            safety=safety,
        )
        if not idempotency_key:
            idempotency_key = f"{case_id}:{event_type}:{_hash({'source_refs': source_refs, 'payload': payload})[:24]}"
        event_id = _event_uuid(idempotency_key, event_type)
        occurred_at = _utc_now()
        source = {
            "system": "hermes_agent",
            "component": "canonical_brain_tool",
            "source_refs": source_refs,
        }
        actor = actors.get("actor") or {"type": "agent", "id": "hermes"}
        subject = actors.get("subject") or {"type": "case", "id": case_id}
        evidence = _normalize_list(payload.get("evidence"), "payload.evidence") if isinstance(payload.get("evidence"), list) else [
            {"label": "hermes_semantic_decision", "verified": True, "source_refs_hash": _hash(source_refs)[:16]}
        ]
        decision = {
            "kind": "hermes_semantic_operational_persistence",
            "decided_by": "hermes_agent_llm_reasoning",
            "keyword_authority": False,
        }
        status = {"state": event_type, "summary": str(summary or "")[:500]}
        next_action = payload.get("next_action") if isinstance(payload.get("next_action"), dict) else {}
        safety_doc = {
            "secret_value_recorded": False,
            "payment_credential_recorded": False,
            "business_mutation": False,
            "outbound": bool(payload.get("outbound", False)),
            **safety,
        }
        clean_payload = {**payload, "idempotency_key": idempotency_key, "summary": summary}
        _block_secret_like_fields(
            summary=summary,
            source_refs=source_refs,
            actors=actors,
            payload=payload,
            safety=safety,
            next_action=next_action,
            clean_payload=clean_payload,
        )
        helper = _load_helper()
        password = helper.get_secret_value()
        try:
            sock = helper.connect(password)
            try:
                sql = f"""
INSERT INTO {EVENT_TABLE} (
  event_id, schema_version, event_type, occurred_at, case_id,
  source, actor, subject, evidence, decision, status, next_action, safety, payload
) VALUES (
  {helper.sql_quote(event_id)}::uuid,
  'canonical_event.v1',
  {helper.sql_quote(event_type)},
  {helper.sql_quote(occurred_at)}::timestamptz,
  {helper.sql_quote(case_id)},
  {helper.json_sql(source)},
  {helper.json_sql(actor)},
  {helper.json_sql(subject)},
  {helper.json_sql(evidence)},
  {helper.json_sql(decision)},
  {helper.json_sql(status)},
  {helper.json_sql(next_action)},
  {helper.json_sql(safety_doc)},
  {helper.json_sql(clean_payload)}
)
ON CONFLICT (event_id) DO NOTHING;
"""
                tag = helper.query(sock, sql)["command_tag"]
                readback = helper.query(sock, f"""
SELECT event_id::text, event_type, case_id, occurred_at::text, payload->>'idempotency_key'
FROM {EVENT_TABLE}
WHERE event_id = {helper.sql_quote(event_id)}::uuid
LIMIT 1;
""")["rows"]
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        finally:
            password = ""
        return json.dumps({
            "success": True,
            "status": "CANONICAL_EVENT_APPEND_PASS",
            "event_id": event_id,
            "event_type": event_type,
            "case_id": case_id,
            "idempotency_key": idempotency_key,
            "command_tag": tag,
            "readback": readback,
            "inserted": tag == "INSERT 0 1",
            "deduped": tag == "INSERT 0 0",
        }, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return tool_error(f"CANONICAL_EVENT_APPEND_FAIL: {exc}")


def route_back_tool(
    case_id: str,
    target_ref: Dict[str, Any],
    message_summary: str,
    source_refs: Dict[str, Any],
    mode: str = "record_required_only",
    receipt: Optional[Dict[str, Any]] = None,
    blocker_reason: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Record route-back required/sent/blocked state.

    This tool does not infer meaning and does not secretly send Discord messages.
    It records the state Hermes decided or the delivery receipt Hermes obtained.
    """
    try:
        target_ref = _normalize_dict(target_ref, "target_ref")
        source_refs = _normalize_dict(source_refs, "source_refs")
        receipt = _normalize_dict(receipt, "receipt")
        allowed_modes = {"record_required_only", "queue_intent", "record_sent_receipt", "record_blocked"}
        if mode not in allowed_modes:
            raise ValueError(f"mode_not_allowed:{mode}")
        if not target_ref.get("id") and not target_ref.get("mention") and not target_ref.get("lane"):
            raise ValueError("target_ref requires id, mention, or lane")
        base_payload = {
            "route_back": {
                "target_ref": target_ref,
                "mode": mode,
                "message_summary": message_summary,
                "receipt": receipt or None,
                "blocker_reason": blocker_reason,
            },
            "next_action": {"kind": "deliver_route_back_or_record_receipt", "target_ref": target_ref},
        }
        if mode == "record_sent_receipt":
            event_type = "route_back.sent"
            if not receipt.get("message_id"):
                raise ValueError("record_sent_receipt requires receipt.message_id")
            base_payload["receipt"] = receipt
        elif mode == "record_blocked":
            event_type = "route_back.blocked"
            if not blocker_reason:
                raise ValueError("record_blocked requires blocker_reason")
        elif mode == "queue_intent":
            event_type = "route_back.intent.created"
        else:
            event_type = "route_back.required"
        _block_secret_like_fields(
            target_ref=target_ref,
            receipt=receipt,
            blocker_reason=blocker_reason,
            message_summary=message_summary,
            next_action=base_payload.get("next_action"),
            clean_payload=base_payload,
        )
        return canonical_event_append_tool(
            event_type=event_type,
            case_id=case_id,
            summary=message_summary,
            source_refs=source_refs,
            actors={"subject": {"type": "route_back", "id": target_ref.get("id") or target_ref.get("lane") or "target"}},
            payload=base_payload,
            safety={"contains_secret": False, "contains_payment_credential": False},
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        return tool_error(f"ROUTE_BACK_STATE_FAIL: {exc}")


def check_canonical_brain_requirements() -> bool:
    """Expose Canonical Brain tools only for explicit private/runtime installs.

    This is not an upstream-generic tool surface: it requires the private Cloud
    SQL helper and an explicit profile config enablement under
    ``canonical_brain.audit_bridge.enabled`` or ``canonical_brain.tools_enabled``.
    """
    if not CLOUD_SQL_HELPER.exists():
        return False
    if load_config is None:
        return False
    try:
        cfg = load_config() or {}
    except Exception:
        return False
    cb = cfg.get("canonical_brain") if isinstance(cfg, dict) else None
    if not isinstance(cb, dict):
        return False
    audit = cb.get("audit_bridge")
    return bool(cb.get("tools_enabled") or (isinstance(audit, dict) and audit.get("enabled")))


CANONICAL_EVENT_APPEND_SCHEMA = {
    "name": "canonical_event_append",
    "description": (
        "Append a durable operational event to a private/runtime Canonical Brain Cloud SQL. "
        "Use when Hermes has reasoned that durable state exists (case note, handoff, "
        "route_back.required/blocked, needs_review, resolver reply, etc.). This tool "
        "does NOT decide meaning; Hermes decides. Do not use keyword matching as authority. "
        "route_back.sent requires a real delivery receipt/message_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": sorted(ALLOWED_EVENT_TYPES)},
            "case_id": {"type": "string", "description": "Canonical case id, must start with case:"},
            "summary": {"type": "string", "description": "Short operational summary"},
            "source_refs": {"type": "object", "description": "Exact source refs: platform + message/thread/event/manual ref"},
            "actors": {"type": "object", "description": "Optional actor/subject/requester/target refs"},
            "payload": {"type": "object", "description": "Event payload; no secrets/payment credentials"},
            "safety": {"type": "object", "description": "Safety flags; contains_secret/payment_credential block append"},
            "idempotency_key": {"type": "string", "description": "Optional stable idempotency key"},
        },
        "required": ["event_type", "case_id", "summary", "source_refs"],
    },
}

ROUTE_BACK_SCHEMA = {
    "name": "route_back_state",
    "description": (
        "Record route-back state in private/runtime Canonical Brain after Hermes decides a target notification "
        "is required, queued, sent, or blocked. This tool does not secretly send messages. "
        "Use record_sent_receipt only after a real delivery result with message_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "target_ref": {"type": "object", "description": "Target person/lane/mention/channel refs"},
            "message_summary": {"type": "string"},
            "source_refs": {"type": "object"},
            "mode": {"type": "string", "enum": ["record_required_only", "queue_intent", "record_sent_receipt", "record_blocked"], "default": "record_required_only"},
            "receipt": {"type": "object", "description": "Delivery receipt; required for record_sent_receipt"},
            "blocker_reason": {"type": "string", "description": "Required for record_blocked"},
            "idempotency_key": {"type": "string"},
        },
        "required": ["case_id", "target_ref", "message_summary", "source_refs"],
    },
}

registry.register(
    name="canonical_event_append",
    toolset="canonical_brain",
    schema=CANONICAL_EVENT_APPEND_SCHEMA,
    handler=lambda args, **kw: canonical_event_append_tool(
        event_type=args.get("event_type", ""),
        case_id=args.get("case_id", ""),
        summary=args.get("summary", ""),
        source_refs=args.get("source_refs") or {},
        actors=args.get("actors"),
        payload=args.get("payload"),
        safety=args.get("safety"),
        idempotency_key=args.get("idempotency_key"),
    ),
    check_fn=check_canonical_brain_requirements,
    emoji="🧠",
)

registry.register(
    name="route_back_state",
    toolset="canonical_brain",
    schema=ROUTE_BACK_SCHEMA,
    handler=lambda args, **kw: route_back_tool(
        case_id=args.get("case_id", ""),
        target_ref=args.get("target_ref") or {},
        message_summary=args.get("message_summary", ""),
        source_refs=args.get("source_refs") or {},
        mode=args.get("mode", "record_required_only"),
        receipt=args.get("receipt"),
        blocker_reason=args.get("blocker_reason"),
        idempotency_key=args.get("idempotency_key"),
    ),
    check_fn=check_canonical_brain_requirements,
    emoji="📨",
)
