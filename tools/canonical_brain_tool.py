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
import os
import pathlib
import uuid
from typing import Any, Dict, Optional

try:
    from hermes_cli.config import load_config
except Exception:  # pragma: no cover - import-safe for tool discovery
    load_config = None  # type: ignore[assignment]

from tools.registry import registry, tool_error

from gateway.support_ops_team_registry import (
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    TeamMember,
    resolve_team_member,
)

CANONICAL_BRAIN_ROOT = pathlib.Path("/opt/adventico-ai-platform/canonical-brain")
CLOUD_SQL_HELPER = CANONICAL_BRAIN_ROOT / "bin" / "cloud_sql_synthetic_write_gate.py"
EVENT_TABLE = "canonical_event_log"
MAX_ROUTE_BACK_MESSAGE_CHARS = 1900
_ROUTE_BACK_RECEIPT_CAPABILITY = object()
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
    "person.alias.learned",
}
RECEIPT_REQUIRED_EVENT_TYPES = {"route_back.sent"}
FORBIDDEN_ROUTE_BACK_DM_KEYS = {
    "dm_channel_id",
    "direct_message_channel_id",
    "recipient_id",
    "dm_recipient_id",
}
FORBIDDEN_ROUTE_BACK_DM_VALUES = {"dm", "direct_message", "private_dm", "user_dm"}
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


def _has_structured_secret_value(value: Any) -> bool:
    """Return True when a secret-keyed field actually carries content.

    Operational safety metadata often uses boolean flags such as
    ``{"secret": False}`` or ``{"payment_credential": False}`` to state that
    no sensitive value is present. Treating the key alone as a secret caused
    harmless Canonical Brain appends to fail closed. Still block non-empty
    values under credential-shaped keys before any helper/connect.
    """
    if value is None or value is False:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, tuple, set, dict)) and not value:
        return False
    return True


def _contains_secret_like(value: Any) -> bool:
    """Return True for secret-looking values or structured secret keys.

    This is deliberately mechanical, not semantic. It recursively inspects
    dict/list payloads before any Cloud SQL helper load/connect so structured
    credentials such as {"token": "..."} are blocked even when the value does
    not contain marker strings like ``token=``.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if (
                _normalize_secret_key(key) in SECRET_KEY_NAMES
                and _has_structured_secret_value(nested)
            ):
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
    receipt_capability: object | None = None,
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
        if receipt_capability is not _ROUTE_BACK_RECEIPT_CAPABILITY:
            raise ValueError("route_back.sent may only be emitted by route_back_execute after an adapter receipt")
        receipt = payload.get("receipt") if isinstance(payload, dict) else None
        if not isinstance(receipt, dict) or not receipt.get("message_id"):
            raise ValueError("route_back.sent requires payload.receipt.message_id")
        _validate_no_route_back_dm_refs(payload)
    if event_type == "person.alias.learned":
        if not str(payload.get("alias") or "").strip() or not str(payload.get("member_key") or "").strip():
            raise ValueError("person.alias.learned requires payload.alias and payload.member_key")


def _contains_forbidden_dm_route_ref(value: Any) -> bool:
    """Return True when route-back target/receipt metadata points at a DM.

    Muncho's SkyVision policy allows public approved channels/threads only for
    team route-backs. A Discord user mention can still be used inside a public
    channel message, but DM channel ids or recipient-id delivery metadata must
    not be recorded as valid route_back.sent evidence.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = _normalize_secret_key(key)
            if normalized_key in FORBIDDEN_ROUTE_BACK_DM_KEYS and _has_structured_secret_value(nested):
                return True
            if normalized_key in {"channel_type", "target_type", "delivery_type", "lane", "role"}:
                normalized_value = str(nested or "").strip().casefold()
                if normalized_value in FORBIDDEN_ROUTE_BACK_DM_VALUES or normalized_value.endswith("_dm"):
                    return True
            if _contains_forbidden_dm_route_ref(nested):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_forbidden_dm_route_ref(item) for item in value)
    return False


def _validate_no_route_back_dm_refs(payload: Dict[str, Any]) -> None:
    route_back = payload.get("route_back") if isinstance(payload, dict) else None
    surfaces = [
        payload.get("target_ref"),
        payload.get("receipt"),
        route_back.get("target_ref") if isinstance(route_back, dict) else None,
        route_back.get("receipt") if isinstance(route_back, dict) else None,
    ]
    if any(_contains_forbidden_dm_route_ref(surface) for surface in surfaces if surface):
        raise ValueError("route_back.sent forbids direct-message/DM delivery receipts; use public approved channel/thread target or record_blocked")


def canonical_event_append_tool(
    event_type: str,
    case_id: str,
    summary: str,
    source_refs: Dict[str, Any],
    actors: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
    safety: Optional[Dict[str, Any]] = None,
    idempotency_key: Optional[str] = None,
    _receipt_capability: object | None = None,
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
            receipt_capability=_receipt_capability,
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
            {"label": "hermes_semantic_decision", "verified": False, "source_refs_hash": _hash(source_refs)[:16]}
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
        response = {
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
        }
        if event_type == "person.alias.learned":
            from gateway.support_ops_team_registry import learn_team_member_alias

            response["alias"] = learn_team_member_alias(
                str(payload.get("alias")),
                str(payload.get("member_key")),
            )
        return json.dumps(response, ensure_ascii=False, sort_keys=True)
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
    _receipt_capability: object | None = None,
) -> str:
    """Record route-back required/sent/blocked state.

    This tool does not infer meaning and does not secretly send Discord messages.
    It records the state Hermes decided or the delivery receipt Hermes obtained.
    """
    try:
        target_ref = _normalize_dict(target_ref, "target_ref")
        source_refs = _normalize_dict(source_refs, "source_refs")
        receipt = _normalize_dict(receipt, "receipt")
        allowed_modes = {"record_required_only", "queue_intent", "record_blocked"}
        if _receipt_capability is _ROUTE_BACK_RECEIPT_CAPABILITY:
            allowed_modes.add("record_sent_receipt")
        if mode not in allowed_modes:
            raise ValueError(f"mode_not_allowed:{mode}")
        if not target_ref.get("id") and not target_ref.get("mention") and not target_ref.get("lane"):
            raise ValueError("target_ref requires id, mention, or lane")
        if _contains_forbidden_dm_route_ref(target_ref) or _contains_forbidden_dm_route_ref(receipt):
            raise ValueError("route_back_state forbids direct-message/DM targets; use public approved channel/thread target or record_blocked")
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
        terminal_outcome = False
        required_next_step = "deliver_route_back_or_record_blocked"
        if mode == "record_sent_receipt":
            event_type = "route_back.sent"
            if not receipt.get("message_id"):
                raise ValueError("record_sent_receipt requires receipt.message_id")
            base_payload["receipt"] = receipt
            terminal_outcome = True
            required_next_step = "none"
        elif mode == "record_blocked":
            event_type = "route_back.blocked"
            if not blocker_reason:
                raise ValueError("record_blocked requires blocker_reason")
            terminal_outcome = True
            required_next_step = "none"
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
        result = canonical_event_append_tool(
            event_type=event_type,
            case_id=case_id,
            summary=message_summary,
            source_refs=source_refs,
            actors={"subject": {"type": "route_back", "id": target_ref.get("id") or target_ref.get("lane") or "target"}},
            payload=base_payload,
            safety={"contains_secret": False, "contains_payment_credential": False},
            idempotency_key=idempotency_key,
            _receipt_capability=_receipt_capability,
        )
        try:
            data = json.loads(result)
        except Exception:
            return result
        if isinstance(data, dict) and data.get("success"):
            data["route_back"] = {
                "mode": mode,
                "event_type": event_type,
                "terminal_outcome": terminal_outcome,
                "required_next_step": required_next_step,
            }
            if not terminal_outcome:
                data["route_back"]["final_answer_guard"] = (
                    "Do not present this as delivered or complete. Continue in the same turn "
                    "until the message is actually sent and record_sent_receipt is recorded, "
                    "or record_blocked is recorded with a concrete blocker."
                )
            return json.dumps(data, ensure_ascii=False, sort_keys=True)
        return result
    except Exception as exc:
        return tool_error(f"ROUTE_BACK_STATE_FAIL: {exc}")


def _resolve_route_back_public_target(target_ref: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve an exact approved public route-back target.

    This intentionally uses the team/channel registry, not business-keyword
    routing.  The first supported executor target is the owner lane: references
    to Emil/owner resolve to the public control-tower channel, never a DM.
    """
    candidate_values = []
    for key in ("id", "mention", "lane", "person", "target_person", "key"):
        value = str(target_ref.get(key) or "").strip()
        if value:
            candidate_values.append(value)

    channel_id = str(target_ref.get("channel_id") or target_ref.get("thread_id") or "").strip()
    raw_id = str(target_ref.get("id") or "").strip()
    if raw_id == SKYVISION_CONTROL_TOWER_CHANNEL_ID:
        channel_id = SKYVISION_CONTROL_TOWER_CHANNEL_ID

    resolved_member: TeamMember | None = None
    for value in candidate_values:
        resolution = resolve_team_member(value)
        if resolution.status == "resolved":
            resolved_member = resolution.member
            break
        if resolution.status == "ambiguous":
            raise ValueError("route_back_execute target_ref ambiguous; ask requester to clarify the public target")

    if resolved_member is not None:
        return {
            "channel_id": resolved_member.default_channel_id,
            "channel_type": "public_channel",
            "target_kind": "member_default_public_channel",
            "target_member_key": resolved_member.key,
            "target_member_id": resolved_member.discord_user_id,
            "target_mention": resolved_member.mention,
        }

    if channel_id == SKYVISION_CONTROL_TOWER_CHANNEL_ID:
        return {
            "channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
            "channel_type": "public_channel",
            "target_kind": "owner_public_channel",
            "target_member_key": "emil_lomliev",
            "target_member_id": "1279454038731264061",
            "target_mention": "<@1279454038731264061>",
        }

    if channel_id:
        from gateway.channel_directory import is_discord_public_target

        if not is_discord_public_target(channel_id):
            raise ValueError("route_back_execute requires a directory-confirmed public Discord channel/thread")
        return {
            "channel_id": channel_id,
            "channel_type": "public_channel_or_thread",
            "target_kind": "exact_public_directory_target",
            "target_member_key": None,
            "target_member_id": None,
            "target_mention": None,
        }

    raise ValueError("route_back_execute target is unresolved; ask the requester to clarify the public channel/thread")


def _discord_post_message(channel_id: str, content: str, *, timeout: int = 15) -> Dict[str, Any]:
    from gateway.config import Platform
    from gateway.run import _gateway_runner_ref
    from model_tools import _run_async

    runner = _gateway_runner_ref()
    adapter = runner.adapters.get(Platform.DISCORD) if runner is not None else None
    if adapter is None:
        raise RuntimeError("live_discord_adapter_unavailable")
    result = _run_async(adapter.send(str(channel_id), content))
    if not getattr(result, "success", False):
        raise RuntimeError(str(getattr(result, "error", None) or "discord_adapter_send_failed"))
    return {
        "id": str(getattr(result, "message_id", None) or ""),
        "channel_id": str(channel_id),
        "adapter_receipt": True,
    }


def _route_back_record_blocked(
    *,
    case_id: str,
    target_ref: Dict[str, Any],
    message_summary: str,
    source_refs: Dict[str, Any],
    blocker_reason: str,
    idempotency_key: Optional[str],
) -> Dict[str, Any]:
    result = route_back_tool(
        case_id=case_id,
        target_ref=target_ref,
        message_summary=message_summary,
        source_refs=source_refs,
        mode="record_blocked",
        blocker_reason=blocker_reason,
        idempotency_key=idempotency_key,
    )
    try:
        data = json.loads(result)
    except Exception:
        data = {"raw": result}
    return data if isinstance(data, dict) else {"raw": result}


def route_back_execute_tool(
    case_id: str,
    target_ref: Dict[str, Any],
    message: str,
    message_summary: str,
    source_refs: Dict[str, Any],
    idempotency_key: Optional[str] = None,
) -> str:
    """Deliver an exact public route-back and record the terminal outcome.

    This is the send+receipt executor counterpart to ``route_back_state``. It
    never sends DMs. If it cannot send to an approved public target, it records
    ``route_back.blocked`` instead of leaving the case in a pending state.
    """
    try:
        target_ref = _normalize_dict(target_ref, "target_ref")
        source_refs = _normalize_dict(source_refs, "source_refs")
        message = str(message or "").strip()
        message_summary = str(message_summary or "").strip() or message[:200]
        if not message:
            raise ValueError("message is required")
        if len(message) > MAX_ROUTE_BACK_MESSAGE_CHARS:
            blocked_target = {**target_ref, "id": target_ref.get("id") or "route_back_target"}
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=blocked_target,
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason=f"message_too_long:{len(message)}>{MAX_ROUTE_BACK_MESSAGE_CHARS}",
                idempotency_key=idempotency_key,
            )
            return json.dumps({
                "success": True,
                "status": "ROUTE_BACK_EXECUTE_BLOCKED",
                "blocker_reason": f"message_too_long:{len(message)}>{MAX_ROUTE_BACK_MESSAGE_CHARS}",
                "route_back_record": blocked,
            }, ensure_ascii=False, sort_keys=True)

        try:
            public_target = _resolve_route_back_public_target(target_ref)
        except Exception as exc:
            blocked_target = {
                **target_ref,
                "id": target_ref.get("id") or target_ref.get("mention") or target_ref.get("lane") or "unresolved_public_route_back_target",
            }
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=blocked_target,
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason=f"target_not_approved_or_unresolved:{type(exc).__name__}",
                idempotency_key=idempotency_key,
            )
            return json.dumps({
                "success": True,
                "status": "ROUTE_BACK_EXECUTE_BLOCKED",
                "blocker_reason": f"target_not_approved_or_unresolved:{type(exc).__name__}",
                "route_back_record": blocked,
            }, ensure_ascii=False, sort_keys=True)

        resolved_target_ref = {
            **target_ref,
            "id": target_ref.get("id") or public_target.get("target_member_id") or public_target["channel_id"],
            "mention": target_ref.get("mention") or public_target.get("target_mention"),
            "channel_id": public_target["channel_id"],
            "channel_type": public_target["channel_type"],
            "target_kind": public_target["target_kind"],
        }

        _block_secret_like_fields(
            target_ref=resolved_target_ref,
            message=message,
            message_summary=message_summary,
            source_refs=source_refs,
        )

        try:
            delivery = _discord_post_message(public_target["channel_id"], message)
        except Exception as exc:
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=resolved_target_ref,
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason=f"discord_send_failed:{type(exc).__name__}",
                idempotency_key=idempotency_key,
            )
            return json.dumps({
                "success": True,
                "status": "ROUTE_BACK_EXECUTE_BLOCKED",
                "blocker_reason": f"discord_send_failed:{type(exc).__name__}",
                "route_back_record": blocked,
            }, ensure_ascii=False, sort_keys=True)

        message_id = str(delivery.get("id") or "").strip() if isinstance(delivery, dict) else ""
        if not message_id:
            blocked = _route_back_record_blocked(
                case_id=case_id,
                target_ref=resolved_target_ref,
                message_summary=message_summary,
                source_refs=source_refs,
                blocker_reason="discord_send_missing_message_id_receipt",
                idempotency_key=idempotency_key,
            )
            return json.dumps({
                "success": True,
                "status": "ROUTE_BACK_EXECUTE_BLOCKED",
                "blocker_reason": "discord_send_missing_message_id_receipt",
                "route_back_record": blocked,
            }, ensure_ascii=False, sort_keys=True)

        receipt = {
            "platform": "discord",
            "message_id": message_id,
            "channel_id": public_target["channel_id"],
            "chat_id": public_target["channel_id"],
            "channel_type": public_target["channel_type"],
            "target_kind": public_target["target_kind"],
        }
        result = route_back_tool(
            case_id=case_id,
            target_ref=resolved_target_ref,
            message_summary=message_summary,
            source_refs=source_refs,
            mode="record_sent_receipt",
            receipt=receipt,
            idempotency_key=idempotency_key,
            _receipt_capability=_ROUTE_BACK_RECEIPT_CAPABILITY,
        )
        try:
            record_data = json.loads(result)
        except Exception:
            record_data = {"raw": result}
        if not isinstance(record_data, dict) or not record_data.get("success"):
            return json.dumps({
                "success": False,
                "status": "ROUTE_BACK_EXECUTE_SENT_BUT_RECEIPT_RECORD_FAILED",
                "receipt": receipt,
                "route_back_record": record_data,
                "final_answer_guard": (
                    "The public message was sent, but durable route_back.sent recording failed. "
                    "Do not resend. Report the receipt and the recording failure."
                ),
            }, ensure_ascii=False, sort_keys=True)

        return json.dumps({
            "success": True,
            "status": "ROUTE_BACK_EXECUTE_SENT",
            "receipt": receipt,
            "route_back_record": record_data,
        }, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return tool_error(f"ROUTE_BACK_EXECUTE_FAIL: {exc}")


def canonical_brain_query_tool(
    *,
    case_id: str = "",
    thread_id: str = "",
    limit: int = 80,
) -> str:
    """Read exact Canonical events and mechanically fold current case state."""
    try:
        case_id = str(case_id or "").strip()
        thread_id = str(thread_id or "").strip()
        if bool(case_id) == bool(thread_id):
            raise ValueError("provide exactly one of case_id or thread_id")
        if case_id and not case_id.startswith("case:"):
            raise ValueError("case_id must start with case:")
        limit = int(limit)
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")

        helper = _load_helper()
        password = helper.get_secret_value()
        try:
            sock = helper.connect(password)
            try:
                if case_id:
                    where = f"case_id = {helper.sql_quote(case_id)}"
                else:
                    ref = helper.sql_quote(thread_id)
                    where = f"""(
 source->'source_refs'->>'thread_id' = {ref}
 OR source->'source_refs'->>'chat_id' = {ref}
 OR payload->'route_back'->'target_ref'->>'thread_id' = {ref}
 OR payload->'route_back'->'target_ref'->>'channel_id' = {ref}
 OR payload->'route_back'->'receipt'->>'thread_id' = {ref}
 OR payload->'route_back'->'receipt'->>'channel_id' = {ref}
)"""
                sql = f"""
SELECT event_id::text, event_type, case_id, occurred_at::text,
       source, status, next_action, payload
FROM {EVENT_TABLE}
WHERE {where}
ORDER BY occurred_at DESC, event_id DESC
LIMIT {limit};
"""
                result = helper.query(sock, sql)
                rows = result.get("rows", []) if isinstance(result, dict) else []
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        finally:
            password = ""

        columns = [
            "event_id", "event_type", "case_id", "occurred_at",
            "source", "status", "next_action", "payload",
        ]
        normalized_rows = []
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict):
                normalized_rows.append(row)
            elif isinstance(row, (list, tuple)):
                normalized_rows.append(dict(zip(columns, row)))

        from gateway.canonical_brain_projection import fold_case_events

        cases = fold_case_events(normalized_rows)
        return json.dumps({
            "success": True,
            "status": "CANONICAL_BRAIN_QUERY_PASS",
            "query": {"case_id": case_id or None, "thread_id": thread_id or None, "limit": limit},
            "event_count": len(normalized_rows),
            "case_count": len(cases),
            "cases": cases,
        }, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return tool_error(f"CANONICAL_BRAIN_QUERY_FAIL: {exc}")


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
            "event_type": {"type": "string", "enum": sorted(ALLOWED_EVENT_TYPES - RECEIPT_REQUIRED_EVENT_TYPES)},
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
        "Record route-back required, queued, or blocked state in private/runtime Canonical Brain. "
        "This tool never records sent state. Use route_back_execute for atomic public send + attested receipt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "target_ref": {"type": "object", "description": "Target person/lane/mention/channel refs"},
            "message_summary": {"type": "string"},
            "source_refs": {"type": "object"},
            "mode": {"type": "string", "enum": ["record_required_only", "queue_intent", "record_blocked"], "default": "record_required_only"},
            "blocker_reason": {"type": "string", "description": "Required for record_blocked"},
            "idempotency_key": {"type": "string"},
        },
        "required": ["case_id", "target_ref", "message_summary", "source_refs"],
    },
}

ROUTE_BACK_EXECUTE_SCHEMA = {
    "name": "route_back_execute",
    "description": (
        "Execute an exact approved public route-back for private/runtime Canonical Brain cases. "
        "Use this when the route-back target is already known and is a directory-confirmed public "
        "Discord channel/thread or an exact registered teammate public lane. The tool sends first "
        "through the live adapter, then records route_back.sent with the real "
        "Discord receipt/message_id. If it cannot send safely, it records route_back.blocked and returns "
        "that terminal outcome instead of leaving route_back.required pending."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "target_ref": {"type": "object", "description": "Exact public target/member/lane/channel refs; no DM refs"},
            "message": {"type": "string", "description": "The public route-back message to send"},
            "message_summary": {"type": "string", "description": "Short durable summary of the route-back"},
            "source_refs": {"type": "object", "description": "Exact source refs: platform + message/thread/event/manual ref"},
            "idempotency_key": {"type": "string", "description": "Optional stable lifecycle idempotency key"},
        },
        "required": ["case_id", "target_ref", "message", "message_summary", "source_refs"],
    },
}

CANONICAL_BRAIN_QUERY_SCHEMA = {
    "name": "canonical_brain_query",
    "description": (
        "Read exact Canonical Brain events by case_id or Discord thread_id and return a mechanical "
        "current-state fold. No keyword search, classification, prioritization, or routing is performed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {"type": "string", "description": "Exact canonical case id"},
            "thread_id": {"type": "string", "description": "Exact Discord source/target thread id"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 80},
        },
    },
}

registry.register(
    name="canonical_brain_query",
    toolset="canonical_brain",
    schema=CANONICAL_BRAIN_QUERY_SCHEMA,
    handler=lambda args, **kw: canonical_brain_query_tool(
        case_id=args.get("case_id", ""),
        thread_id=args.get("thread_id", ""),
        limit=args.get("limit", 80),
    ),
    check_fn=check_canonical_brain_requirements,
    emoji="🧠",
)

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

registry.register(
    name="route_back_execute",
    toolset="canonical_brain",
    schema=ROUTE_BACK_EXECUTE_SCHEMA,
    handler=lambda args, **kw: route_back_execute_tool(
        case_id=args.get("case_id", ""),
        target_ref=args.get("target_ref") or {},
        message=args.get("message", ""),
        message_summary=args.get("message_summary", ""),
        source_refs=args.get("source_refs") or {},
        idempotency_key=args.get("idempotency_key"),
    ),
    check_fn=check_canonical_brain_requirements,
    emoji="📨",
)
