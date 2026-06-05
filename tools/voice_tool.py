"""Agent-native voice/telephony tools with Vapi as the first adapter.

The local Voice Core tables are the canonical ledger for assistants, phone
numbers, calls, and provider webhook events.  Vapi is treated as an external
provider adapter reached through REST with credentials supplied by
Infisical/runtime env (``VAPI_API_KEY``).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from hermes_cli import agent_core_sql as sql
from tools.registry import registry, tool_error

VOICE_METADATA_DESCRIPTION = (
    "Optional JSON metadata. Keep it generic and tenant-neutral: business_id, "
    "client_id, source_channel, contact_id, organization_id, opportunity_id, "
    "external_ref, labels, notes."
)

VAPI_DEFAULT_BASE_URL = "https://api.vapi.ai"
VAPI_DEFAULT_MODEL_PROVIDER = "openai"
VAPI_DEFAULT_MODEL = "gpt-4o"
VAPI_DEFAULT_VOICE_PROVIDER = "11labs"
VAPI_DEFAULT_VOICE_ID = "cgSgspJ2msm6clMCkdW9"


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False, sort_keys=True)


def _err(exc: Exception | str) -> str:
    return tool_error(str(exc))


def _user() -> str:
    return sql.runtime_env().get("VOICE_DB_RUNTIME_USER", "voice_runtime")


def _q(value: Any) -> str:
    return sql.quote_literal(value)


def _j(value: Any) -> str:
    return sql.quote_jsonb(value)


def _slug(prefix: str, value: str) -> str:
    return f"{prefix}-{sql.slugify(value)}"


def _limit(value: Any, default: int = 20, maximum: int = 100) -> int:
    try:
        n = int(value or default)
    except Exception:
        n = default
    return max(1, min(maximum, n))


def _check_voice() -> bool:
    try:
        if not sql.enabled():
            return False
        sql.psql("SELECT 1;", user=_user())
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Vapi adapter helpers
# ---------------------------------------------------------------------------

def _vapi_base_url() -> str:
    env = sql.runtime_env()
    return (
        env.get("VAPI_BASE_URL")
        or env.get("VAPI_API_BASE_URL")
        or env.get("VAPI_API_BASE")
        or os.getenv("VAPI_BASE_URL")
        or VAPI_DEFAULT_BASE_URL
    ).rstrip("/")


def _vapi_api_key() -> str:
    env = sql.runtime_env()
    return env.get("VAPI_API_KEY") or os.getenv("VAPI_API_KEY") or ""


def _vapi_configured() -> bool:
    return bool(_vapi_api_key())


def _vapi_request(method: str, path: str, body: dict[str, Any] | None = None,
                  query: dict[str, Any] | None = None) -> dict[str, Any]:
    key = _vapi_api_key()
    if not key:
        return {
            "ok": False,
            "configured": False,
            "error": "Vapi adapter is not configured. Set VAPI_API_KEY via Infisical/runtime env.",
        }

    url = f"{_vapi_base_url()}/{path.lstrip('/')}"
    if query:
        clean = {k: v for k, v in query.items() if v is not None and v != ""}
        if clean:
            url += "?" + urllib.parse.urlencode(clean)

    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Cloudflare may block Python's default urllib signature (403 / error 1010).
        # Use a normal integration UA instead of relying on library defaults.
        "User-Agent": "hermes-agent-vapi-adapter/1.0",
    }
    req = urllib.request.Request(url=url, method=method.upper(), data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else None
            return {"ok": True, "status": resp.status, "url": url, "data": payload}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return {"ok": False, "status": exc.code, "url": url, "error": payload}
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc)}


def _first_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("id", "assistantId", "phoneNumberId", "callId"):
            if payload.get(key):
                return str(payload[key])
        for value in payload.values():
            found = _first_id(value)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _first_id(value)
            if found:
                return found
    return None


def _extract_call_id(payload: dict[str, Any]) -> str | None:
    if payload.get("id"):
        return str(payload["id"])
    call = payload.get("call")
    if isinstance(call, dict) and call.get("id"):
        return str(call["id"])
    return None


def _metadata(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("metadata")
    meta: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    for key in ("business_id", "client_id", "contact_id", "organization_id", "opportunity_id", "source_channel", "external_ref"):
        if args.get(key) is not None:
            meta.setdefault(key, args.get(key))
    return meta


# ---------------------------------------------------------------------------
# Local persistence helpers
# ---------------------------------------------------------------------------

def _upsert_assistant(local_id: str, provider_assistant_id: str | None, name: str,
                      config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    return sql.statement_one(f"""
      INSERT INTO voice.assistants (assistant_id, provider, provider_assistant_id, name, business_id, client_id, status, config, metadata, created_at, updated_at)
      VALUES ({_q(local_id)}, 'vapi', {_q(provider_assistant_id)}, {_q(name)}, {_q(metadata.get('business_id'))}, {_q(metadata.get('client_id'))}, 'active', {_j(config)}, {_j(metadata)}, now(), now())
      ON CONFLICT (assistant_id) DO UPDATE SET provider_assistant_id=EXCLUDED.provider_assistant_id, name=EXCLUDED.name, business_id=EXCLUDED.business_id, client_id=EXCLUDED.client_id, config=EXCLUDED.config, metadata=EXCLUDED.metadata, updated_at=now()
      RETURNING *
    """, user=_user())


def _upsert_phone_number(local_id: str, provider_phone_number_id: str | None, number: str | None,
                         name: str | None, assistant_id: str | None, config: dict[str, Any],
                         metadata: dict[str, Any]) -> dict[str, Any] | None:
    return sql.statement_one(f"""
      INSERT INTO voice.phone_numbers (phone_number_id, provider, provider_phone_number_id, number, name, assistant_id, status, config, metadata, created_at, updated_at)
      VALUES ({_q(local_id)}, 'vapi', {_q(provider_phone_number_id)}, {_q(number)}, {_q(name)}, {_q(assistant_id)}, 'active', {_j(config)}, {_j(metadata)}, now(), now())
      ON CONFLICT (phone_number_id) DO UPDATE SET provider_phone_number_id=EXCLUDED.provider_phone_number_id, number=EXCLUDED.number, name=EXCLUDED.name, assistant_id=EXCLUDED.assistant_id, config=EXCLUDED.config, metadata=EXCLUDED.metadata, updated_at=now()
      RETURNING *
    """, user=_user())


def _upsert_call(local_id: str, provider_call_id: str | None, direction: str, status: str,
                 assistant_id: str | None, phone_number_id: str | None, from_number: str | None,
                 to_number: str | None, artifact: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    return sql.statement_one(f"""
      INSERT INTO voice.calls (call_id, provider, provider_call_id, direction, status, assistant_id, phone_number_id, from_number, to_number, customer_id, contact_id, organization_id, opportunity_id, artifact, metadata, created_at, updated_at)
      VALUES ({_q(local_id)}, 'vapi', {_q(provider_call_id)}, {_q(direction)}, {_q(status)}, {_q(assistant_id)}, {_q(phone_number_id)}, {_q(from_number)}, {_q(to_number)}, {_q(metadata.get('customer_id'))}, {_q(metadata.get('contact_id'))}, {_q(metadata.get('organization_id'))}, {_q(metadata.get('opportunity_id'))}, {_j(artifact)}, {_j(metadata)}, now(), now())
      ON CONFLICT (call_id) DO UPDATE SET provider_call_id=EXCLUDED.provider_call_id, direction=EXCLUDED.direction, status=EXCLUDED.status, assistant_id=EXCLUDED.assistant_id, phone_number_id=EXCLUDED.phone_number_id, from_number=EXCLUDED.from_number, to_number=EXCLUDED.to_number, customer_id=EXCLUDED.customer_id, contact_id=EXCLUDED.contact_id, organization_id=EXCLUDED.organization_id, opportunity_id=EXCLUDED.opportunity_id, artifact=EXCLUDED.artifact, metadata=EXCLUDED.metadata, updated_at=now()
      RETURNING *
    """, user=_user())


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_voice_status(args: dict, **_kwargs) -> str:
    try:
        counts = sql.one("""
          SELECT
            (SELECT count(*) FROM voice.assistants) AS assistants,
            (SELECT count(*) FROM voice.phone_numbers) AS phone_numbers,
            (SELECT count(*) FROM voice.calls) AS calls,
            (SELECT count(*) FROM voice.call_events) AS call_events
        """, user=_user())
        return _ok(
            db_backend="agent_core_postgres",
            counts=counts,
            adapters={"vapi": {"configured": _vapi_configured(), "base_url": _vapi_base_url()}},
        )
    except Exception as exc:
        return _err(exc)


def _handle_vapi_raw_request(args: dict, **_kwargs) -> str:
    method = str(args.get("method") or "").upper()
    path = str(args.get("path") or "").strip()
    if not method or not path:
        return _err("method and path are required")
    if method not in {"GET", "POST", "PATCH", "PUT", "DELETE"}:
        return _err("method must be GET, POST, PATCH, PUT, or DELETE")
    return json.dumps(_vapi_request(method, path, args.get("body"), args.get("query")), ensure_ascii=False, sort_keys=True)


def _handle_assistant_create(args: dict, **_kwargs) -> str:
    try:
        name = str(args.get("name") or "").strip()
        if not name:
            return _err("name is required")
        system_prompt = str(args.get("system_prompt") or args.get("systemPrompt") or "").strip()
        if not system_prompt:
            return _err("system_prompt is required")

        model_provider = args.get("model_provider") or VAPI_DEFAULT_MODEL_PROVIDER
        model = args.get("model") or VAPI_DEFAULT_MODEL
        body: dict[str, Any] = {
            "name": name,
            "model": {
                "provider": model_provider,
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}],
            },
        }
        if args.get("first_message"):
            body["firstMessage"] = args["first_message"]
        voice_provider = args.get("voice_provider") or VAPI_DEFAULT_VOICE_PROVIDER
        voice_id = args.get("voice_id") or VAPI_DEFAULT_VOICE_ID
        if voice_provider and voice_id:
            body["voice"] = {"provider": voice_provider, "voiceId": voice_id}
        if args.get("server_url"):
            body["server"] = {"url": args["server_url"]}
            if args.get("server_credential_id"):
                body["server"]["credentialId"] = args["server_credential_id"]

        result = _vapi_request("POST", "/assistant", body)
        provider_id = _first_id(result.get("data")) if result.get("ok") else None
        local_id = args.get("assistant_id") or (f"vapi-assistant-{provider_id}" if provider_id else _slug("assistant", name))
        meta = _metadata(args)
        row = _upsert_assistant(local_id, provider_id, name, body, meta)
        return _ok(assistant=row, provider_result=result)
    except Exception as exc:
        return _err(exc)


def _handle_assistant_list(args: dict, **_kwargs) -> str:
    result = _vapi_request("GET", "/assistant", query={"limit": _limit(args.get("limit"), 50, 100)})
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _handle_phone_number_list(args: dict, **_kwargs) -> str:
    result = _vapi_request("GET", "/phone-number")
    if result.get("ok") and isinstance(result.get("data"), list):
        rows = []
        for item in result["data"]:
            if not isinstance(item, dict):
                continue
            provider_id = str(item.get("id") or "") or None
            number = item.get("number") or item.get("sipUri")
            local_id = f"vapi-phone-{provider_id}" if provider_id else _slug("phone", str(number or "unknown"))
            row = _upsert_phone_number(local_id, provider_id, number, item.get("name"), None, item, {})
            rows.append(row)
        result["local_rows"] = rows
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _handle_call_start(args: dict, **_kwargs) -> str:
    try:
        customer_number = str(args.get("customer_number") or args.get("to_number") or "").strip()
        if not customer_number:
            return _err("customer_number is required")
        assistant_id = args.get("provider_assistant_id") or args.get("assistant_id")
        phone_number_id = args.get("provider_phone_number_id") or args.get("phone_number_id")
        if not assistant_id:
            return _err("assistant_id or provider_assistant_id is required")
        if not phone_number_id:
            return _err("phone_number_id or provider_phone_number_id is required")

        body: dict[str, Any] = {
            "assistantId": assistant_id,
            "phoneNumberId": phone_number_id,
            "customer": {"number": customer_number},
        }
        if args.get("name"):
            body["name"] = args["name"]
        if isinstance(args.get("assistant_overrides"), dict):
            body["assistantOverrides"] = args["assistant_overrides"]
        if isinstance(args.get("metadata"), dict):
            body["metadata"] = args["metadata"]

        result = _vapi_request("POST", "/call", body)
        provider_call_id = _first_id(result.get("data")) if result.get("ok") else None
        local_id = args.get("call_id") or (f"vapi-call-{provider_call_id}" if provider_call_id else _slug("call", customer_number))
        meta = _metadata(args)
        data = result.get("data")
        artifact: dict[str, Any] = data if isinstance(data, dict) else {"provider_result": result}
        row = _upsert_call(
            str(local_id),
            provider_call_id,
            "outbound",
            "created" if result.get("ok") else "failed",
            args.get("assistant_id") if args.get("assistant_id") != args.get("provider_assistant_id") else None,
            args.get("phone_number_id") if args.get("phone_number_id") != args.get("provider_phone_number_id") else None,
            None,
            customer_number,
            artifact,
            meta,
        )
        return _ok(call=row, provider_result=result)
    except Exception as exc:
        return _err(exc)


def _handle_call_get(args: dict, **_kwargs) -> str:
    call_id = str(args.get("call_id") or args.get("provider_call_id") or "").strip()
    if not call_id:
        return _err("call_id or provider_call_id is required")
    provider_id = call_id.removeprefix("vapi-call-")
    result = _vapi_request("GET", f"/call/{provider_id}")
    if result.get("ok") and isinstance(result.get("data"), dict):
        data = result["data"]
        local_id = args.get("call_id") if args.get("call_id") and not str(args.get("call_id")).startswith(provider_id) else f"vapi-call-{provider_id}"
        _upsert_call(
            str(local_id),
            provider_id,
            str(data.get("type") or data.get("direction") or "unknown").replace("PhoneCall", "").lower() or "unknown",
            str(data.get("status") or "unknown"),
            None,
            None,
            (data.get("customer") or {}).get("number") if isinstance(data.get("customer"), dict) else None,
            None,
            data,
            {},
        )
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _handle_call_event_record(args: dict, **_kwargs) -> str:
    try:
        event_type = str(args.get("event_type") or "").strip()
        if not event_type:
            return _err("event_type is required")
        raw_payload = args.get("payload")
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
        provider_call_id = args.get("provider_call_id") or _extract_call_id(payload)
        call_id = args.get("call_id") or (f"vapi-call-{provider_call_id}" if provider_call_id else None)
        row = sql.statement_one(f"""
          INSERT INTO voice.call_events (call_id, provider, provider_call_id, event_type, verified, payload, received_at)
          VALUES ({_q(call_id)}, 'vapi', {_q(provider_call_id)}, {_q(event_type)}, {sql.quote_literal(bool(args.get('verified')))}, {_j(payload)}, now())
          RETURNING *
        """, user=_user())
        return _ok(event=row)
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Schemas / registration
# ---------------------------------------------------------------------------

def _schema(name: str, description: str, props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required or [],
        },
    }


def _meta_props() -> dict[str, Any]:
    return {"metadata": {"type": "object", "description": VOICE_METADATA_DESCRIPTION}}


registry.register(name="voice_status", toolset="voice", schema=_schema("voice_status", "Return Voice Core row counts, DB backend, and Vapi adapter configuration status.", {}), handler=_handle_voice_status, check_fn=_check_voice, emoji="📞")
registry.register(name="voice_assistant_create", toolset="voice", schema=_schema("voice_assistant_create", "Create a Vapi assistant and mirror it in Voice Core.", {"assistant_id": {"type": "string"}, "name": {"type": "string"}, "system_prompt": {"type": "string"}, "first_message": {"type": "string"}, "model_provider": {"type": "string"}, "model": {"type": "string"}, "voice_provider": {"type": "string"}, "voice_id": {"type": "string"}, "server_url": {"type": "string"}, "server_credential_id": {"type": "string"}, **_meta_props()}, ["name", "system_prompt"]), handler=_handle_assistant_create, check_fn=_check_voice, emoji="📞")
registry.register(name="voice_assistant_list", toolset="voice", schema=_schema("voice_assistant_list", "List Vapi assistants through the configured adapter.", {"limit": {"type": "integer"}}), handler=_handle_assistant_list, check_fn=_check_voice, emoji="📞")
registry.register(name="voice_phone_number_list", toolset="voice", schema=_schema("voice_phone_number_list", "List Vapi phone numbers and mirror them in Voice Core.", {}), handler=_handle_phone_number_list, check_fn=_check_voice, emoji="📞")
registry.register(name="voice_call_start", toolset="voice", schema=_schema("voice_call_start", "Start an outbound Vapi call and store the call ledger row in Voice Core.", {"call_id": {"type": "string"}, "name": {"type": "string"}, "assistant_id": {"type": "string", "description": "Local or provider assistant id. If provider_assistant_id is set, this may be local metadata only."}, "provider_assistant_id": {"type": "string"}, "phone_number_id": {"type": "string", "description": "Local or provider phone number id. If provider_phone_number_id is set, this may be local metadata only."}, "provider_phone_number_id": {"type": "string"}, "customer_number": {"type": "string"}, "assistant_overrides": {"type": "object"}, **_meta_props()}, ["customer_number"]), handler=_handle_call_start, check_fn=_check_voice, emoji="📞")
registry.register(name="voice_call_get", toolset="voice", schema=_schema("voice_call_get", "Get a Vapi call by provider call id and mirror the latest data locally.", {"call_id": {"type": "string"}, "provider_call_id": {"type": "string"}}), handler=_handle_call_get, check_fn=_check_voice, emoji="📞")
registry.register(name="voice_call_event_record", toolset="voice", schema=_schema("voice_call_event_record", "Record a Vapi webhook/server-url event in Voice Core after endpoint authentication.", {"call_id": {"type": "string"}, "provider_call_id": {"type": "string"}, "event_type": {"type": "string"}, "verified": {"type": "boolean"}, "payload": {"type": "object"}}, ["event_type"]), handler=_handle_call_event_record, check_fn=_check_voice, emoji="📞")
registry.register(name="voice_vapi_raw_request", toolset="voice", schema=_schema("voice_vapi_raw_request", "Advanced escape hatch for Vapi REST endpoints. Prefer canonical Voice tools first.", {"method": {"type": "string", "enum": ["GET", "POST", "PATCH", "PUT", "DELETE"]}, "path": {"type": "string"}, "body": {"type": "object"}, "query": {"type": "object"}}, ["method", "path"]), handler=_handle_vapi_raw_request, check_fn=_check_voice, emoji="📞")
