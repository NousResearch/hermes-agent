"""Telephony tools for outbound calling via Vapi and SMS via textbee.dev.

The tools are intentionally small REST wrappers with environment-based
configuration:

Vapi calling:
- ``VAPI_API_KEY`` (required)
- ``VAPI_PHONE_NUMBER_ID`` (optional default)
- ``VAPI_ASSISTANT_ID`` (optional default)
- ``VAPI_BASE_URL`` (optional, defaults to https://api.vapi.ai)

textbee.dev SMS:
- ``TEXTBEE_API_KEY`` (required)
- ``TEXTBEE_DEVICE_ID`` (optional default)
- ``TEXTBEE_BASE_URL`` (optional, defaults to https://api.textbee.dev)

These tools perform real-world side effects (phone calls/SMS). The model should
only call them when the user explicitly requests sending a call/text to the
specified recipient(s).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, Optional

from tools.registry import registry

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")
_MAX_SMS_RECIPIENTS = 50


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _vapi_base_url() -> str:
    return _env("VAPI_BASE_URL", "https://api.vapi.ai").rstrip("/")


def _textbee_base_url() -> str:
    return _env("TEXTBEE_BASE_URL", "https://api.textbee.dev").rstrip("/")


def _check_vapi_available() -> bool:
    return bool(_env("VAPI_API_KEY"))


def _check_textbee_available() -> bool:
    return bool(_env("TEXTBEE_API_KEY"))


def _json_response(success: bool, **payload: Any) -> str:
    return json.dumps({"success": success, **payload}, ensure_ascii=False)


def _require_e164(number: str, field: str = "phone_number") -> Optional[str]:
    if not isinstance(number, str) or not _E164_RE.match(number.strip()):
        return f"{field} must be in E.164 format, e.g. +12345678900"
    return None


def _parse_json_or_text(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _http_json(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    data = None
    request_headers = dict(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url, data=data, headers=request_headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return {
                "ok": 200 <= resp.status < 300,
                "status_code": resp.status,
                "data": _parse_json_or_text(raw),
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return {
            "ok": False,
            "status_code": exc.code,
            "data": _parse_json_or_text(raw),
        }
    except urllib.error.URLError as exc:
        return {"ok": False, "status_code": None, "error": str(exc.reason)}
    except TimeoutError:
        return {"ok": False, "status_code": None, "error": "request timed out"}


def _clean_metadata(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return None


def _safe_call_summary(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"raw": data}
    keys = [
        "id", "orgId", "createdAt", "updatedAt", "type", "status", "endedReason",
        "phoneNumberId", "assistantId", "customer", "destination", "startedAt", "endedAt",
        "summary", "analysis", "transcript", "recordingUrl", "stereoRecordingUrl",
        "cost", "costs", "artifact",
    ]
    return {k: data[k] for k in keys if k in data}


def _handle_vapi_create_call(args: Dict[str, Any], **_: Any) -> str:
    api_key = _env("VAPI_API_KEY")
    if not api_key:
        return _json_response(False, error="VAPI_API_KEY is not configured")

    customer_number = (args.get("customer_number") or args.get("phone_number") or "").strip()
    err = _require_e164(customer_number, "customer_number")
    if err:
        return _json_response(False, error=err)

    phone_number_id = (args.get("phone_number_id") or _env("VAPI_PHONE_NUMBER_ID")).strip()
    assistant_id = (args.get("assistant_id") or _env("VAPI_ASSISTANT_ID")).strip()
    if not phone_number_id:
        return _json_response(False, error="phone_number_id is required (or set VAPI_PHONE_NUMBER_ID)")
    if not assistant_id:
        return _json_response(False, error="assistant_id is required (or set VAPI_ASSISTANT_ID)")

    payload: Dict[str, Any] = {
        "phoneNumberId": phone_number_id,
        "assistantId": assistant_id,
        "customer": {"number": customer_number},
    }

    customer_name = (args.get("customer_name") or "").strip()
    if customer_name:
        payload["customer"]["name"] = customer_name

    assistant_overrides = args.get("assistant_overrides")
    if isinstance(assistant_overrides, dict) and assistant_overrides:
        payload["assistantOverrides"] = assistant_overrides

    metadata = _clean_metadata(args.get("metadata"))
    if metadata:
        payload["metadata"] = metadata

    response = _http_json(
        "POST",
        f"{_vapi_base_url()}/call",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload=payload,
        timeout=30,
    )
    if not response.get("ok"):
        return _json_response(False, status_code=response.get("status_code"), error=response.get("error") or response.get("data"))

    data = response.get("data")
    return _json_response(True, status_code=response.get("status_code"), call=_safe_call_summary(data))


def _handle_vapi_get_call(args: Dict[str, Any], **_: Any) -> str:
    api_key = _env("VAPI_API_KEY")
    if not api_key:
        return _json_response(False, error="VAPI_API_KEY is not configured")
    call_id = (args.get("call_id") or "").strip()
    if not call_id:
        return _json_response(False, error="call_id is required")

    response = _http_json(
        "GET",
        f"{_vapi_base_url()}/call/{urllib.parse.quote(call_id, safe='')}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    if not response.get("ok"):
        return _json_response(False, status_code=response.get("status_code"), error=response.get("error") or response.get("data"))
    return _json_response(True, status_code=response.get("status_code"), call=_safe_call_summary(response.get("data")))


def _coerce_recipients(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, Iterable):
        return [str(v).strip() for v in value]
    return []


def _handle_textbee_send_sms(args: Dict[str, Any], **_: Any) -> str:
    api_key = _env("TEXTBEE_API_KEY")
    if not api_key:
        return _json_response(False, error="TEXTBEE_API_KEY is not configured")

    device_id = (args.get("device_id") or _env("TEXTBEE_DEVICE_ID")).strip()
    if not device_id:
        return _json_response(False, error="device_id is required (or set TEXTBEE_DEVICE_ID)")

    recipients = _coerce_recipients(args.get("recipients") or args.get("recipient"))
    recipients = [r for r in recipients if r]
    if not recipients:
        return _json_response(False, error="at least one recipient is required")
    if len(recipients) > _MAX_SMS_RECIPIENTS:
        return _json_response(False, error=f"too many recipients; max is {_MAX_SMS_RECIPIENTS}")
    for recipient in recipients:
        err = _require_e164(recipient, "recipient")
        if err:
            return _json_response(False, error=err, recipient=recipient)

    message = args.get("message")
    if not isinstance(message, str) or not message.strip():
        return _json_response(False, error="message is required")

    payload: Dict[str, Any] = {"recipients": recipients, "message": message}
    sim_subscription_id = args.get("sim_subscription_id")
    if sim_subscription_id is not None:
        try:
            payload["simSubscriptionId"] = int(sim_subscription_id)
        except (TypeError, ValueError):
            return _json_response(False, error="sim_subscription_id must be a number")

    response = _http_json(
        "POST",
        f"{_textbee_base_url()}/api/v1/gateway/devices/{urllib.parse.quote(device_id, safe='')}/send-sms",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        payload=payload,
        timeout=30,
    )
    if not response.get("ok"):
        return _json_response(False, status_code=response.get("status_code"), error=response.get("error") or response.get("data"))
    return _json_response(True, status_code=response.get("status_code"), result=response.get("data"), recipients_count=len(recipients))


registry.register(
    name="vapi_create_call",
    toolset="telephony",
    schema={
        "name": "vapi_create_call",
        "description": (
            "Place an outbound phone call using Vapi. Use only after the user explicitly requests "
            "a call to a specified phone number. Requires VAPI_API_KEY and either args or env defaults "
            "for assistant_id and phone_number_id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_number": {"type": "string", "description": "Recipient phone number in E.164 format, e.g. +12345678900"},
                "customer_name": {"type": "string", "description": "Optional recipient/customer name"},
                "assistant_id": {"type": "string", "description": "Vapi assistant ID; defaults to VAPI_ASSISTANT_ID"},
                "phone_number_id": {"type": "string", "description": "Vapi phone number ID; defaults to VAPI_PHONE_NUMBER_ID"},
                "assistant_overrides": {"type": "object", "description": "Optional Vapi assistantOverrides payload"},
                "metadata": {"type": "object", "description": "Optional metadata to attach to the call"},
            },
            "required": ["customer_number"],
        },
    },
    handler=_handle_vapi_create_call,
    check_fn=_check_vapi_available,
    requires_env=["VAPI_API_KEY"],
    description="Place outbound calls via Vapi",
    emoji="📞",
)

registry.register(
    name="vapi_get_call",
    toolset="telephony",
    schema={
        "name": "vapi_get_call",
        "description": "Fetch status/details for a Vapi call by ID. Requires VAPI_API_KEY.",
        "parameters": {
            "type": "object",
            "properties": {"call_id": {"type": "string", "description": "Vapi call ID"}},
            "required": ["call_id"],
        },
    },
    handler=_handle_vapi_get_call,
    check_fn=_check_vapi_available,
    requires_env=["VAPI_API_KEY"],
    description="Get Vapi call details",
    emoji="📞",
)

registry.register(
    name="textbee_send_sms",
    toolset="telephony",
    schema={
        "name": "textbee_send_sms",
        "description": (
            "Send an SMS/text message using textbee.dev through a registered Android device. "
            "Use only after the user explicitly requests texting specified recipient(s). Requires TEXTBEE_API_KEY "
            "and either device_id or TEXTBEE_DEVICE_ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "recipients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recipient phone numbers in E.164 format, e.g. +12345678900",
                },
                "message": {"type": "string", "description": "SMS body to send"},
                "device_id": {"type": "string", "description": "textbee.dev device ID; defaults to TEXTBEE_DEVICE_ID"},
                "sim_subscription_id": {"type": "integer", "description": "Optional SIM subscription ID for dual-SIM devices"},
            },
            "required": ["recipients", "message"],
        },
    },
    handler=_handle_textbee_send_sms,
    check_fn=_check_textbee_available,
    requires_env=["TEXTBEE_API_KEY"],
    description="Send SMS via textbee.dev",
    emoji="💬",
)
