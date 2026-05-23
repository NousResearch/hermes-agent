from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "voice_phone_gateway.py"


def load_module():
    spec = importlib.util.spec_from_file_location("voice_phone_gateway", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vapi_webhook_requires_auth_by_default(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("VAPI_WEBHOOK_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("VAPI_WEBHOOK_HMAC_SECRET", raising=False)
    monkeypatch.delenv("VAPI_WEBHOOK_AUTH_DISABLED", raising=False)

    request = SimpleNamespace(headers={}, remote="127.0.0.1")

    assert mod._validate_vapi_auth(request, b"{}") is False


def test_vapi_webhook_accepts_configured_bearer(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("VAPI_WEBHOOK_BEARER_TOKEN", "guest-pilot-secret")
    monkeypatch.delenv("VAPI_WEBHOOK_HMAC_SECRET", raising=False)

    request = SimpleNamespace(headers={"Authorization": "Bearer guest-pilot-secret"}, remote="203.0.113.10")

    assert mod._validate_vapi_auth(request, b"{}") is True


def test_vapi_forbidden_tool_does_not_execute(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    result = mod._handle_vapi_tool_call(
        {"id": "tool-1", "name": "create_booking", "arguments": {"room": "BV3"}},
        {"type": "tool-calls", "call": {"id": "call-1"}},
    )

    assert result["toolCallId"] == "tool-1"
    assert result["result"]["ok"] is False
    assert result["result"]["status"] == "forbidden_tool"
    assert not (tmp_path / ".hermes" / "reports" / "vapi-guest-intake").exists()


def test_vapi_guest_intake_is_recorded_for_review(tmp_path: Path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    result = mod._handle_vapi_tool_call(
        {
            "id": "tool-2",
            "name": "create_guest_intake",
            "arguments": {"guest_name": "Ada", "request": "callback"},
        },
        {"type": "tool-calls", "call": {"id": "call-2", "customer": {"number": "+491234"}}},
    )

    assert result["result"]["ok"] is True
    assert result["result"]["status"] == "queued_for_human_review"
    record_path = Path(result["result"]["handoff_path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["source"] == "vapi"
    assert record["kind"] == "create_guest_intake"
    assert record["call_id"] == "call-2"
    assert record["detail"]["arguments"]["guest_name"] == "Ada"


def test_vapi_booking_link_missing_dates_asks_for_dates():
    mod = load_module()

    result = mod._booking_link_from_args({"adults": 2})

    assert result["ok"] is False
    assert result["status"] == "missing_dates"


def test_vapi_booking_link_uses_read_only_helper(monkeypatch):
    mod = load_module()

    def fake_run(cmd, check, capture_output, text, timeout):
        assert "/Users/appleserver/.hermes/bin/hotelrunner-booking-link" in cmd[0]
        assert "--checkin" in cmd
        return SimpleNamespace(returncode=0, stdout='{"url":"https://example.test/bv3/search"}', stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = mod._booking_link_from_args({"checkin": "2026-06-01", "checkout": "2026-06-03", "adults": 2})

    assert result["ok"] is True
    assert result["booking_link"]["url"] == "https://example.test/bv3/search"
