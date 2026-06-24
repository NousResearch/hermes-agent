import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "verify_voice_whatsapp_calling_live_sidecar.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_voice_whatsapp_calling_live_sidecar",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_call_connect_payload_carries_offer_session():
    script = _load_script_module()

    payload = script.call_connect_payload(
        call_id="wacid.call",
        caller="1355",
        callee="1555",
        contact_name="Tester",
        remote_sdp="v=0\r\n",
    )

    call = payload["entry"][0]["changes"][0]["value"]["calls"][0]
    contact = payload["entry"][0]["changes"][0]["value"]["contacts"][0]
    assert call["id"] == "wacid.call"
    assert call["event"] == "connect"
    assert call["session"] == {"sdp_type": "offer", "sdp": "v=0\r\n"}
    assert contact["wa_id"] == "1355"
    assert contact["profile"]["name"] == "Tester"


def test_call_terminate_payload_carries_completed_status():
    script = _load_script_module()

    payload = script.call_terminate_payload(
        call_id="wacid.call",
        caller="1355",
        callee="1555",
    )

    call = payload["entry"][0]["changes"][0]["value"]["calls"][0]
    assert call["id"] == "wacid.call"
    assert call["event"] == "terminate"
    assert call["status"] == "COMPLETED"


def test_graph_actions_extracts_pre_accept_and_accept():
    script = _load_script_module()
    requests = [
        {"method": "GET", "url": "https://graph.facebook.com/v20.0/1/calls", "kwargs": {}},
        {
            "method": "POST",
            "url": "https://graph.facebook.com/v20.0/1/calls",
            "kwargs": {"json": {"action": "pre_accept"}},
        },
        {
            "method": "POST",
            "url": "https://graph.facebook.com/v20.0/1/calls",
            "kwargs": {"json": {"action": "accept"}},
        },
    ]

    assert script.graph_actions(
        requests,
        "https://graph.facebook.com/v20.0/1/calls",
    ) == ["pre_accept", "accept"]


def test_recorded_response_behaves_like_httpx_response():
    script = _load_script_module()
    response = script.RecordedResponse(200, {"success": True})

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert json.loads(response.text) == {"success": True}


@pytest.mark.asyncio
async def test_signed_webhook_request_carries_valid_hmac_signature():
    script = _load_script_module()
    payload = {"object": "whatsapp_business_account", "entry": []}

    request = script.signed_webhook_request(payload, app_secret="secret")
    raw = await request.read()

    assert json.loads(raw.decode("utf-8")) == payload
    adapter = script.WhatsAppCloudAdapter(
        script.PlatformConfig(enabled=True, extra={"app_secret": "secret"})
    )
    assert adapter._verify_signature(  # noqa: SLF001
        raw,
        request.headers["X-Hub-Signature-256"],
    )


def test_parse_args_accepts_existing_sidecar_url(monkeypatch):
    script = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_whatsapp_calling_live_sidecar.py",
            "--voice-repo",
            "/voice",
            "--sidecar-url",
            "http://127.0.0.1:8787/",
            "--app-secret",
            "secret",
        ],
    )

    args = script.parse_args()

    assert args.voice_repo == Path("/voice")
    assert args.sidecar_url == "http://127.0.0.1:8787/"
    assert args.app_secret == "secret"
