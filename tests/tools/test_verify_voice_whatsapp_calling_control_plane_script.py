import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "verify_voice_whatsapp_calling_control_plane.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_voice_whatsapp_calling_control_plane",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    script = _load_script_module()
    values = {
        "sidecar_url": script.DEFAULT_SIDECAR_URL,
        "phone_number_id": script.DEFAULT_PHONE_NUMBER_ID,
        "call_id": script.DEFAULT_CALL_ID,
        "caller": script.DEFAULT_CALLER,
        "callee": script.DEFAULT_CALLEE,
        "contact_name": script.DEFAULT_CONTACT_NAME,
        "remote_sdp": script.DEFAULT_REMOTE_SDP,
        "answer_sdp": script.DEFAULT_ANSWER_SDP,
        "timeout": 10.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_call_connect_payload_matches_meta_calls_shape():
    script = _load_script_module()

    payload = script.call_connect_payload(
        call_id="wacid.call-1",
        caller="15551234567",
        callee="15557654321",
        contact_name="Alice",
        remote_sdp="v=0\r\n",
    )

    assert payload["object"] == "whatsapp_business_account"
    change = payload["entry"][0]["changes"][0]
    assert change["field"] == "calls"
    value = change["value"]
    assert value["contacts"][0]["profile"]["name"] == "Alice"
    call = value["calls"][0]
    assert call["id"] == "wacid.call-1"
    assert call["event"] == "connect"
    assert call["session"] == {"sdp_type": "offer", "sdp": "v=0\r\n"}


@pytest.mark.asyncio
async def test_fake_http_client_records_contract_offer_graph_and_close():
    script = _load_script_module()
    client = script.RecordingHttpClient(
        sidecar_url="http://127.0.0.1:8787",
        api_version="v20.0",
        phone_number_id="phone-1",
        call_id="call-1",
        answer_sdp="v=0\r\nanswer\r\n",
    )

    contract = await client.get("http://127.0.0.1:8787/contract", timeout=1)
    offer = await client.post(
        "http://127.0.0.1:8787/offer",
        json={"call_id": "call-1", "type": "offer", "sdp": "v=0\r\n"},
    )
    graph = await client.post(
        "https://graph.facebook.com/v20.0/phone-1/calls",
        json={"call_id": "call-1", "action": "pre_accept"},
    )
    closed = await client.post("http://127.0.0.1:8787/calls/call-1/close")

    assert contract.json()["contract"] == "voice.webrtc_sidecar"
    assert offer.json()["type"] == "answer"
    assert offer.json()["sdp"] == "v=0\r\nanswer\r\n"
    assert offer.json()["state"]["ready_for_accept"] is True
    assert graph.json()["action"] == "pre_accept"
    assert closed.json()["closed"] is True
    assert [request["method"] for request in client.requests] == [
        "GET",
        "POST",
        "POST",
        "POST",
    ]


@pytest.mark.asyncio
async def test_control_plane_smoke_proves_accept_and_terminate_sequence():
    script = _load_script_module()

    result = await script.run_control_plane_smoke(_args(call_id="wacid.call-1"))

    assert result["success"] is True
    assert result["call_id"] == "wacid.call-1"
    assert result["contract_loaded"] is True
    assert result["sidecar_offer_url"] == "http://127.0.0.1:8787/offer"
    assert result["graph_actions"] == ["pre_accept", "accept"]
    assert result["sidecar_ready_for_accept"] is True
    assert result["sidecar_close_url"] == (
        "http://127.0.0.1:8787/calls/wacid.call-1/close"
    )
    assert result["drain_starts"] == [
        {
            "call_id": "wacid.call-1",
            "chat_id": script.DEFAULT_CALLER,
            "sender_name": script.DEFAULT_CONTACT_NAME,
        }
    ]
    assert result["audio"]["frame_bytes"] == 1920


def test_main_prints_json(monkeypatch, capsys):
    script = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_voice_whatsapp_calling_control_plane.py",
            "--call-id",
            "wacid.call-2",
        ],
    )

    assert script.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output["success"] is True
    assert output["call_id"] == "wacid.call-2"
