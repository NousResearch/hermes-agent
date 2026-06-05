import json

from tools import voice_tool


def test_vapi_request_reports_unconfigured_without_network(monkeypatch):
    monkeypatch.setattr(voice_tool.sql, "runtime_env", lambda: {})
    monkeypatch.delenv("VAPI_API_KEY", raising=False)

    result = voice_tool._vapi_request("GET", "/assistant")

    assert result["ok"] is False
    assert result["configured"] is False
    assert "VAPI_API_KEY" in result["error"]


def test_call_start_validates_required_customer_before_network(monkeypatch):
    called = False

    def fake_request(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"ok": True}

    monkeypatch.setattr(voice_tool, "_vapi_request", fake_request)

    payload = json.loads(voice_tool._handle_call_start({"assistant_id": "asst", "phone_number_id": "pn"}))

    assert payload["error"] == "customer_number is required"
    assert called is False


def test_metadata_merges_generic_context_without_mutating_input():
    args = {"metadata": {"labels": ["demo"]}, "client_id": "jean", "contact_id": "contact-1"}

    meta = voice_tool._metadata(args)

    assert meta == {"labels": ["demo"], "client_id": "jean", "contact_id": "contact-1"}
    assert args["metadata"] == {"labels": ["demo"]}


def test_toolset_exports_voice_tools():
    import toolsets

    voice_tools = set(toolsets.TOOLSETS["voice"]["tools"])

    assert "voice_status" in voice_tools
    assert "voice_assistant_create" in voice_tools
    assert "voice_phone_number_list" in voice_tools
    assert "voice_call_start" in voice_tools
    assert "voice_vapi_raw_request" in voice_tools
