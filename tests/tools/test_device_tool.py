"""``device_request`` — the agent's side of a handheld client's capability advertisement (#11911).

Two contracts live here, both of them the reason this tool exists rather than a raw JSON-RPC call:

* a capture never enters the conversation as text (it goes to the media cache and the tool result
  carries a handle), and
* no capability that reaches a camera, microphone, position, screen, notifications or files is asked
  for without the shared human-approval gate agreeing first (``device.approval.<capability>``).
"""

import base64
import json

import pytest

from tools import approval
from tools.device_tool import CAPABILITIES, approval_policy, device_request_tool


def _config_with(monkeypatch, **device):
    """Point ``load_config`` at *device* (falsy = no config at all)."""
    cfg = {"device": device} if device else {}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)


def _interactive(monkeypatch):
    """A plain interactive CLI turn: prompts reach a human, the gate is not short-circuited."""
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)


def _jpeg(byte_count: int) -> bytes:
    return b"\xff\xd8\xff" + (b"\x00" * (byte_count - 3))


def _capture(byte_count: int = 300 * 1024, mime: str = "image/jpeg") -> dict:
    return {"mime_type": mime, "data": base64.b64encode(_jpeg(byte_count)).decode("ascii"),
            "label": "rear camera"}


@pytest.fixture
def allow_requests(monkeypatch):
    """``device.approval.*: allow`` — the capture tests are not about the gate."""
    _config_with(monkeypatch, approval={capability: "allow" for capability in CAPABILITIES})


# ── token cost (F1) ─────────────────────────────────────────────────────────────────────────────


def test_capture_reaches_the_model_as_a_handle_not_base64(allow_requests, tmp_path):
    """A 300 KB frame costs a few hundred characters of context, not the whole image, and the bytes
    are on disk for ``vision_analyze`` to read."""
    capture = _capture()
    naive = len(json.dumps({"value": capture["data"]}))  # what the wire used to turn into a tool result

    result = device_request_tool("camera", callback=lambda payload: {"value": "", "captures": [capture]})
    payload = json.loads(result)

    handle = payload["captures"][0]
    assert handle["mime_type"] == "image/jpeg"
    assert handle["bytes"] == 300 * 1024
    assert handle["label"] == "rear camera"
    assert handle["path"].endswith(".jpg")
    with open(handle["path"], "rb") as fh:
        assert fh.read() == _jpeg(300 * 1024)
    # The base64 body is nowhere in the tool result.
    assert capture["data"][:64] not in result
    assert "data" not in handle
    assert len(result) < 600, result
    assert len(result) < naive / 500  # the whole point of the change


def test_oversize_capture_is_a_note_not_a_payload(allow_requests, monkeypatch):
    _config_with(monkeypatch, max_capture_bytes=64 * 1024,
                 approval={capability: "allow" for capability in CAPABILITIES})
    result = json.loads(device_request_tool(
        "camera", callback=lambda payload: {"captures": [_capture(100 * 1024)]}))
    assert "captures" not in result
    assert "device.max_capture_bytes" in result["notes"][0]
    assert len(json.dumps(result)) < 400


def test_unknown_media_type_is_refused_rather_than_guessed(allow_requests):
    """An extension is never inferred from client bytes: an unlisted MIME type is dropped with a note."""
    result = json.loads(device_request_tool(
        "camera", callback=lambda payload: {"captures": [{"mime_type": "application/x-evil", "data": "AAEC"}]}))
    assert "captures" not in result
    assert "unsupported media type application/x-evil" in result["notes"][0]


def test_malformed_base64_does_not_lose_the_rest_of_the_answer(allow_requests):
    result = json.loads(device_request_tool(
        "camera", callback=lambda payload: {"value": "ok", "captures": [{"mime_type": "image/png",
                                                                        "data": "!!!not base64!!!"}]}))
    assert result["value"] == "ok"
    assert "not valid base64" in result["notes"][0]


def test_oversize_value_is_truncated(allow_requests):
    result = json.loads(device_request_tool("location", callback=lambda payload: {"value": "x" * 9000}))
    assert len(result["value"]) < 4200
    assert "truncated" in result["value"]


def test_a_client_refusal_is_not_a_success(allow_requests):
    result = json.loads(device_request_tool(
        "camera", callback=lambda payload: {"denied": True, "reason": "the user declined the request"}))
    assert result == {"capability": "camera", "success": False, "denied": True,
                      "reason": "the user declined the request"}


# ── privacy (F2) ───────────────────────────────────────────────────────────────────────────────


def test_default_policy_is_ask(monkeypatch):
    """No config at all still means "ask": a privacy default never drifts to allow."""
    _config_with(monkeypatch)
    for capability in CAPABILITIES:
        assert approval_policy(capability) == "ask"


def test_denied_approval_never_writes_a_request(monkeypatch):
    """The prompt is on the path by default, and a denial stops the request before the transport."""
    _config_with(monkeypatch)
    _interactive(monkeypatch)
    asked = []
    monkeypatch.setattr(approval, "prompt_dangerous_approval",
                        lambda command, description, **kw: asked.append((command, description)) or "deny")
    calls = []

    payload = json.loads(device_request_tool("camera", callback=lambda p: calls.append(p) or {"value": "x"}))

    assert asked and asked[0][0] == "device_request: camera"
    assert "camera" in asked[0][1]
    assert calls == []
    assert payload["denied"] is True and payload["success"] is False
    assert "denied" in payload["reason"].lower()


def test_approved_once_still_calls_the_client(monkeypatch):
    _config_with(monkeypatch)
    _interactive(monkeypatch)
    monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "once")
    calls = []

    payload = json.loads(device_request_tool(
        "location", callback=lambda p: calls.append(p) or {"value": '{"lat":1}'}))

    assert calls == [{"capability": "location"}]
    assert payload["success"] is True and payload["value"] == '{"lat":1}'


def test_policy_deny_never_asks(monkeypatch):
    _config_with(monkeypatch, approval={"camera": "deny"})
    monkeypatch.setattr(approval, "prompt_dangerous_approval",
                        lambda *a, **k: pytest.fail("policy deny must not prompt"))
    calls = []

    payload = json.loads(device_request_tool("camera", callback=lambda p: calls.append(p)))

    assert calls == []
    assert payload["denied"] is True
    assert "device.approval.camera is deny" in payload["reason"]


def test_policy_allow_skips_the_gate(monkeypatch):
    _config_with(monkeypatch, approval={"camera": "allow"})
    monkeypatch.setattr(approval, "_run_approval_gate",
                        lambda **kw: pytest.fail("policy allow must not reach the gate"))
    payload = json.loads(device_request_tool("camera", callback=lambda p: {"value": "shot"}))
    assert payload["success"] is True and payload["value"] == "shot"


def test_ask_with_nobody_to_answer_fails_closed(monkeypatch):
    """A cron/unattended context has no human to prompt: the request is blocked, not auto-approved."""
    _config_with(monkeypatch)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr("tools.approval_context._get_approval_mode", lambda: "manual")
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    calls = []

    payload = json.loads(device_request_tool("microphone", callback=lambda p: calls.append(p)))

    assert calls == []
    assert payload["denied"] is True
    assert "requires approval" in payload["reason"]


# ── the closed vocabulary is one vocabulary ────────────────────────────────────────────────────


def test_tool_vocabulary_matches_the_wire_contract():
    """``tools/device_tool.py`` mirrors ``DeviceCapability``: a divergence would offer the model a
    capability no client can be asked for (or hide one it can)."""
    from tui_gateway.contracts.liveness import DeviceCapability

    assert CAPABILITIES == tuple(member.value for member in DeviceCapability)


def test_tool_is_registered_in_the_mobile_device_toolset():
    from tools.registry import registry

    entry = registry.get_entry("device_request")
    assert entry is not None
    assert entry.toolset == "mobile_device"
