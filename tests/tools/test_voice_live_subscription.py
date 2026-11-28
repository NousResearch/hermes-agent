"""Native Live billing selection keeps Codex OAuth private and never falls back."""

import base64
import json

import httpx
import pytest

from hermes_cli import auth_codex
from hermes_cli.auth_constants import AuthError
from tools import voice_live


def _token(account="account-fixture"):
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"fixture.{payload}.fixture"


def _configure(monkeypatch, tmp_path, **live):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({
        "voice": {"voice_chat_mode": "gpt-live", "gpt_live": live},
    }), encoding="utf-8")


def test_subscription_request_uses_hermes_oauth_and_its_own_wire(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, auth="subscription", legacy_client_auth="api", model="api-model", voice="marin",
               subscription_model="gpt-live-1-codex", subscription_voice="cove")
    token = _token()
    refresh = []
    requests = []
    history = [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "Keep the existing Hermes conversation."}]}]

    def credentials(**kwargs):
        refresh.append(kwargs["refresh_if_expiring"])
        return {"api_key": token, "base_url": "https://chatgpt.com/backend-api/codex", "auth_mode": "chatgpt"}

    def transport(request):
        requests.append(request)
        return httpx.Response(201, text="v=0\r\nanswer", headers={
            "Location": "/realtime/calls/rtc_fixture", "Content-Type": "application/sdp"})

    client = httpx.Client
    monkeypatch.setattr(auth_codex, "resolve_codex_runtime_credentials", credentials)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(
        transport=httpx.MockTransport(transport), **kwargs))
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda *_: pytest.fail("paid API resolver"))
    status = voice_live.resolve_gpt_live_status()
    result = voice_live.create_webrtc_session("v=0 offer", history, expected_auth="subscription")

    assert status["auth"] == result["auth"] == "subscription"
    assert status["available"] and refresh == [False, True]
    request, = requests
    assert str(request.url) == (
        "https://chatgpt.com/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas")
    assert request.headers["Authorization"] == f"Bearer {token}"
    assert request.headers["ChatGPT-Account-Id"] == "account-fixture"
    assert request.headers["OpenAI-Alpha"] == "quicksilver=v2"
    payload = json.loads(request.content)
    assert payload["sdp"] == "v=0 offer" and "transport" not in payload
    assert payload["session"]["initial_items"] == history and "input" not in payload["session"]
    assert payload["session"]["delegation"] == {"type": "client"}
    assert payload["session"]["model"] == status["model"] == "gpt-live-1-codex"
    assert "audio" not in payload["session"]
    assert status["voice"] == "provider default"
    assert result["session"]["id"] == "rtc_fixture"
    assert result["transport"] == {"type": "webrtc", "sdp": "v=0\r\nanswer"}
    assert token not in json.dumps([status, result])
    assert "account-fixture" not in json.dumps([status, result])


@pytest.mark.parametrize("failure", ["missing", "account", "invalid-auth", "redirect", "rejected", "sdp", "timeout"])
def test_subscription_failure_cannot_select_paid_api(monkeypatch, tmp_path, failure):
    _configure(monkeypatch, tmp_path, auth="automatic" if failure == "invalid-auth" else "subscription")
    attempted = []
    private = "private-provider-detail"

    def credentials(**_kwargs):
        if failure == "missing":
            raise AuthError(provider="openai-codex", message=private, code="codex_auth_missing")
        return {"base_url": "https://chatgpt.com/backend-api/codex", "api_key": _token(None if failure == "account" else "account-fixture")}

    def transport(request):
        attempted.append(request)
        if failure == "timeout":
            raise httpx.TimeoutException("private timeout detail", request=request)
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "https://other.example/collect"})
        return httpx.Response(403 if failure == "rejected" else 200, text=private)

    client = httpx.Client
    monkeypatch.setattr(auth_codex, "resolve_codex_runtime_credentials", credentials)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(
        transport=httpx.MockTransport(transport), **kwargs))
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda *_: pytest.fail("paid API resolver"))
    monkeypatch.setattr(voice_live.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("paid API request"))
    with pytest.raises((ValueError, RuntimeError)) as error:
        voice_live.create_webrtc_session("v=0 offer", expected_auth="subscription")
    assert private not in str(error.value)
    assert len(attempted) == (0 if failure in {"missing", "account", "invalid-auth"} else 1)
    if not attempted:
        status = voice_live.resolve_gpt_live_status()
        assert status["available"] is False
        assert private not in json.dumps(status)


def test_status_is_read_only_and_session_auth_is_pinned(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, auth="subscription")
    calls = []

    def credentials(**kwargs):
        calls.append(kwargs)
        return {"api_key": _token(), "base_url": "https://chatgpt.com/backend-api/codex", "auth_mode": "chatgpt"}

    monkeypatch.setattr(auth_codex, "resolve_codex_runtime_credentials", credentials)
    status = voice_live.resolve_gpt_live_status()

    assert status["available"] is True
    assert calls == [{"refresh_if_expiring": False, "read_only": True}]

    with pytest.raises(RuntimeError, match="changed"):
        voice_live.create_webrtc_session("v=0 offer", expected_auth="api")

    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda *_: pytest.fail("paid API resolver"))
    assert len(calls) == 1


@pytest.mark.parametrize("legacy", [None, "api"])
def test_unpinned_client_never_receives_subscription_wire(monkeypatch, tmp_path, legacy):
    from io import BytesIO

    _configure(monkeypatch, tmp_path, auth="subscription", api_key="fixture",
               **({"legacy_client_auth": legacy} if legacy else {}))
    monkeypatch.setattr(voice_live, "_subscription_credentials", lambda: pytest.fail("subscription wire"))
    history = [{"type": "message", "content": []}]
    requests = []

    def respond(req, **kwargs):
        requests.append(json.loads(req.data))
        return BytesIO(b'{}')

    monkeypatch.setattr(voice_live.urllib.request, "urlopen", respond)
    if legacy:
        assert voice_live.create_webrtc_session("offer", history)["auth"] == "api"
        session = requests[0]["session"]
        assert session["input"] == history and session["model"] == "gpt-live-1"
        assert session["audio"]["output"]["voice"] == "marin"
    else:
        with pytest.raises(ValueError, match="upgrade"):
            voice_live.create_webrtc_session("offer", history)
        assert not requests


@pytest.mark.parametrize("route", [None, "https://relay.invalid/backend-api/codex"])
def test_subscription_requires_canonical_credential_route(monkeypatch, tmp_path, route):
    _configure(monkeypatch, tmp_path, auth="subscription", legacy_client_auth="api")
    monkeypatch.setattr(auth_codex, "resolve_codex_runtime_credentials", lambda **kw: {
        "api_key": _token(), "base_url": route})
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("voice request"))
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda *_: pytest.fail("paid API"))
    assert not voice_live.resolve_gpt_live_status()["available"]
    with pytest.raises(ValueError, match="route"):
        voice_live.create_webrtc_session("offer", expected_auth="subscription")


@pytest.mark.parametrize("document", ["false", "[]", "0", "null", "~", "voice: [", "voice: false", "voice: {gpt_live: false}", "exception"])
def test_unreadable_voice_policy_blocks_unpinned_billing(monkeypatch, tmp_path, document):
    from hermes_cli import config

    monkeypatch.setenv("OPENAI_API_KEY", "available-fixture")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(document)
    if document == "exception":
        def fail():
            raise OSError("private config detail")
        monkeypatch.setattr(config, "load_config", fail)
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda *_: pytest.fail("API resolver"))
    monkeypatch.setattr(voice_live, "_subscription_credentials", lambda **kw: pytest.fail("OAuth resolver"))
    for operation in [voice_live.resolve_gpt_live_status, lambda: voice_live.create_webrtc_session("offer")]:
        with pytest.raises(ValueError, match="configuration"):
            operation()
