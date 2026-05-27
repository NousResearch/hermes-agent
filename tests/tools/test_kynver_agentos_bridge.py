import json
import io
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from tools.kynver_agentos_bridge import (
    KynverAgentOSClient,
    KynverAgentOSConfig,
    KynverAgentOSError,
    _load_profile_env,
    agentos_available,
    load_kynver_agentos_config,
)


def test_load_profile_env_parses_basic_dotenv(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\nKYNVER_API_URL=https://example.test/\nKYNVER_API_KEY='secret'\nKYNVER_AGENT_OS_SLUG=forge\n",
        encoding="utf-8",
    )

    assert _load_profile_env(env_path) == {
        "KYNVER_API_URL": "https://example.test/",
        "KYNVER_API_KEY": "secret",
        "KYNVER_AGENT_OS_SLUG": "forge",
    }


def test_load_config_prefers_process_env_over_profile_env(monkeypatch, tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KYNVER_API_URL=https://profile.example\nKYNVER_API_KEY=profile-key\nKYNVER_AGENT_OS_SLUG=profile\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("tools.kynver_agentos_bridge._active_env_path", lambda: env_path)

    cfg = load_kynver_agentos_config(
        {
            "KYNVER_API_URL": "https://env.example/",
            "KYNVER_API_KEY": "env-key",
            "KYNVER_AGENT_OS_SLUG": "forge",
            "KYNVER_FETCH_TIMEOUT_MS": "2500",
        }
    )

    assert cfg.api_url == "https://env.example"
    assert cfg.api_key == "env-key"
    assert cfg.slug == "forge"
    assert cfg.timeout == 2.5
    assert cfg.enabled is True


def test_agentos_available_requires_api_key():
    assert not agentos_available({"KYNVER_API_URL": "https://example.test", "KYNVER_AGENT_OS_SLUG": "forge"})
    assert agentos_available(
        {
            "KYNVER_API_URL": "https://example.test",
            "KYNVER_AGENT_OS_SLUG": "forge",
            "KYNVER_API_KEY": "key",
        }
    )


def test_api_path_scopes_relative_paths_to_agentos_slug():
    client = KynverAgentOSClient(KynverAgentOSConfig(api_url="https://example.test", api_key="k", slug="forge"))

    assert client.api_path("/stats") == "/api/agent-os/forge/stats"
    assert client.api_path("tasks?status=ready") == "/api/agent-os/forge/tasks?status=ready"
    assert client.api_path("/agent-os/ghost/stats") == "/api/agent-os/ghost/stats"
    assert client.api_path("/api/agent-os/ghost/stats") == "/api/agent-os/ghost/stats"


class _FakeResponse:
    def __init__(self, payload: str):
        self.payload = payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_request_sends_bearer_and_parses_json(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["auth"] = req.headers.get("Authorization")
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse('{"ok": true}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = KynverAgentOSClient(KynverAgentOSConfig(api_url="https://example.test", api_key="secret", slug="forge", timeout=3))

    result = client.post("/sessions", {"channel": "telegram"})

    assert result == {"ok": True}
    assert seen == {
        "url": "https://example.test/api/agent-os/forge/sessions",
        "timeout": 3,
        "auth": "Bearer secret",
        "method": "POST",
        "body": {"channel": "telegram"},
    }


def test_request_refuses_when_not_configured():
    client = KynverAgentOSClient(KynverAgentOSConfig(api_url="https://example.test", api_key="", slug="forge"))

    with pytest.raises(KynverAgentOSError, match="not configured"):
        client.get("/stats")


def test_http_error_redacts_bearer_token(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            hdrs=Message(),
            fp=io.BytesIO(b"bad Bearer secret-token token=abc123"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = KynverAgentOSClient(KynverAgentOSConfig(api_url="https://example.test", api_key="secret", slug="forge"))

    with pytest.raises(KynverAgentOSError) as exc:
        client.get("/stats")

    msg = str(exc.value)
    assert "secret-token" not in msg
    assert "abc123" not in msg
    assert "[REDACTED]" in msg
