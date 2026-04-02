import json
import os

from tools.env_passthrough import clear_env_passthrough, get_all_passthrough
from tools.secrets_tool import secrets_tool, set_secrets_request_callback


def setup_function(_fn):
    clear_env_passthrough()
    set_secrets_request_callback(None)


def test_list_returns_names_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    result = json.loads(secrets_tool({"action": "list"}))
    assert "OPENAI_API_KEY" in result["secrets"]
    assert "PATH" not in result["secrets"]
    assert "sk-test-secret" not in json.dumps(result)


def test_check_splits_configured_and_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    result = json.loads(secrets_tool({"action": "check", "keys": ["OPENAI_API_KEY", "MISSING_API_KEY"]}))
    assert result["configured"] == ["OPENAI_API_KEY"]
    assert result["missing"] == ["MISSING_API_KEY"]
    assert result["rejected"] == []


def test_check_rejects_non_secret_like_vars(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/tmp")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("SESSION_COOKIE_NAME", "sid")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    result = json.loads(secrets_tool({"action": "check", "keys": ["PATH", "HOME", "SSH_AUTH_SOCK", "SESSION_COOKIE_NAME", "OPENAI_API_KEY"]}))
    assert result["configured"] == ["OPENAI_API_KEY"]
    assert sorted(result["rejected"]) == ["HOME", "PATH", "SESSION_COOKIE_NAME", "SSH_AUTH_SOCK"]


def test_request_uses_secure_callback():
    calls = []

    def fake_callback(var_name, prompt, metadata=None):
        calls.append((var_name, prompt, metadata))
        return {"success": True, "skipped": False, "message": "stored"}

    set_secrets_request_callback(fake_callback)
    result = json.loads(secrets_tool({
        "action": "request",
        "key": "TENOR_API_KEY",
        "description": "Tenor API key",
        "instructions": "Find it in Tenor dashboard",
        "prompt": "Enter Tenor API key",
    }))
    assert result["success"] is True
    assert result["stored"] is True
    assert calls[0][0] == "TENOR_API_KEY"
    assert calls[0][1] == "Enter Tenor API key"
    assert calls[0][2]["description"] == "Tenor API key"
    assert calls[0][2]["instructions"] == "Find it in Tenor dashboard"


def test_request_without_callback_returns_hint():
    result = json.loads(secrets_tool({"action": "request", "key": "TENOR_API_KEY"}))
    assert result["success"] is False
    assert "local cli" in result["hint"].lower()


def test_delete_clears_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    result = json.loads(secrets_tool({"action": "delete", "key": "OPENAI_API_KEY"}))
    assert result["success"] is True
    assert "OPENAI_API_KEY" not in os.environ


def test_delete_removes_key_from_env_file(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    env_path = hermes_home / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-test-secret\nKEEP_ME=value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")

    result = json.loads(secrets_tool({"action": "delete", "key": "OPENAI_API_KEY"}))
    assert result["success"] is True
    content = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" not in content
    assert "KEEP_ME=value" in content


def test_inject_registers_existing_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    result = json.loads(secrets_tool({"action": "inject", "keys": ["OPENAI_API_KEY", "MISSING_API_KEY"]}))
    assert result["injected"] == ["OPENAI_API_KEY"]
    assert result["missing"] == ["MISSING_API_KEY"]
    assert "OPENAI_API_KEY" in get_all_passthrough()
