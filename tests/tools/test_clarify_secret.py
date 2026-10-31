"""clarify(secret_env=...) collects a secret through the masked prompt without it reaching the model."""

import json

from agent.redact import redact_sensitive_text
from hermes_cli.config import load_env, save_env_value_secure
from tools.clarify_tool import clarify_tool
from tools.env_passthrough import is_env_passthrough
from tools.skills_tool import bind_turn_secret_capture

SECRET = "wx-live-" + "7f3c9d2e1b8a4f60c5d7e9a1"  # built by concatenation: a fake value
QUESTION = [{"question": "Weather API key, used by the forecast script"}]


def _never_called(_questions):
    raise AssertionError("a secret question must not reach the plain clarify card")


def test_secret_question_stores_masked_value_and_returns_only_its_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen = []

    def masked_prompt(name, prompt, metadata=None):  # what every surface's capture callback does
        seen.append((name, prompt))
        return {**save_env_value_secure(name, SECRET), "skipped": False}

    with bind_turn_secret_capture(masked_prompt):
        raw = clarify_tool(QUESTION, callback=_never_called, secret_env="WEATHER_DEMO_KEY")

    assert SECRET not in raw
    result = json.loads(raw)
    assert result["outcome"] == "submitted"
    assert result["responses"][0]["stored_as"] == "WEATHER_DEMO_KEY"
    assert "$WEATHER_DEMO_KEY" in result["notice"]
    # A kernel started before the capture keeps its old environment; the model must be told how to refresh it.
    assert "reset=true" in result["notice"]
    assert seen == [("WEATHER_DEMO_KEY", QUESTION[0]["question"])]
    assert load_env()["WEATHER_DEMO_KEY"] == SECRET
    assert is_env_passthrough("WEATHER_DEMO_KEY")
    assert SECRET not in redact_sensitive_text(f"curl -H 'X-Key: {SECRET}' → {SECRET}")


def test_messaging_surface_refuses_without_prompting_or_accepting_a_value(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # A Telegram turn in a process that also serves an interactive viewer (global callback wired).
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    from tools import skills_tool
    monkeypatch.setattr(skills_tool, "_secret_capture_callback",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("prompted on messaging")))

    result = json.loads(clarify_tool(QUESTION, callback=_never_called, secret_env="WEATHER_DEMO_KEY"))

    assert "paste it into chat" in result["error"]
    assert "WEATHER_DEMO_KEY" not in load_env()
