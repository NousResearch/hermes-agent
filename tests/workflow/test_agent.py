"""Step config resolves a real Hermes profile's model, then the card override."""

from pathlib import Path

from workflow.agent import is_failed_reply, is_user_fixable, resolve_step_model


def test_card_model_wins(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "profiles" / "designer").mkdir(parents=True)
    (home / "profiles" / "designer" / "config.yaml").write_text(
        "model:\n  default: profile-model\n  provider: anthropic\n",
        encoding="utf-8",
    )
    model, provider, profile = resolve_step_model({"profile": "designer", "model": "card-model"})
    assert model == "card-model"
    assert provider == "anthropic"
    assert profile == "designer"


def test_profile_fills_empty_model(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "profiles" / "reviewer").mkdir(parents=True)
    (home / "profiles" / "reviewer" / "config.yaml").write_text(
        "model:\n  default: profile-model\n  provider: openai\n",
        encoding="utf-8",
    )
    model, provider, profile = resolve_step_model({"profile": "reviewer"})
    assert model == "profile-model"
    assert provider == "openai"
    assert profile == "reviewer"


def test_placeholder_model_falls_through_to_profile(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "profiles" / "designer").mkdir(parents=True)
    (home / "profiles" / "designer" / "config.yaml").write_text(
        "model:\n  default: profile-model\n  provider: nous\n",
        encoding="utf-8",
    )
    model, provider, profile = resolve_step_model({"profile": "designer", "model": "claude-opus-4.8"})
    assert model == "profile-model"
    assert provider == "nous"
    assert profile == "designer"


def test_missing_profile_falls_back_to_default(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n  default: default-model\n  provider: nous\n",
        encoding="utf-8",
    )
    model, provider, profile = resolve_step_model({"profile": "designer"})
    assert model == "default-model"
    assert provider == "nous"
    assert profile == "designer"


def test_http_error_is_a_failed_reply():
    assert is_failed_reply("HTTP 404: Model 'claude-opus-4.8' not found. The requested model does not exist in our configuration or OpenRouter catalog.")
    assert is_failed_reply(
        '"Could not resolve authentication method. Expected either api_key or auth_token to be set."'
    )
    assert not is_failed_reply("Implemented the header and ended with PASS")
    assert is_user_fixable("HTTP 400: Model parameter is required")
    assert is_user_fixable("Model parameter is required. Set a model on this step or in Hermes settings.")
    assert is_user_fixable("Could not resolve authentication method. Expected either api_key or auth_token to be set.")
