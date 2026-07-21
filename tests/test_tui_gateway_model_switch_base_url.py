"""Tests for `tui_gateway.server._persist_model_switch` and its interaction
with `custom_providers`.

Companion to `tests/hermes_cli/test_web_server_picker_base_url.py` — the
dashboard's `POST /api/model/set` and the chat TUI's `/model` slash command
both write `model.base_url` and share the same bug class: clearing
`base_url` when switching to a custom provider breaks routing for any
user fronting a self-hosted LLM gateway.

These tests pin the contract for the TUI side:
  * `/model` switching within a custom provider preserves `model.base_url`.
  * `/model` switching to a built-in provider clears `model.base_url`
    (existing behaviour — built-ins have hardcoded endpoints).
"""

from types import SimpleNamespace
from unittest.mock import patch

import yaml
import pytest


@pytest.fixture
def _config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with the standard custom-provider config."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_cfg(home, content: str):
    (home / "config.yaml").write_text(content)


def _read_model(home) -> dict:
    cfg = yaml.safe_load((home / "config.yaml").read_text()) or {}
    model = cfg.get("model")
    return model if isinstance(model, dict) else {}


class TestTuiPersistPreservesCustomProviderBaseUrl:
    """`/model` slash command must keep base_url for custom providers."""

    def test_switching_model_within_custom_provider_preserves_base_url(
        self, _config_home
    ):
        from tui_gateway.server import _persist_model_switch

        _write_cfg(_config_home, (
            "custom_providers:\n"
            "- name: my-litellm\n"
            "  base_url: http://192.168.1.10:4000\n"
            "  api_key: sk-test-custom\n"
            "  api_format: openai\n"
            "model:\n"
            "  default: gpt-4o-mini\n"
            "  provider: my-litellm\n"
            "  base_url: http://192.168.1.10:4000\n"
            "  api_key: sk-test-custom\n"
        ))

        # The switch handler returned no base_url override (model_switch
        # doesn't know about the custom provider's URL) — so the old code
        # would pop base_url. New code must check custom_providers first.
        result = SimpleNamespace(
            new_model="claude-3-haiku",
            target_provider="my-litellm",
            base_url=None,
        )
        _persist_model_switch(result)

        model = _read_model(_config_home)
        assert model.get("default") == "claude-3-haiku"
        assert model.get("provider") == "my-litellm"
        assert model.get("base_url") == "http://192.168.1.10:4000", (
            "/model slash command cleared base_url for a custom provider; "
            "regresses the self-hosted-gateway use case."
        )

    def test_provider_name_match_is_case_insensitive(self, _config_home):
        from tui_gateway.server import _persist_model_switch

        _write_cfg(_config_home, (
            "custom_providers:\n"
            "- name: My-LiteLLM\n"
            "  base_url: http://192.168.1.10:4000\n"
            "  api_key: sk-test\n"
            "model:\n"
            "  default: model-a\n"
            "  provider: My-LiteLLM\n"
            "  base_url: http://192.168.1.10:4000\n"
        ))

        result = SimpleNamespace(
            new_model="model-b",
            target_provider="my-litellm",  # lowercased
            base_url=None,
        )
        _persist_model_switch(result)

        assert _read_model(_config_home).get("base_url") == "http://192.168.1.10:4000"


class TestTuiPersistStillClearsBuiltInBaseUrl:
    """Built-in providers: clearing stale base_url is correct."""

    def test_switching_to_anthropic_clears_base_url(self, _config_home):
        from tui_gateway.server import _persist_model_switch

        _write_cfg(_config_home, (
            "custom_providers: []\n"
            "model:\n"
            "  default: gpt-4o-mini\n"
            "  provider: openai\n"
            "  base_url: https://stale-openai-mirror.example.com/v1\n"
            "  api_key: sk-test\n"
        ))

        result = SimpleNamespace(
            new_model="claude-haiku-4-5",
            target_provider="anthropic",
            base_url=None,
        )
        _persist_model_switch(result)

        model = _read_model(_config_home)
        assert model.get("provider") == "anthropic"
        # Built-in — stale base_url should be popped.
        assert "base_url" not in model or model.get("base_url") in ("", None)

    def test_explicit_base_url_in_result_always_used(self, _config_home):
        """If model_switch resolved a new base_url, use it — for both
        built-in and custom providers."""
        from tui_gateway.server import _persist_model_switch

        _write_cfg(_config_home, (
            "custom_providers:\n"
            "- name: my-litellm\n"
            "  base_url: http://192.168.1.10:4000\n"
            "  api_key: sk-test\n"
            "model:\n"
            "  default: old\n"
            "  provider: my-litellm\n"
            "  base_url: http://192.168.1.10:4000\n"
        ))

        result = SimpleNamespace(
            new_model="new",
            target_provider="my-litellm",
            base_url="http://192.168.1.10:4001",  # explicit new url
        )
        _persist_model_switch(result)

        assert _read_model(_config_home).get("base_url") == "http://192.168.1.10:4001"
