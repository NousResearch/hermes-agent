"""TUI/Desktop model config writes must respect fixed model policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


_FIXED_POLICY = {
    "model_policy": {
        "fixed_model": "gpt-5.5",
        "forbid_lower_fallback": True,
    }
}


def _switch_result(model: str):
    return SimpleNamespace(
        new_model=model,
        target_provider="openai-codex",
        base_url="",
    )


def test_persist_model_switch_rejects_lower_model_under_fixed_policy(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _FIXED_POLICY)
    from tui_gateway.server import _persist_model_switch

    with patch("cli.save_config_value", return_value=True) as save_config_value:
        with pytest.raises(ValueError) as raised:
            _persist_model_switch(_switch_result("gpt-5.4-mini"))

    assert "model policy" in str(raised.value).lower()
    assert "gpt-5.5" in str(raised.value)
    save_config_value.assert_not_called()


def test_persist_model_switch_preserves_existing_behavior_without_fixed_policy(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    from tui_gateway.server import _persist_model_switch

    with patch("cli.save_config_value", return_value=True) as save_config_value:
        _persist_model_switch(_switch_result("gpt-5.4-mini"))

    save_config_value.assert_any_call("model.default", "gpt-5.4-mini")
    save_config_value.assert_any_call("model.provider", "openai-codex")
