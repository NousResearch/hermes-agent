"""Tests for the TUI ``/model reset`` handler (NS-563).

A session-scoped ``/model`` switch pins ``session["model_override"]``, which
persists to the session DB row and is restored on resume. Before ``/model
reset`` the only escape from a stale pin was ``/new`` (losing history).
``_reset_session_model_override`` clears the pin, returns the live agent to
the config.yaml default, and persists the cleared state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway.server import _reset_session_model_override


def _session(**extra) -> dict:
    base = {
        "session_key": "tui-reset-test",
        "history": [],
        "agent": SimpleNamespace(provider="openrouter", model="deepseek/deepseek-v4"),
        "model_override": {
            "model": "deepseek/deepseek-v4",
            "provider": "openrouter",
            "base_url": None,
            "api_mode": None,
        },
    }
    base.update(extra)
    return base


class TestResetClearsPin:
    def test_override_and_one_turn_restore_removed(self) -> None:
        session = _session(one_turn_model_restore={"model": "old"})
        with patch(
            "tui_gateway.server._config_model_target",
            return_value=("deepseek/deepseek-v4", "openrouter"),
        ), patch("tui_gateway.server._persist_live_session_runtime") as persist:
            result = _reset_session_model_override("sid", session)
        assert "model_override" not in session
        assert "one_turn_model_restore" not in session
        persist.assert_called_once_with(session)
        assert result["confirm_required"] is False
        assert result["scope"] == "session"

    def test_config_model_seen_cleared_so_adopt_sync_reruns(self) -> None:
        session = _session(config_model_seen=("some/model", "prov"))
        with patch(
            "tui_gateway.server._config_model_target",
            return_value=("deepseek/deepseek-v4", "openrouter"),
        ), patch("tui_gateway.server._persist_live_session_runtime"):
            _reset_session_model_override("sid", session)
        assert "config_model_seen" not in session


class TestResetSwitchesAgentBack:
    def test_switches_when_agent_differs_from_default(self) -> None:
        session = _session()
        with patch(
            "tui_gateway.server._config_model_target",
            return_value=("anthropic/claude-sonnet-4.6", "anthropic"),
        ), patch("tui_gateway.server._apply_model_switch") as apply_switch, patch(
            "tui_gateway.server._persist_live_session_runtime"
        ):
            result = _reset_session_model_override("sid", session)
        apply_switch.assert_called_once()
        args, kwargs = apply_switch.call_args
        assert args[2] == "anthropic/claude-sonnet-4.6 --provider anthropic"
        # Restoring the default must not create a fresh pin or write config.
        assert kwargs["pin_session_override"] is False
        assert kwargs["persist_override"] is False
        assert result["value"] == "anthropic/claude-sonnet-4.6"
        assert "cleared" in result["warning"]

    def test_no_switch_when_agent_already_on_default(self) -> None:
        session = _session()
        with patch(
            "tui_gateway.server._config_model_target",
            return_value=("deepseek/deepseek-v4", "openrouter"),
        ), patch("tui_gateway.server._apply_model_switch") as apply_switch, patch(
            "tui_gateway.server._persist_live_session_runtime"
        ):
            result = _reset_session_model_override("sid", session)
        apply_switch.assert_not_called()
        # The pin is still cleared and persisted even without a live switch.
        assert "model_override" not in session
        assert "cleared" in result["warning"]

    def test_no_switch_without_agent(self) -> None:
        session = _session(agent=None)
        with patch(
            "tui_gateway.server._config_model_target",
            return_value=("anthropic/claude-sonnet-4.6", "anthropic"),
        ), patch("tui_gateway.server._apply_model_switch") as apply_switch, patch(
            "tui_gateway.server._persist_live_session_runtime"
        ):
            _reset_session_model_override("sid", session)
        apply_switch.assert_not_called()


class TestResetWithoutActivePin:
    def test_reports_no_override_active(self) -> None:
        session = _session()
        session.pop("model_override")
        with patch(
            "tui_gateway.server._config_model_target",
            return_value=("deepseek/deepseek-v4", "openrouter"),
        ), patch("tui_gateway.server._persist_live_session_runtime"):
            result = _reset_session_model_override("sid", session)
        assert "No session model override" in result["warning"]

    def test_provider_only_difference_triggers_switch(self) -> None:
        session = _session(
            agent=SimpleNamespace(provider="openrouter", model="deepseek/deepseek-v4")
        )
        with patch(
            "tui_gateway.server._config_model_target",
            return_value=("deepseek/deepseek-v4", "nousresearch"),
        ), patch("tui_gateway.server._apply_model_switch") as apply_switch, patch(
            "tui_gateway.server._persist_live_session_runtime"
        ):
            _reset_session_model_override("sid", session)
        apply_switch.assert_called_once()
        assert (
            apply_switch.call_args[0][2]
            == "deepseek/deepseek-v4 --provider nousresearch"
        )


class TestResetDispatch:
    """/model reset routed through _apply_model_switch reaches the handler."""

    def test_apply_model_switch_dispatches_reset(self) -> None:
        session = _session()
        from tui_gateway.server import _apply_model_switch

        with patch(
            "tui_gateway.server._reset_session_model_override",
            return_value={"value": "x", "warning": "", "confirm_required": False,
                          "scope": "session"},
        ) as reset:
            result = _apply_model_switch("sid", session, "reset")
        reset.assert_called_once_with("sid", session)
        assert result["scope"] == "session"

    def test_reset_with_flags_raises_value_error(self) -> None:
        import pytest

        session = _session()
        from tui_gateway.server import _apply_model_switch

        with pytest.raises(ValueError, match="reset"):
            _apply_model_switch("sid", session, "reset --global")
