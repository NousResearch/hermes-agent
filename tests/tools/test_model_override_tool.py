#!/usr/bin/env python3
"""Tests for the agent-facing model_override tool.

Covers: gateway-bridge resolution, set/clear snapshot-restore semantics,
status reporting, and the non-gateway error path. No real LLM calls.
"""

import json
import types
import unittest
from unittest.mock import MagicMock, patch

from tools.model_override_tool import model_override


def _make_agent(**kwargs):
    """Mock parent agent with the fields model_override reads."""
    agent = MagicMock()
    agent._gateway_session_key = kwargs.get("session_key", "agent:main:discord:dm:123")
    agent.provider = kwargs.get("provider", "opencode-go")
    agent.model = kwargs.get("model", "deepseek-v4-flash")
    agent.base_url = kwargs.get("base_url", "https://opencode.ai/zen/go/v1")
    agent.api_key = kwargs.get("api_key", "key-abc")
    return agent


def _make_runner(**kwargs):
    """Fake gateway runner with the session-state surface model_override uses."""
    runner = MagicMock()

    state = types.SimpleNamespace()
    state.conversation = types.SimpleNamespace(model_override=kwargs.get("override"))

    def _peek(key):
        return state if key == runner._peek_key else None

    runner._peek_key = kwargs.get("session_key", "agent:main:discord:dm:123")
    runner._peek_session_state = MagicMock(side_effect=_peek)
    runner._session_state = MagicMock(return_value=state)
    runner._evict_cached_agent = MagicMock()
    runner.async_session_store = MagicMock()
    return runner, state


class TestModelOverrideNonGateway(unittest.TestCase):
    def test_requires_gateway(self):
        with patch("tools.model_override_tool._get_gateway_runner", return_value=None):
            out = json.loads(model_override(action="status", parent_agent=_make_agent()))
        self.assertIn("error", out)
        self.assertIn("gateway", out["error"].lower())

    def test_requires_session_key(self):
        runner, _ = _make_runner()
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            out = json.loads(
                model_override(action="status", parent_agent=_make_agent(session_key=None))
            )
        self.assertIn("error", out)

    def test_unknown_action(self):
        runner, _ = _make_runner()
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            out = json.loads(
                model_override(action="bogus", parent_agent=_make_agent())
            )
        self.assertIn("error", out)


class TestModelOverrideStatus(unittest.TestCase):
    def test_status_no_override(self):
        runner, _ = _make_runner(override=None)
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            out = json.loads(model_override(action="status", parent_agent=_make_agent()))
        self.assertTrue(out["ok"])
        self.assertIsNone(out["override_model"])
        self.assertEqual(out["effective_model"], "deepseek-v4-flash")

    def test_status_with_override(self):
        override = {"model": "grok-4.5", "provider": "xai-oauth"}
        runner, _ = _make_runner(override=override)
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            out = json.loads(model_override(action="status", parent_agent=_make_agent()))
        self.assertEqual(out["override_model"], "grok-4.5")
        self.assertEqual(out["override_provider"], "xai-oauth")


class TestModelOverrideSet(unittest.TestCase):
    def test_set_writes_override_and_snapshots_prior(self):
        prior = {"model": "deepseek-v4-flash", "provider": "opencode-go"}
        runner, state = _make_runner(override=prior)
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            with patch(
                "hermes_cli.model_switch.switch_model"
            ) as mock_switch:
                mock_switch.return_value = types.SimpleNamespace(
                    success=True,
                    new_model="grok-4.5",
                    target_provider="xai-oauth",
                    api_key="xk",
                    base_url="https://api.x.ai/v1",
                    api_mode="codex_responses",
                    error_message=None,
                )
                out = json.loads(
                    model_override(
                        action="set",
                        model="grok-4.5",
                        provider="xai-oauth",
                        parent_agent=_make_agent(),
                    )
                )
        self.assertTrue(out["ok"])
        self.assertEqual(out["model"], "grok-4.5")
        # In-memory override written with restore marker.
        written = state.conversation.model_override
        self.assertEqual(written["model"], "grok-4.5")
        self.assertEqual(written["provider"], "xai-oauth")
        self.assertEqual(written["_restore_override"]["model"], "deepseek-v4-flash")
        # Store write-through + cache eviction.
        runner.async_session_store.set_model_override.assert_called()
        runner._evict_cached_agent.assert_called_once_with(runner._peek_key)

    def test_set_requires_model(self):
        runner, _ = _make_runner(override=None)
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            out = json.loads(model_override(action="set", parent_agent=_make_agent()))
        self.assertIn("error", out)

    def test_set_resolution_failure(self):
        runner, _ = _make_runner(override=None)
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            with patch("hermes_cli.model_switch.switch_model") as mock_switch:
                mock_switch.return_value = types.SimpleNamespace(
                    success=False, error_message="Unknown model"
                )
                out = json.loads(
                    model_override(
                        action="set", model="nope", parent_agent=_make_agent()
                    )
                )
        self.assertIn("error", out)


class TestModelOverrideClear(unittest.TestCase):
    def test_clear_restores_prior_override(self):
        override = {
            "model": "grok-4.5",
            "provider": "xai-oauth",
            "_restore_override": {"model": "deepseek-v4-flash", "provider": "opencode-go"},
        }
        runner, state = _make_runner(override=override)
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            out = json.loads(model_override(action="clear", parent_agent=_make_agent()))
        self.assertTrue(out["ok"])
        self.assertEqual(out["restored"], "deepseek-v4-flash")
        restored = state.conversation.model_override
        self.assertEqual(restored["model"], "deepseek-v4-flash")
        self.assertNotIn("_restore_override", restored)
        runner._evict_cached_agent.assert_called_once_with(runner._peek_key)

    def test_clear_without_prior_override_clears(self):
        override = {"model": "grok-4.5", "provider": "xai-oauth", "_restore_override": None}
        runner, state = _make_runner(override=override)
        with patch("tools.model_override_tool._get_gateway_runner", return_value=runner):
            out = json.loads(model_override(action="clear", parent_agent=_make_agent()))
        self.assertEqual(out["restored"], "global default")
        self.assertIsNone(state.conversation.model_override)


class TestModelOverrideRegistry(unittest.TestCase):
    def test_module_registers(self):
        from tools.registry import registry

        entry = registry.get_entry("model_override")
        self.assertIsNotNone(entry, "model_override must be registered")
        schema = entry.schema
        self.assertEqual(schema["name"], "model_override")
        self.assertIn("action", schema["parameters"]["properties"])
        self.assertEqual(schema["parameters"]["properties"]["action"]["enum"],
                         ["set", "clear", "status"])

    def test_toolset_includes_tool(self):
        import toolsets

        self.assertIn("model_override", toolsets.TOOLSETS["delegation"]["tools"])


if __name__ == "__main__":
    unittest.main()
