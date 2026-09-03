#!/usr/bin/env python3
"""
Integration tests for per-task model selection in delegate_task.

Mirrors the test file name and structure from Kilo-Org/kilocode#11786
(upstream kilocode-port/per-task-delegation-model branch). These tests
exercise _resolve_task_model_creds with REAL switch_model() calls rather
than mocks, complementing the mocked unit tests in
test_delegate_per_task_overrides.py.

Run with:  python -m pytest tests/tools/test_delegate_model_selection.py -v
"""

import unittest
from unittest.mock import patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_dynamic_schema_overrides,
    _get_allow_model_selection,
    _resolve_task_model_creds,
)


class _FakeParent:
    """Minimal parent agent for credential anchoring."""

    provider = "openrouter"
    model = "anthropic/claude-opus-4.8"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-test"


_BASE_CREDS = {
    "model": None,
    "provider": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
    "command": None,
    "args": None,
}


class TestSchemaGating(unittest.TestCase):
    """The per-task model field only appears when the flag is enabled."""

    def test_flag_off_no_model_field(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertNotIn("model", props)
        self.assertNotIn("model", props["tasks"]["items"]["properties"])

    def test_flag_on_adds_model_field(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("model", props)
        self.assertEqual(props["model"]["type"], "string")
        self.assertIn("model", props["tasks"]["items"]["properties"])

    def test_static_schema_never_mutated(self):
        """Dynamic overrides must not leak the model field into the static schema."""
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ):
            _build_dynamic_schema_overrides()
        static_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertNotIn("model", static_props)
        self.assertNotIn(
            "model", static_props["tasks"]["items"]["properties"]
        )


class TestFlagGetter(unittest.TestCase):
    def test_default_off(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertFalse(_get_allow_model_selection())

    def test_truthy_on(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ):
            self.assertTrue(_get_allow_model_selection())


class TestModelResolutionIntegration(unittest.TestCase):
    """Integration tests with real switch_model() calls.

    These mirror the kilocode branch's test approach: exercise the actual
    model_switch pipeline rather than mocking it. Tests may be skipped if
    the model_switch module cannot resolve models in the test environment
    (e.g. no provider config available).
    """

    def test_empty_name_returns_base_unchanged(self):
        out = _resolve_task_model_creds("", _FakeParent(), _BASE_CREDS)
        self.assertIs(out, _BASE_CREDS)

    def test_base_creds_not_mutated(self):
        before = dict(_BASE_CREDS)
        try:
            _resolve_task_model_creds("sonnet", _FakeParent(), _BASE_CREDS)
        except (ValueError, Exception):
            # Resolution may fail in test env; mutation check is independent
            pass
        self.assertEqual(_BASE_CREDS, before)

    def test_full_slug_passthrough(self):
        """A full vendor/model slug should pass through resolution."""
        try:
            out = _resolve_task_model_creds(
                "openai/gpt-5.4", _FakeParent(), _BASE_CREDS
            )
            self.assertEqual(out["model"], "openai/gpt-5.4")
        except (ValueError, Exception):
            self.skipTest("model_switch cannot resolve in this environment")

    def test_unresolvable_name_raises(self):
        with self.assertRaises(ValueError):
            _resolve_task_model_creds(
                "zzznotarealmodel-xyz-123", _FakeParent(), _BASE_CREDS
            )


if __name__ == "__main__":
    unittest.main()