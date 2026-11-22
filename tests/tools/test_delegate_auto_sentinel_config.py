#!/usr/bin/env python3
"""Regression: delegation.provider literal 'auto' must inherit from parent."""
import unittest
from types import SimpleNamespace
from tools.delegate_tool_config import _resolve_delegation_credentials

def _parent():
    return SimpleNamespace(provider="openrouter", requested_provider="openrouter",
        base_url="https://openrouter.ai/api/v1", model="parent-model",
        request_overrides=None, _credential_pool=None)

class AutoSentinelInheritTests(unittest.TestCase):
    def test_auto_provider_inherits_everything(self):
        for spelling in ("auto", "AUTO", " Auto "):
            with self.subTest(spelling=spelling):
                creds = _resolve_delegation_credentials({"provider": spelling}, _parent())
                self.assertIsNone(creds["provider"])
                self.assertIsNone(creds["base_url"])
                self.assertIsNone(creds["api_key"])
                self.assertIsNone(creds["api_mode"])
    def test_auto_model_still_passes_through(self):
        creds = _resolve_delegation_credentials({"provider": "auto", "model": "auto"}, _parent())
        self.assertEqual(creds["model"], "auto")
        self.assertIsNone(creds["provider"])
    def test_explicit_provider_untouched(self):
        creds = _resolve_delegation_credentials(
            {"provider": "deepseek", "model": "deepseek-chat",
             "base_url": "https://api.deepseek.com/v1", "api_key": "sk-test"}, _parent())
        self.assertEqual(creds["model"], "deepseek-chat")
        self.assertEqual(creds["api_key"], "sk-test")
        self.assertIsNotNone(creds["provider"])

if __name__ == "__main__":
    unittest.main()
