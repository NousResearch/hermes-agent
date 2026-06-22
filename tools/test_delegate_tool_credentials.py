"""Unit tests for delegate_tool delegation credential resolution."""

import unittest
from unittest.mock import MagicMock, patch


class ResolveDelegationCredentialsTests(unittest.TestCase):
    def test_configured_api_key_overrides_provider_env(self):
        """delegation.api_key must be preferred over provider env-var lookup."""
        from tools import delegate_tool

        cfg = {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "api_key": "sk-explicit",
        }

        def fake_resolve(requested, target_model=None):
            return {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "api_key": "",  # env var not set
                "base_url": None,
                "api_mode": "chat_completions",
            }

        runtime_provider_module = MagicMock()
        runtime_provider_module.resolve_runtime_provider = fake_resolve

        with patch.dict("sys.modules", {"hermes_cli.runtime_provider": runtime_provider_module}):
            creds = delegate_tool._resolve_delegation_credentials(cfg, None)

        self.assertEqual(creds["api_key"], "sk-explicit")
        self.assertEqual(creds["provider"], "deepseek")
        self.assertEqual(creds["model"], "deepseek-v4-flash")

    def test_provider_env_used_when_api_key_not_configured(self):
        """Normal env-var lookup still works when delegation.api_key is absent."""
        from tools import delegate_tool

        cfg = {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
        }

        def fake_resolve(requested, target_model=None):
            return {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "api_key": "sk-from-env",
                "base_url": None,
                "api_mode": "chat_completions",
            }

        runtime_provider_module = MagicMock()
        runtime_provider_module.resolve_runtime_provider = fake_resolve

        with patch.dict("sys.modules", {"hermes_cli.runtime_provider": runtime_provider_module}):
            creds = delegate_tool._resolve_delegation_credentials(cfg, None)

        self.assertEqual(creds["api_key"], "sk-from-env")


if __name__ == "__main__":
    unittest.main()
