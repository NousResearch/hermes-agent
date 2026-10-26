"""Inherited custom-provider identity with shared gateway URLs."""
import unittest
from unittest.mock import patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import _resolve_delegation_credentials


class TestDelegationCredentialResolution(unittest.TestCase):
    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    @patch("hermes_cli.runtime_provider.canonical_custom_identity")
    def test_no_override_disambiguates_shared_custom_url_by_parent_model(
        self, mock_canonical, mock_resolve
    ):
        """A switched custom model must not inherit the previous provider key."""
        mock_canonical.return_value = "custom:provider-b"
        mock_resolve.return_value = {
            "provider": "custom",
            "model": "model-b-default",
            "base_url": "https://shared.invalid/v1",
            "api_key": "provider-b-key",
            "api_mode": "chat_completions",
        }
        parent = _make_mock_parent(depth=0)
        parent.model = "model-b-current"
        parent.provider = "custom:provider-a"
        parent.base_url = "https://shared.invalid/v1"

        creds = _resolve_delegation_credentials(
            {"model": "", "provider": ""}, parent
        )

        self.assertEqual(creds["model"], "model-b-current")
        self.assertEqual(creds["provider"], "custom:provider-b")
        self.assertEqual(creds["api_key"], "provider-b-key")
        mock_canonical.assert_called_once_with(
            base_url="https://shared.invalid/v1",
            config_provider="custom:provider-a",
            model="model-b-current",
        )
        mock_resolve.assert_called_once_with(
            requested="custom:provider-b", target_model="model-b-current"
        )

    @patch("hermes_cli.runtime_provider.resolve_runtime_provider")
    @patch("hermes_cli.runtime_provider.canonical_custom_identity")
    def test_builtin_parent_keeps_ordinary_inheritance_path(
        self, mock_canonical, mock_resolve
    ):
        """A built-in parent must not be remapped by a matching custom URL."""
        parent = _make_mock_parent(depth=0)
        parent.base_url = "https://shared.invalid/v1"

        creds = _resolve_delegation_credentials(
            {"model": "", "provider": ""}, parent
        )

        self.assertIsNone(creds["provider"])
        self.assertIsNone(creds["base_url"])
        self.assertIsNone(creds["api_key"])
        mock_canonical.assert_not_called()
        mock_resolve.assert_not_called()

