"""Gateway self-restart hard block tests for approval guards."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tools.approval import check_all_command_guards, check_dangerous_command


class TestGatewaySelfRestartGuard(unittest.TestCase):
    def _gateway_env(self):
        return patch.dict(
            os.environ,
            {
                "_HERMES_GATEWAY": "1",
                "HERMES_APPROVAL_MODE": "off",
            },
            clear=False,
        )

    def test_systemctl_restart_own_gateway_is_blocked_even_when_approval_off(self):
        with self._gateway_env():
            result = check_all_command_guards(
                "systemctl --user restart hermes-gateway",
                "local",
            )

        self.assertFalse(result["approved"])
        self.assertTrue(result.get("hardline"))
        self.assertIn("gateway self-protection", result["message"])

    def test_quoted_systemctl_restart_own_gateway_is_blocked(self):
        with self._gateway_env():
            result = check_all_command_guards(
                "\"systemctl\" --user restart hermes-gateway",
                "local",
            )

        self.assertFalse(result["approved"])
        self.assertIn("gateway self-protection", result["message"])

    def test_quoted_hermes_gateway_restart_is_blocked(self):
        with self._gateway_env():
            result = check_all_command_guards(
                "'hermes' gateway restart",
                "local",
            )

        self.assertFalse(result["approved"])
        self.assertIn("gateway self-protection", result["message"])

    def test_systemctl_restart_profile_gateway_is_blocked(self):
        with self._gateway_env():
            result = check_all_command_guards(
                "systemctl --user restart hermes-gateway-selina-email.service",
                "local",
            )

        self.assertFalse(result["approved"])
        self.assertIn("self-protection", result["message"])

    def test_legacy_dangerous_command_entrypoint_also_blocks(self):
        with self._gateway_env():
            result = check_dangerous_command(
                "hermes gateway restart",
                "local",
            )

        self.assertFalse(result["approved"])
        self.assertTrue(result.get("hardline"))

    def test_non_gateway_context_still_uses_normal_approval_path(self):
        with patch.dict(os.environ, {"HERMES_APPROVAL_MODE": "off"}, clear=True):
            result = check_all_command_guards(
                "systemctl --user restart hermes-gateway",
                "local",
            )

        self.assertTrue(result["approved"])


if __name__ == "__main__":
    unittest.main()
