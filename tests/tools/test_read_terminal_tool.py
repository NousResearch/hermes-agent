#!/usr/bin/env python3
"""Tests for tools.read_terminal_tool error guidance."""

import json
import unittest

from tools.read_terminal_tool import read_terminal_tool


class TestReadTerminalEmptyBufferGuidance(unittest.TestCase):
    def test_empty_read_error_is_actionable(self):
        """An empty in-app buffer must tell the agent not to retry and what
        to use instead — weak models otherwise poll read_terminal in a loop
        (observed 14x in session 20260731_230301_2c274e)."""
        result = read_terminal_tool(callback=lambda **_: "")
        payload = json.loads(result)
        self.assertIn("error", payload)
        msg = payload["error"]
        self.assertIn("No in-app terminal answered", msg)
        self.assertIn("do not retry", msg.lower())
        self.assertIn("execute_code", msg)

    def test_desktop_only_error_unchanged(self):
        result = read_terminal_tool(callback=None)
        payload = json.loads(result)
        self.assertEqual(
            payload["error"],
            "read_terminal is only available in the Hermes desktop app.",
        )


if __name__ == "__main__":
    unittest.main()
