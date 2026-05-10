#!/usr/bin/env python3
"""
Tests for subagent resource limit diagnostics.

Tests the post-hoc resource limit checks (max_context_tokens, context_growth_ratio)
that mark subagents as failed after completion when limits are exceeded.

Run with:  python -m pytest tests/tools/test_delegate_resource_limits.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _get_max_context_tokens,
    _get_context_growth_ratio,
    _run_single_child,
)


class TestResourceLimitConfigReaders(unittest.TestCase):
    """Test config reader functions for resource limits."""

    @patch("tools.delegate_tool._load_config")
    def test_max_context_tokens_default(self, mock_load_config):
        """Default value when config key is missing."""
        mock_load_config.return_value = {}
        self.assertEqual(_get_max_context_tokens(), 100000)

    @patch("tools.delegate_tool._load_config")
    def test_max_context_tokens_custom_value(self, mock_load_config):
        """Custom value from config."""
        mock_load_config.return_value = {"max_context_tokens": 50000}
        self.assertEqual(_get_max_context_tokens(), 50000)

    @patch("tools.delegate_tool._load_config")
    def test_max_context_tokens_floor_constraint(self, mock_load_config):
        """Floor constraint at 10k tokens."""
        mock_load_config.return_value = {"max_context_tokens": 5000}
        self.assertEqual(_get_max_context_tokens(), 10000)

    @patch("tools.delegate_tool._load_config")
    def test_max_context_tokens_invalid_value(self, mock_load_config):
        """Invalid value falls back to default."""
        mock_load_config.return_value = {"max_context_tokens": "invalid"}
        self.assertEqual(_get_max_context_tokens(), 100000)

    @patch("tools.delegate_tool._load_config")
    def test_context_growth_ratio_default(self, mock_load_config):
        """Default value when config key is missing."""
        mock_load_config.return_value = {}
        self.assertEqual(_get_context_growth_ratio(), 2.5)

    @patch("tools.delegate_tool._load_config")
    def test_context_growth_ratio_custom_value(self, mock_load_config):
        """Custom value from config."""
        mock_load_config.return_value = {"context_growth_ratio": 3.0}
        self.assertEqual(_get_context_growth_ratio(), 3.0)

    @patch("tools.delegate_tool._load_config")
    def test_context_growth_ratio_floor_constraint(self, mock_load_config):
        """Floor constraint at 1.0."""
        mock_load_config.return_value = {"context_growth_ratio": 0.5}
        self.assertEqual(_get_context_growth_ratio(), 1.0)

    @patch("tools.delegate_tool._load_config")
    def test_context_growth_ratio_invalid_value(self, mock_load_config):
        """Invalid value falls back to default."""
        mock_load_config.return_value = {"context_growth_ratio": "invalid"}
        self.assertEqual(_get_context_growth_ratio(), 2.5)


class TestResourceLimitChecks(unittest.TestCase):
    """Test post-hoc resource limit checks in _run_single_child."""

    def _make_mock_child(self, input_tokens=1000, output_tokens=2000, model="test-model"):
        """Create a mock child agent with token counts."""
        child = MagicMock()
        child.session_prompt_tokens = input_tokens
        child.session_completion_tokens = output_tokens
        child.model = model
        child.session_estimated_cost_usd = 0.01
        child.session_reasoning_tokens = 0
        child._delegate_role = "leaf"
        
        # Mock run_conversation to return a simple result
        child.run_conversation.return_value = {
            "final_response": "Task completed",
            "messages": [],
            "completed": True,
            "interrupted": False,
        }
        
        return child

    @patch("tools.delegate_tool._get_max_context_tokens")
    @patch("tools.delegate_tool._get_context_growth_ratio")
    @patch("tools.delegate_tool._get_child_timeout")
    @patch("tools.delegate_tool.file_state")
    def test_subagent_exceeds_max_context_tokens(
        self, mock_file_state, mock_timeout, mock_growth_ratio, mock_max_tokens
    ):
        """Subagent exceeding max_context_tokens is marked as failed."""
        mock_max_tokens.return_value = 50000  # Set limit to 50k
        mock_growth_ratio.return_value = 2.5
        mock_timeout.return_value = 600
        mock_file_state.known_reads.return_value = []
        mock_file_state.writes_since.return_value = {}

        # Child with 60k total tokens (exceeds 50k limit)
        child = self._make_mock_child(input_tokens=30000, output_tokens=30000)
        
        result = _run_single_child(
            task_index=0,
            goal="Test task",
            child=child,
            parent_agent=None,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_reason"], "resource_limit_exceeded")
        self.assertIn("resource_warnings", result)
        self.assertTrue(
            any("Context limit exceeded" in w for w in result["resource_warnings"])
        )

    @patch("tools.delegate_tool._get_max_context_tokens")
    @patch("tools.delegate_tool._get_context_growth_ratio")
    @patch("tools.delegate_tool._get_child_timeout")
    @patch("tools.delegate_tool.file_state")
    def test_subagent_exceeds_context_growth_ratio(
        self, mock_file_state, mock_timeout, mock_growth_ratio, mock_max_tokens
    ):
        """Subagent exceeding context_growth_ratio is marked as failed."""
        mock_max_tokens.return_value = 100000
        mock_growth_ratio.return_value = 2.5  # Set ratio limit to 2.5
        mock_timeout.return_value = 600
        mock_file_state.known_reads.return_value = []
        mock_file_state.writes_since.return_value = {}

        # Child with growth ratio of 4.0 (output/input = 8000/2000 = 4.0, exceeds 2.5)
        child = self._make_mock_child(input_tokens=2000, output_tokens=8000)
        
        result = _run_single_child(
            task_index=0,
            goal="Test task",
            child=child,
            parent_agent=None,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_reason"], "resource_limit_exceeded")
        self.assertIn("resource_warnings", result)
        self.assertTrue(
            any("Context growth ratio exceeded" in w for w in result["resource_warnings"])
        )

    @patch("tools.delegate_tool._get_max_context_tokens")
    @patch("tools.delegate_tool._get_context_growth_ratio")
    @patch("tools.delegate_tool._get_child_timeout")
    @patch("tools.delegate_tool.file_state")
    def test_subagent_within_limits(
        self, mock_file_state, mock_timeout, mock_growth_ratio, mock_max_tokens
    ):
        """Subagent within limits completes successfully."""
        mock_max_tokens.return_value = 100000
        mock_growth_ratio.return_value = 2.5
        mock_timeout.return_value = 600
        mock_file_state.known_reads.return_value = []
        mock_file_state.writes_since.return_value = {}

        # Child within limits: 5k total tokens, growth ratio 2.0
        child = self._make_mock_child(input_tokens=2000, output_tokens=4000)
        
        result = _run_single_child(
            task_index=0,
            goal="Test task",
            child=child,
            parent_agent=None,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_reason"], "completed")
        self.assertNotIn("resource_warnings", result)

    @patch("tools.delegate_tool._get_max_context_tokens")
    @patch("tools.delegate_tool._get_context_growth_ratio")
    @patch("tools.delegate_tool._get_child_timeout")
    @patch("tools.delegate_tool.file_state")
    def test_subagent_exceeds_both_limits(
        self, mock_file_state, mock_timeout, mock_growth_ratio, mock_max_tokens
    ):
        """Subagent exceeding both limits shows both warnings."""
        mock_max_tokens.return_value = 50000
        mock_growth_ratio.return_value = 2.5
        mock_timeout.return_value = 600
        mock_file_state.known_reads.return_value = []
        mock_file_state.writes_since.return_value = {}

        # Child exceeds both: 60k total tokens AND growth ratio 4.0
        child = self._make_mock_child(input_tokens=15000, output_tokens=60000)
        
        result = _run_single_child(
            task_index=0,
            goal="Test task",
            child=child,
            parent_agent=None,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_reason"], "resource_limit_exceeded")
        self.assertIn("resource_warnings", result)
        self.assertEqual(len(result["resource_warnings"]), 2)
        self.assertTrue(
            any("Context limit exceeded" in w for w in result["resource_warnings"])
        )
        self.assertTrue(
            any("Context growth ratio exceeded" in w for w in result["resource_warnings"])
        )


if __name__ == "__main__":
    unittest.main()
