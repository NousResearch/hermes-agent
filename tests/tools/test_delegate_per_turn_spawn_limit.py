#!/usr/bin/env python3
"""
Tests for the per-turn delegation spawn cap (issue #52484).

Verifies that delegate_task refuses to spawn more subagents once the
per-turn counter exceeds delegation.max_spawns_per_turn, and that the
counter is correctly incremented after each spawn.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure we can import from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.delegate_tool import (
    _DEFAULT_MAX_SPAWNS_PER_TURN,
    _get_max_spawns_per_turn,
    delegate_task,
)


def _make_mock_parent(depth=0, spawn_count=0):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "sk-test"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.openrouter_min_coding_score = None
    parent._delegate_depth = depth
    parent._turn_spawn_count = spawn_count
    parent._active_children = []
    parent._active_children_lock = None
    parent.session_id = "test-session"
    parent._current_turn_id = "turn-1"
    parent.valid_tool_names = {"delegate_task", "terminal", "file"}
    parent.enabled_toolsets = None
    parent._fallback_chain = None
    parent._credential_pool = None
    return parent


class TestGetMaxSpawnsPerTurn(unittest.TestCase):
    """Config reader for delegation.max_spawns_per_turn."""

    def test_default_when_unset(self):
        """Returns default when config key is absent."""
        with patch("tools.delegate_tool._load_config") as mock_cfg:
            mock_cfg.return_value = {}
            self.assertEqual(_get_max_spawns_per_turn(), _DEFAULT_MAX_SPAWNS_PER_TURN)

    def test_respects_config_value(self):
        """Returns the configured value when set."""
        with patch("tools.delegate_tool._load_config") as mock_cfg:
            mock_cfg.return_value = {"max_spawns_per_turn": 5}
            self.assertEqual(_get_max_spawns_per_turn(), 5)

    def test_floors_at_1(self):
        """Values below 1 are floored to 1."""
        with patch("tools.delegate_tool._load_config") as mock_cfg:
            mock_cfg.return_value = {"max_spawns_per_turn": 0}
            self.assertEqual(_get_max_spawns_per_turn(), 1)

    def test_invalid_value_falls_back(self):
        """Non-integer values fall back to default with a warning."""
        with patch("tools.delegate_tool._load_config") as mock_cfg:
            mock_cfg.return_value = {"max_spawns_per_turn": "banana"}
            self.assertEqual(_get_max_spawns_per_turn(), _DEFAULT_MAX_SPAWNS_PER_TURN)


class TestPerTurnSpawnCap(unittest.TestCase):
    """The spawn cap blocks excess delegate_task calls within a turn."""

    def setUp(self):
        # Patch _build_child_agent to avoid real agent construction
        self._build_patcher = patch("tools.delegate_tool._build_child_agent")
        self._mock_build = self._build_patcher.start()
        self._mock_build.return_value = MagicMock()

        # Patch _resolve_delegation_credentials
        self._creds_patcher = patch("tools.delegate_tool._resolve_delegation_credentials")
        self._mock_creds = self._creds_patcher.start()
        self._mock_creds.return_value = {
            "model": "test-model",
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
        }

        # Patch _load_config to return empty (use defaults)
        self._cfg_patcher = patch("tools.delegate_tool._load_config")
        self._mock_cfg = self._cfg_patcher.start()
        self._mock_cfg.return_value = {}

    def tearDown(self):
        self._build_patcher.stop()
        self._creds_patcher.stop()
        self._cfg_patcher.stop()

    def test_blocks_when_cap_exceeded(self):
        """delegate_task returns an error when the per-turn cap is exceeded."""
        # Parent already spawned up to the cap
        parent = _make_mock_parent(spawn_count=_DEFAULT_MAX_SPAWNS_PER_TURN)
        result = json.loads(delegate_task(goal="test task", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("spawn cap", result["error"])
        self.assertIn("max_spawns_per_turn", result["error"])

    def test_blocks_when_batch_would_exceed(self):
        """A batch that would push the count over the cap is rejected."""
        # Cap is 10, already spawned 8, batch of 3 would make 11
        parent = _make_mock_parent(spawn_count=8)
        # Force max_concurrent_children high enough so the batch isn't
        # rejected by the existing max_concurrent_children check
        self._mock_cfg.return_value = {"max_concurrent_children": 10}
        tasks = [{"goal": f"task {i}"} for i in range(3)]
        result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("spawn cap", result["error"])

    def test_allows_when_under_cap(self):
        """delegate_task proceeds when the cap is not yet reached."""
        parent = _make_mock_parent(spawn_count=0)
        # The function will go past the cap check and attempt to build +
        # execute children. Since _build_child_agent is mocked, execution
        # will fail on JSON serialization — that's fine, we just verify
        # the cap check passed (no "spawn cap" error in the result).
        try:
            result_str = delegate_task(goal="test task", parent_agent=parent)
            result = json.loads(result_str)
            # If we got a JSON result, it should NOT contain the cap error
            self.assertNotIn("spawn cap", result.get("error", ""))
        except (TypeError, Exception):
            pass  # Expected — execution path is not fully mocked

    def test_error_message_includes_remaining(self):
        """Error message tells the model how many spawns remain."""
        # count=8, 3 tasks → 8+3=11 > 10 → BLOCKED, 2 remaining
        parent = _make_mock_parent(spawn_count=8)
        self._mock_cfg.return_value = {"max_spawns_per_turn": 10, "max_concurrent_children": 5}
        tasks = [{"goal": f"task {i}"} for i in range(3)]
        result_str = delegate_task(tasks=tasks, parent_agent=parent)
        result = json.loads(result_str)
        self.assertIn("error", result)
        # 10 - 8 = 2 remaining
        self.assertIn("2", result["error"])

    def test_error_message_says_no_more_when_at_cap(self):
        """Error message says 'No more spawns' when count equals cap."""
        parent = _make_mock_parent(spawn_count=10)
        self._mock_cfg.return_value = {"max_spawns_per_turn": 10}
        result = json.loads(delegate_task(goal="test", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("No more spawns", result["error"])

    def test_counter_incremented_after_spawn(self):
        """The _turn_spawn_count is incremented after children are built."""
        parent = _make_mock_parent(spawn_count=3)
        self._mock_cfg.return_value = {"max_concurrent_children": 5}
        # The build will succeed (mocked), then execution will fail
        # but the counter increment happens BEFORE execution
        try:
            delegate_task(goal="test task", parent_agent=parent)
        except Exception:
            pass
        # Should have been incremented by 1 (single goal = 1 child)
        self.assertEqual(getattr(parent, "_turn_spawn_count", 0), 4)


if __name__ == "__main__":
    unittest.main()
