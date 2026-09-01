#!/usr/bin/env python3
"""W1-B regression: Tier-3 gate must enforce for pre-resolved profile_content.

The Tier-3 gate in _build_child_agent was nested inside the
``if loaded_profile_cfg is None and profile:`` block — so when
``profile_content`` is supplied directly (the batch pre-resolve path via
delegate_task), the gate was skipped entirely, allowing a Tier-3 profile
to be delegated to.

These tests exercise the actual bypass: a Tier-3 profile supplied via
``profile_content`` must raise RuntimeError, while a permitted Tier-2 /
preloaded profile path remains usable.
"""

import threading
from unittest.mock import MagicMock

import pytest

from tools.delegate_tool import _build_child_agent


def _make_mock_parent(depth: int = 0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "«redacted»"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.session_id = "test-parent-session"
    parent._subagent_id = None
    parent._current_turn_id = "turn-1"
    parent._memory_manager = None
    parent._delegate_saved_tool_names = []
    parent._fallback_chain = None
    parent.reasoning_config = {"effort": "off"}
    parent.max_tokens = 4096
    parent.enabled_toolsets = ["terminal", "file", "web", "kanban", "hermes-cli"]
    parent.valid_tool_names = ["read_file", "write_file", "web_search", "terminal"]
    return parent


_TIER3_CONTENT = {
    "name": "dormant-worker",
    "config": {
        "tier": 3,
        "model": {"default": "glm-5.1", "provider": "ollama-cloud"},
        "toolsets": ["hermes-cli", "kanban"],
        "skills": {"always_skills": []},
    },
    "soul_md": "# dormant-worker",
    "skills": [],
}

_TIER2_CONTENT = {
    "name": "active-worker",
    "config": {
        "tier": 2,
        "model": {"default": "deepseek-v4-flash", "provider": "ollama-cloud"},
        "toolsets": ["hermes-cli", "kanban"],
        "skills": {"always_skills": []},
    },
    "soul_md": "# active-worker",
    "skills": [],
}


class TestTier3ProfileContentBlocked:
    """W1-B: a Tier-3 profile supplied via profile_content cannot create a child."""

    def test_tier3_via_profile_content_raises(self):
        """The bypass: profile_content with tier=3 must raise RuntimeError."""
        parent = _make_mock_parent()
        with pytest.raises(RuntimeError, match="tier 3"):
            _build_child_agent(
                task_index=0,
                goal="Test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=30,
                task_count=1,
                parent_agent=parent,
                profile="dormant-worker",
                profile_content=_TIER3_CONTENT,
            )

    def test_tier3_profile_content_non_integer_tier_passes(self):
        """Non-integer tier must continue to be allowed (existing behaviour)."""
        parent = _make_mock_parent()
        content = dict(_TIER3_CONTENT)
        content = {
            **_TIER3_CONTENT,
            "config": {**_TIER3_CONTENT["config"], "tier": "undefined"},
        }
        # Should NOT raise — non-integer tier is let through
        try:
            _build_child_agent(
                task_index=0,
                goal="Test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=30,
                task_count=1,
                parent_agent=parent,
                profile="dormant-worker",
                profile_content=content,
            )
        except RuntimeError:
            pytest.fail("Non-integer tier should not be blocked")
        except Exception:
            # Other construction errors (mock parent incompleteness) are fine;
            # we only care that the Tier-3 gate did not fire.
            pass


class TestTier2ProfileContentAllowed:
    """A permitted Tier-2 / preloaded profile path remains usable."""

    def test_tier2_via_profile_content_builds(self):
        parent = _make_mock_parent()
        child = _build_child_agent(
            task_index=0,
            goal="Test",
            context=None,
            toolsets=None,
            model=None,
            max_iterations=30,
            task_count=1,
            parent_agent=parent,
            profile="active-worker",
            profile_content=_TIER2_CONTENT,
        )
        assert child is not None
        assert getattr(child, "model", "") == "deepseek-v4-flash"
        assert child.__dict__.get("_delegate_profile") == "active-worker"
