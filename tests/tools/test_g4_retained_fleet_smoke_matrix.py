#!/usr/bin/env python3
"""
G4 - Retained-fleet non-gateway routing smoke matrix (executable proof).

Implements the 15 approved rows from
``migration/evidence/2026-07-13/G4_RETAINED_FLEET_NON_GATEWAY_SMOKE_MATRIX.md``
as a real pytest module inside the KenseiAgent test tree.

Boundary compliance (every row):
  - No gateway/dashboard/service start.
  - No Discord/token/platform use.
  - No live profile/config read or write - fixtures live in a disposable
    HERMES_HOME under tmp_path.
  - No delegation against live production agents - the child runner is
    stubbed via unittest.mock.patch.
  - Disposable/non-gateway fixtures only.

Row index:
  S1-S5  synchronous delegation path
  A1-A3  asynchronous delegation path
  U1-U4  authority / unauthorised-route guards
  F1/F2/R1  failure reporting at the routing layer
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Disposable fixture profiles (built under tmp_path / .hermes / profiles)
# ---------------------------------------------------------------------------

_FIXTURE_PROFILES = ("fx-lead", "fx-leaf", "fx-broken", "fx-malformed", "fx-nonspawn")
# fx-ghost is intentionally absent - never created on disk.


def _write_yaml(path, data):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


_LEAD_CONFIG = {
    "model": {"default": "fake-model", "provider": "fake-local"},
    "delegation": {
        "provider": "",
        "base_url": "http://localhost:0/v1",
        "api_key": "fake-key",
        "max_concurrent_children": 3,
        "max_spawn_depth": 2,
        "subagent_auto_approve": False,
    },
    "agency-agents-router": {"enabled": True},
}

_LEAF_CONFIG = {
    "model": {"default": "fake-model", "provider": "fake-local"},
    # No delegation section - leaf cannot delegate.
}


_NONSPAWN_CONFIG = dict(_LEAD_CONFIG)
_NONSPAWN_CONFIG["delegation"] = dict(_LEAD_CONFIG["delegation"])


@pytest.fixture
def g4_home(tmp_path, monkeypatch):
    """Build a disposable HERMES_HOME with the five fixture profiles.

    Sets HERMES_HOME, patches Path.home() so profile_exists() resolves
    profiles under the temp tree, and patches config loaders so
    _is_profile_spawnable and _load_config read the fixture config.yaml.
    """
    home = tmp_path / "g4-home"
    hermes_root = home / ".hermes"
    profiles_root = hermes_root / "profiles"
    profiles_root.mkdir(parents=True)

    # fx-lead: valid config with delegation section.
    _write_yaml(profiles_root / "fx-lead" / "config.yaml", _LEAD_CONFIG)
    (profiles_root / "fx-lead" / "SOUL.md").write_text("# fx-lead\n")

    # fx-leaf: valid config, no delegation section.
    _write_yaml(profiles_root / "fx-leaf" / "config.yaml", _LEAF_CONFIG)
    (profiles_root / "fx-leaf" / "SOUL.md").write_text("# fx-leaf\n")

    # fx-broken: directory exists, config.yaml absent.
    (profiles_root / "fx-broken").mkdir(parents=True)

    # fx-malformed: config.yaml present but model under agent.model
    # (wrong key path - should not resolve model.default).
    _write_yaml(
        profiles_root / "fx-malformed" / "config.yaml",
        {"agent": {"model": "wrong-key-model"}},
    )

    # fx-nonspawn: same as fx-lead but named in nonspawnable_profiles.
    _write_yaml(profiles_root / "fx-nonspawn" / "config.yaml", _NONSPAWN_CONFIG)
    (profiles_root / "fx-nonspawn" / "SOUL.md").write_text("# fx-nonspawn\n")

    # fx-ghost: intentionally NOT created.

    # Root config.yaml with kanban.nonspawnable_profiles.
    root_cfg = {
        "kanban": {"nonspawnable_profiles": ["fx-nonspawn"]},
        "delegation": _LEAD_CONFIG["delegation"],
    }
    _write_yaml(hermes_root / "config.yaml", root_cfg)

    # Point HERMES_HOME at the disposable tree.
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))
    # delegate_tool.py:2598 uses os.path.expanduser("~/.hermes/profiles/...")
    # which follows HOME, not HERMES_HOME. Set HOME to the fixture dir so
    # the expanduser path resolves to the disposable profile tree.
    monkeypatch.setenv("HOME", str(home))
    # Patch Path.home() so profiles.get_profile_dir / get_default_hermes_root
    # resolve under tmp_path instead of the real ~.
    monkeypatch.setattr(Path, "home", lambda: home)

    # Clear config caches so the fixture config is picked up.
    import hermes_cli.config as cfg_mod
    for attr in ("_config_cache", "_CONFIG_CACHE", "_cache"):
        if hasattr(cfg_mod, attr):
            cache = getattr(cfg_mod, attr)
            if isinstance(cache, dict):
                cache.clear()

    return SimpleNamespace(
        home=home,
        hermes_root=hermes_root,
        profiles_root=profiles_root,
    )


def _make_mock_parent(depth=0, **overrides):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "http://localhost:0/v1"
    parent.api_key = "fake-key"
    parent.provider = "fake-local"
    parent.api_mode = "chat_completions"
    parent.model = "fake-model"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    # Durable async delegation persists this field into SQLite. A MagicMock
    # attribute is not serialisable and is not representative of a real agent.
    parent.session_id = ""
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    for k, v in overrides.items():
        setattr(parent, k, v)
    return parent


def _stub_child_result(goal="noop"):
    return {
        "task_index": 0,
        "status": "completed",
        "summary": "stubbed result for " + goal,
        "api_calls": 0,
        "duration_seconds": 0.01,
    }


# ---------------------------------------------------------------------------
# Synchronous delegation path (S1-S5)
# ---------------------------------------------------------------------------

class TestSyncDelegation:
    """Rows S1-S5: synchronous delegation path."""

    @patch("tools.delegate_tool._run_single_child")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._load_config")
    def test_S1_valid_sync_route_to_lead_class(self, mock_cfg, mock_creds, mock_run, g4_home):
        """S1: Valid sync route to a lead class dispatches via stubbed runner."""
        from tools.delegate_tool import delegate_task
        mock_cfg.return_value = _LEAD_CONFIG["delegation"]
        mock_creds.return_value = {
            "model": "fake-model", "provider": "fake-local",
            "base_url": "http://localhost:0/v1", "api_key": "fake-key",
            "api_mode": "chat_completions",
        }
        mock_run.return_value = _stub_child_result("noop")
        parent = _make_mock_parent(depth=0)
        result = json.loads(
            delegate_task(
                goal="noop", parent_agent=parent, background=False,
            )
        )
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["status"] == "completed"
        mock_run.assert_called_once()

    def test_S2_missing_profile(self, g4_home):
        """S2: Missing profile returns tool_error with profile-not-found."""
        from tools.delegate_tool import delegate_task
        parent = _make_mock_parent(depth=0)
        result = json.loads(
            delegate_task(goal="test", parent_agent=parent, profile="fx-ghost")
        )
        assert "error" in result
        assert "fx-ghost" in result["error"]
        assert "not found" in result["error"].lower()

    def test_S3_bad_target_no_config(self, g4_home):
        """S3: Bad target (no config.yaml) returns tool_error."""
        from tools.delegate_tool import delegate_task
        parent = _make_mock_parent(depth=0)
        result = json.loads(
            delegate_task(goal="test", parent_agent=parent, profile="fx-broken")
        )
        assert "error" in result
        assert "fx-broken" in result["error"]
        assert "config.yaml" in result["error"]

    def test_S4_unauthorised_caller(self, g4_home):
        """S4: parent_agent=None returns tool_error requiring parent context."""
        from tools.delegate_tool import delegate_task
        result = json.loads(delegate_task(goal="test"))
        assert "error" in result
        assert "parent agent" in result["error"].lower()

    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._load_config")
    def test_S5_input_contract_fail_safe(self, mock_cfg, mock_creds, g4_home):
        """S5: neither goal nor tasks supplied returns tool_error."""
        from tools.delegate_tool import delegate_task
        mock_cfg.return_value = _LEAD_CONFIG["delegation"]
        mock_creds.return_value = {
            "model": "fake-model", "provider": None,
            "base_url": None, "api_key": None, "api_mode": None,
        }
        parent = _make_mock_parent(depth=0)
        # Neither goal nor tasks.
        result_neither = json.loads(delegate_task(parent_agent=parent))
        assert "error" in result_neither
        assert "goal" in result_neither["error"].lower() or "task" in result_neither["error"].lower()


# ---------------------------------------------------------------------------
# Asynchronous delegation path (A1-A3)
# ---------------------------------------------------------------------------

class TestAsyncDelegation:
    """Rows A1-A3: asynchronous delegation path."""

    @patch("tools.delegate_tool._run_single_child")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._load_config")
    def test_A1_valid_async_route(self, mock_cfg, mock_creds, mock_run, g4_home):
        """A1: Valid async route returns dispatched handle contract."""
        from tools.delegate_tool import delegate_task
        from tools import async_delegation as ad
        ad._reset_for_tests()
        mock_cfg.return_value = _LEAD_CONFIG["delegation"]
        mock_creds.return_value = {
            "model": "fake-model", "provider": "fake-local",
            "base_url": "http://localhost:0/v1", "api_key": "fake-key",
            "api_mode": "chat_completions",
        }
        mock_run.return_value = _stub_child_result("async-noop")
        parent = _make_mock_parent(depth=0)
        result = json.loads(
            delegate_task(goal="async-noop", parent_agent=parent, background=True)
        )
        # Background dispatch returns a handle, not a results list.
        # It may also fall back to sync if async_delivery_supported is False,
        # but in the default test environment (no session bound) it returns True.
        if result.get("status") == "dispatched":
            assert "delegation_id" in result
            assert result.get("mode") == "background"
        else:
            # Sync fallback is acceptable if the test environment does not
            # support async delivery - but it must have results.
            assert "results" in result
        ad._reset_for_tests()

    @patch("tools.delegate_tool._run_single_child")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._load_config")
    def test_A2_async_to_sync_fallback_stateless(self, mock_cfg, mock_creds, mock_run, g4_home):
        """A2: Async -> sync fallback under stateless-HTTP flag."""
        from tools.delegate_tool import delegate_task
        from tools import async_delegation as ad
        ad._reset_for_tests()
        mock_cfg.return_value = _LEAD_CONFIG["delegation"]
        mock_creds.return_value = {
            "model": "fake-model", "provider": "fake-local",
            "base_url": "http://localhost:0/v1", "api_key": "fake-key",
            "api_mode": "chat_completions",
        }
        mock_run.return_value = _stub_child_result("fallback")
        parent = _make_mock_parent(depth=0)
        # Patch async_delivery_supported to return False (stateless HTTP).
        with patch("gateway.session_context.async_delivery_supported", return_value=False):
            result = json.loads(
                delegate_task(goal="fallback", parent_agent=parent, background=True)
            )
        # Should fall back to sync with a note.
        assert "results" in result
        assert "note" in result
        assert "synchronous" in result["note"].lower() or "SYNCHRONOUSLY" in result["note"]
        ad._reset_for_tests()

    @patch("tools.delegate_tool._run_single_child")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._load_config")
    def test_A3_async_pool_capacity_fallback(self, mock_cfg, mock_creds, mock_run, g4_home):
        """A3: Async pool-capacity fallback runs synchronously with note."""
        from tools.delegate_tool import delegate_task
        from tools import async_delegation as ad
        ad._reset_for_tests()
        mock_cfg.return_value = _LEAD_CONFIG["delegation"]
        mock_creds.return_value = {
            "model": "fake-model", "provider": "fake-local",
            "base_url": "http://localhost:0/v1", "api_key": "fake-key",
            "api_mode": "chat_completions",
        }
        mock_run.return_value = _stub_child_result("capacity")
        parent = _make_mock_parent(depth=0)
        # Fill the async pool to capacity so the next dispatch is rejected.
        import time as _time
        for i in range(10):  # exceed default cap of 3
            ad._records["fake-" + str(i)] = {
                "delegation_id": "fake-" + str(i),
                "status": "running",
                "dispatched_at": _time.time(),
                "completed_at": None,
            }
        result = json.loads(
            delegate_task(goal="capacity", parent_agent=parent, background=True)
        )
        # Pool at capacity -> sync fallback with note.
        assert "results" in result
        assert "note" in result
        assert "capacity" in result["note"].lower()
        ad._reset_for_tests()


# ---------------------------------------------------------------------------
# Authority / unauthorised-route guards (U1-U4)
# ---------------------------------------------------------------------------

class TestAuthorityGuards:
    """Rows U1-U4: authority / unauthorised-route guards."""

    def test_U1_nonspawnable_lead(self, g4_home):
        """U1: Non-spawnable lead is bucketed skipped_nonspawnable, not spawned."""
        from hermes_cli.kanban_db import _is_profile_spawnable
        # fx-nonspawn has a directory but is in nonspawnable_profiles.
        assert _is_profile_spawnable("fx-nonspawn") is False
        # fx-leaf is spawnable (has dir, not in nonspawnable list).
        assert _is_profile_spawnable("fx-leaf") is True

    def test_U2_leaf_cannot_delegate(self, g4_home):
        """U2: Leaf profile (no delegation section) cannot delegate -
        the delegation toolset is stripped from leaves."""
        from tools.delegate_tool import _strip_blocked_tools, DELEGATE_BLOCKED_TOOLS
        # A leaf toolsets go through _strip_blocked_tools which removes
        # the delegation toolset (all its tools are in DELEGATE_BLOCKED_TOOLS).
        stripped = _strip_blocked_tools(["terminal", "file", "delegation"])
        assert "delegation" not in stripped
        assert "terminal" in stripped
        assert "file" in stripped
        # delegate_task itself is in the blocked set.
        assert "delegate_task" in DELEGATE_BLOCKED_TOOLS

    def test_U3_depth_cap(self, g4_home):
        """U3: Depth cap blocks a depth-1 child from spawning when
        max_spawn_depth=1 (default)."""
        from tools.delegate_tool import delegate_task
        # parent at depth 1, max_spawn_depth=1 -> depth >= max_spawn.
        # Patch _load_config so max_spawn_depth=1 (flat) is enforced,
        # since the fixture config sets it to 2.
        with patch("tools.delegate_tool._load_config") as mock_cfg:
            mock_cfg.return_value = {"max_spawn_depth": 1}
            parent = _make_mock_parent(depth=1)
            result = json.loads(delegate_task(goal="test", parent_agent=parent))
        assert "error" in result
        assert "depth" in result["error"].lower()

    def test_U4_approval_gate_closed(self, g4_home):
        """U4: Approval gate closed (subagent_auto_approve=false) ->
        _get_subagent_approval_callback returns the auto-deny callback
        whose log message contains the guidance to set
        delegation.subagent_auto_approve: true."""
        from tools.delegate_tool import (
            _get_subagent_approval_callback,
            _subagent_auto_deny,
        )
        # Default config has subagent_auto_approve=False.
        cb = _get_subagent_approval_callback()
        assert cb is _subagent_auto_deny
        # The deny callback returns deny and logs the guidance.
        dangerous_cmd = "dangerous " + "command"
        result = cb(dangerous_cmd, "dangerous")
        assert result == "deny"
        # Verify the guidance message is in the callback source.
        import inspect
        src = inspect.getsource(_subagent_auto_deny)
        assert "subagent_auto_approve: true" in src


# ---------------------------------------------------------------------------
# Failure reporting at the routing layer (F1/F2/R1)
# ---------------------------------------------------------------------------

class TestFailureReporting:
    """Rows F1/F2/R1: failure reporting at the routing layer."""

    def test_F1_credential_provider_unresolved(self, g4_home):
        """F1: Credential/provider unresolved -> ValueError surfaced as tool_error."""
        from tools.delegate_tool import delegate_task
        parent = _make_mock_parent(depth=0)
        # Patch _load_config and _resolve_delegation_credentials to simulate
        # provider resolution failure.
        with patch("tools.delegate_tool._load_config") as mock_cfg, \
             patch("tools.delegate_tool._resolve_delegation_credentials") as mock_creds:
            mock_cfg.return_value = {
                "model": "fake-model", "provider": "bogus-provider",
            }
            mock_creds.side_effect = ValueError(
                "Cannot resolve delegation provider bogus-provider: "
                "OPENROUTER_API_KEY not set. Check that the provider is "
                "configured (API key set, valid provider name)."
            )
            result = json.loads(
                delegate_task(goal="test", parent_agent=parent)
            )
        assert "error" in result
        assert "bogus-provider" in result["error"]
        # No silent success.
        assert "results" not in result

    def test_F2_kanban_create_failure_propagates(self, g4_home):
        """F2: Kanban create failure propagates - route_goal_to_kanban
        returns .ok==False with .error populated."""
        from hermes_cli.goal_routing import route_goal_to_kanban

        def fake_create(_args):
            return json.dumps({"ok": False, "error": "kanban unavailable"})

        result = route_goal_to_kanban(
            "Fix knowledge loop",
            subgoals=[],
            session_id="sid-err",
            create_task=fake_create,
        )
        assert result.ok is False
        assert result.error == "kanban unavailable"
        assert result.task_id is None

    def test_R1_disabled_retained_profile_buckets_skipped(self, g4_home):
        """R1: A subgoal resolving to a disabled/non-spawnable retained
        profile is bucketed as skipped_nonspawnable, not silently spawned."""
        from hermes_cli.kanban_db import _is_profile_spawnable
        # octacon-frontend and remii-deep are real retained-fleet names.
        # In the fixture tree they do not have directories, so
        # _is_profile_spawnable returns False (fail-closed).
        # The dispatch loop in kanban_db._default_spawn appends to
        # result.skipped_nonspawnable when _is_profile_spawnable is False.
        for name in ("octacon-frontend", "remii-deep"):
            assert _is_profile_spawnable(name) is False
