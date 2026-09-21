#!/usr/bin/env python3
"""W1-D regression: plugin skill serving must obey _skill_access_decision.

``_serve_plugin_skill`` bypassed the allow/shadow/block decision that local
skills go through. A qualified plugin skill (``namespace:bare``) must obey the
same access enforcement as local skills.

Tests cover:
- enforce mode: non-allowlisted plugin skill → access-denied result
- enforce mode: explicitly allowed plugin skill → served normally
- shadow mode: non-allowlisted plugin skill → shadow_block (served, logged)
- disabled plugin → still returns disabled error (preserved behavior)
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import tools.skills_tool as st


_SKILL_MD_CONTENT = """---
description: A plugin skill
---
# Plugin Skill Body

Hello from the plugin.
"""


@pytest.fixture
def skill_md(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text(_SKILL_MD_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def patch_access(monkeypatch):
    """Patch _skill_access_decision to a controlled return value."""
    def _set(decision: str):
        monkeypatch.setattr(st, "_skill_access_decision", lambda name: decision)
        monkeypatch.setattr(st, "_current_profile", lambda: "octacon")
    return _set


@pytest.fixture(autouse=True)
def patch_plugin_infra(monkeypatch):
    """Patch plugin discovery / disabled-list / bundle-context so the
    access gate is the only variable."""
    monkeypatch.setattr(
        "hermes_cli.plugins._get_disabled_plugins", lambda: set()
    )
    # get_plugin_manager().list_plugin_skills(ns) → [] (no siblings)
    fake_pm = type("FakePM", (), {})()
    fake_pm.list_plugin_skills = lambda ns: []
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager", lambda: fake_pm
    )
    # platform check → True (supported)
    monkeypatch.setattr(st, "skill_matches_platform", lambda fm: True)


class TestPluginSkillAccessEnforce:
    """Under enforce mode, a non-allowlisted plugin skill is blocked."""

    def test_non_allowlisted_plugin_skill_returns_denied(self, skill_md, patch_access):
        patch_access("block")
        result = st._serve_plugin_skill(skill_md, "myplugin", "secret-skill")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "not enabled" in parsed["error"].lower()
        assert parsed["readiness_status"] == "not_enabled"

    def test_allowed_plugin_skill_is_served(self, skill_md, patch_access):
        patch_access("allow")
        result = st._serve_plugin_skill(skill_md, "myplugin", "allowed-skill")
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert "Hello from the plugin" in parsed["content"]


class TestPluginSkillAccessShadow:
    """Shadow mode: non-allowlisted plugin skill is served but logged."""

    def test_shadow_plugin_skill_is_served_with_warning(self, skill_md, patch_access):
        patch_access("shadow_block")
        result = st._serve_plugin_skill(skill_md, "myplugin", "shadowed-skill")
        parsed = json.loads(result)
        # shadow → load proceeds (success=True), but would-block is logged
        assert parsed["success"] is True
        assert "Hello from the plugin" in parsed["content"]


class TestPluginSkillDisabledPreserved:
    """Disabled-plugin check still fires before access gate."""

    def test_disabled_plugin_returns_disabled_error(self, skill_md, patch_access, monkeypatch):
        # Even if access decision would allow, disabled plugin wins
        patch_access("allow")
        monkeypatch.setattr(
            "hermes_cli.plugins._get_disabled_plugins", lambda: {"myplugin"}
        )
        result = st._serve_plugin_skill(skill_md, "myplugin", "any-skill")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "disabled" in parsed["error"].lower()
