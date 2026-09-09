"""Tests for tools/profile_manager_tool.py (the ``profile_manage`` tool).

Covers: the opt-in availability gate, action dispatch and validation, profile
creation with identity metadata, configuration of an existing profile, and the
guardrail/toolset wiring.
"""

import json
from pathlib import Path

import pytest

from tools import profile_manager_tool
from tools.profile_manager_tool import (
    MAX_SOUL_CHARS,
    VALID_ACTIONS,
    _check_profile_manage_mode,
    profile_manage,
)


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated profile root, mirroring tests/hermes_cli/test_profiles.py.

    Skills seeding is stubbed out: it copies the bundled skills tree, which is
    slow and irrelevant to what these tests assert.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        "hermes_cli.profiles.seed_profile_skills", lambda *a, **kw: None
    )
    return tmp_path


def _call(**kwargs) -> dict:
    """Invoke the tool and parse its JSON envelope."""
    return json.loads(profile_manage(**kwargs))


# ===================================================================
# Availability gate
# ===================================================================

class TestAvailabilityGate:
    def test_hidden_by_default(self, monkeypatch):
        """No `profiles` toolset in config -> tool is not exposed."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda *a, **kw: {"toolsets": ["all"]}
        )
        assert _check_profile_manage_mode() is False

    def test_visible_when_opted_in(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"toolsets": ["coding", "profiles"]},
        )
        assert _check_profile_manage_mode() is True

    def test_hidden_from_delegated_children(self, monkeypatch):
        """A delegate_task child must not mint profiles even when opted in."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **kw: {"toolsets": ["profiles"]},
        )
        monkeypatch.setattr(
            profile_manager_tool, "_is_delegated_child_context", lambda: True
        )
        assert _check_profile_manage_mode() is False

    def test_unreadable_config_fails_closed(self, monkeypatch):
        def _boom(*a, **kw):
            raise RuntimeError("config unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config", _boom)
        assert _check_profile_manage_mode() is False


# ===================================================================
# Input validation
# ===================================================================

class TestValidation:
    def test_unknown_action_rejected(self):
        assert "Unknown action" in _call(action="destroy", name="x")["error"]

    def test_delete_is_not_an_action(self):
        """Deletion is deliberately out of scope for this tool."""
        assert "delete" not in VALID_ACTIONS
        assert "error" in _call(action="delete", name="whatever")

    @pytest.mark.parametrize("action", ["create", "configure"])
    def test_name_required(self, action):
        assert "requires a profile 'name'" in _call(action=action, name="")["error"]

    def test_oversized_soul_rejected(self, profile_env):
        result = _call(
            action="create", name="bigsoul", soul="x" * (MAX_SOUL_CHARS + 1)
        )
        assert "over the" in result["error"]
        # Rejected before any directory was created.
        assert not (profile_env / ".hermes" / "profiles" / "bigsoul").exists()

    def test_invalid_name_rejected(self, profile_env):
        assert "error" in _call(action="create", name="../escape")

    def test_reserved_name_rejected(self, profile_env):
        assert "error" in _call(action="create", name="hermes")

    def test_cannot_create_default(self, profile_env):
        assert "built-in" in _call(action="create", name="default")["error"]


# ===================================================================
# create
# ===================================================================

class TestCreate:
    def test_creates_profile_directory(self, profile_env):
        result = _call(action="create", name="researcher")
        assert result["created"] == "researcher"
        assert (profile_env / ".hermes" / "profiles" / "researcher").is_dir()
        assert result["chat_command"] == "hermes -p researcher chat"

    def test_name_is_normalized(self, profile_env):
        """Title-cased model output resolves to the on-disk id."""
        result = _call(action="create", name="Researcher")
        assert result["created"] == "researcher"

    def test_writes_identity_metadata(self, profile_env):
        import yaml

        _call(
            action="create",
            name="scribe",
            display_name="Release Scribe",
            description="Writes release notes from merged PRs.",
        )
        meta = yaml.safe_load(
            (profile_env / ".hermes" / "profiles" / "scribe" / "profile.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert meta["display_name"] == "Release Scribe"
        assert meta["description"] == "Writes release notes from merged PRs."
        assert meta["description_auto"] is False

    def test_writes_soul(self, profile_env):
        _call(action="create", name="critic", soul="# Critic\n\nBe blunt.\n")
        soul = (
            profile_env / ".hermes" / "profiles" / "critic" / "SOUL.md"
        ).read_text(encoding="utf-8")
        assert "Be blunt." in soul

    def test_duplicate_name_rejected(self, profile_env):
        _call(action="create", name="dupe")
        assert "already exists" in _call(action="create", name="dupe")["error"]

    def test_missing_clone_source_rejected(self, profile_env):
        result = _call(action="create", name="child", clone_from="nonexistent")
        assert "does not exist" in result["error"]
        assert not (profile_env / ".hermes" / "profiles" / "child").exists()

    def test_no_skills_conflicts_with_clone_from(self, profile_env):
        _call(action="create", name="parent")
        result = _call(
            action="create", name="kid", clone_from="parent", no_skills=True
        )
        assert "mutually exclusive" in result["error"]
        # Nothing was created before the conflict was caught.
        assert not (profile_env / ".hermes" / "profiles" / "kid").exists()


# ===================================================================
# configure
# ===================================================================

class TestConfigure:
    def test_updates_existing_profile(self, profile_env):
        import yaml

        _call(action="create", name="analyst")
        result = _call(
            action="configure",
            name="analyst",
            display_name="Data Analyst",
            description="Slices metrics.",
        )
        assert set(result["applied"]) == {"display_name", "description"}
        meta = yaml.safe_load(
            (profile_env / ".hermes" / "profiles" / "analyst" / "profile.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert meta["display_name"] == "Data Analyst"

    def test_rewrites_soul(self, profile_env):
        _call(action="create", name="poet", soul="old")
        _call(action="configure", name="poet", soul="new")
        soul = (
            profile_env / ".hermes" / "profiles" / "poet" / "SOUL.md"
        ).read_text(encoding="utf-8")
        assert soul == "new"

    def test_missing_profile_rejected(self, profile_env):
        assert "does not exist" in _call(action="configure", name="ghost")["error"]

    def test_requires_at_least_one_field(self, profile_env):
        _call(action="create", name="bare")
        result = _call(action="configure", name="bare")
        assert "at least one of" in result["error"]


# ===================================================================
# list
# ===================================================================

class TestList:
    def test_lists_created_profiles(self, profile_env):
        _call(action="create", name="alpha", description="First.")
        _call(action="create", name="beta")
        result = _call(action="list")
        names = {p["name"] for p in result["profiles"]}
        assert {"alpha", "beta"} <= names
        assert result["count"] == len(result["profiles"])
        alpha = next(p for p in result["profiles"] if p["name"] == "alpha")
        assert alpha["description"] == "First."

    def test_list_needs_no_name(self, profile_env):
        assert "error" not in _call(action="list")


# ===================================================================
# Wiring
# ===================================================================

class TestWiring:
    def test_classified_as_mutating(self):
        from agent.tool_guardrails import MUTATING_TOOL_NAMES

        assert "profile_manage" in MUTATING_TOOL_NAMES

    def test_profiles_toolset_exposes_the_tool(self):
        from toolsets import resolve_toolset

        assert "profile_manage" in resolve_toolset("profiles")

    def test_schema_matches_handler_actions(self):
        enum = profile_manager_tool.PROFILE_MANAGE_SCHEMA["function"]["parameters"][
            "properties"
        ]["action"]["enum"]
        assert set(enum) == set(VALID_ACTIONS)
