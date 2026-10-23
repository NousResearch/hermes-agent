"""Focused tests for profile-owned skill creation routing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.skill_owner_routing import (
    active_profile_from_home,
    create_skill_with_owner_routing,
    declared_skill_owner,
    owner_routing_policy,
    reset_owner_routing_scope,
)


ROUTED_SKILL = """\
---
name: routed-skill
description: A routed test skill.
metadata:
  hermes:
    owner_profile: trt
---

# Routed Skill

Verify ownership routing.
"""

PLAIN_SKILL = """\
---
name: routed-skill
description: A plain test skill.
---

# Routed Skill
"""


def _enabled_policy(**overrides):
    value = {
        "enabled": True,
        "require_owner_metadata": True,
        "route_from_default": True,
    }
    value.update(overrides)
    return value


def _creator(observed, *, success=True):
    def create(name, content, category=None):
        observed["home"] = get_hermes_home().resolve(strict=False)
        observed["args"] = (name, content, category)
        return {"success": success, "message": "created" if success else "failed"}

    return create


def test_policy_uses_canonical_default_root_for_custom_profile_home(tmp_path, monkeypatch):
    root = tmp_path / "fleet-data"
    active = root / "profiles" / "growth"
    active.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active))
    observed = {}

    def fake_load_config():
        observed["home"] = get_hermes_home().resolve(strict=False)
        return {
            "skills": {
                "owner_routing": {
                    "enabled": True,
                    "require_owner_metadata": False,
                    "route_from_default": False,
                }
            }
        }

    with patch("hermes_cli.config.load_config", side_effect=fake_load_config):
        policy = owner_routing_policy()

    assert observed["home"] == root.resolve(strict=False)
    assert policy == {
        "enabled": True,
        "require_owner_metadata": False,
        "route_from_default": False,
    }


def test_policy_failure_disables_cross_profile_routing(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/definitely/not/used")
    with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
        assert owner_routing_policy()["enabled"] is False


def test_declared_owner_reads_nested_hermes_metadata():
    assert declared_skill_owner(ROUTED_SKILL) == "trt"
    assert declared_skill_owner(PLAIN_SKILL) is None


def test_active_profile_comes_from_live_context_home_not_sticky_file(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    profile = root / "profiles" / "growth"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    token = set_hermes_home_override(profile)
    try:
        assert active_profile_from_home() == "growth"
    finally:
        reset_hermes_home_override(token)


def test_default_routes_create_and_holds_owner_scope_for_post_write_hooks(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    target = root / "profiles" / "trt"
    target.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    observed = {}

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=ROUTED_SKILL,
            category=None,
            create_fn=_creator(observed),
        )

    assert result["success"] is True
    assert result["owner_profile"] == "trt"
    assert result["routed_from_profile"] == "default"
    assert "hand those mutations" in result["hint"]
    assert observed["home"] == target.resolve(strict=False)
    assert token is not None
    assert get_hermes_home().resolve(strict=False) == target.resolve(strict=False)

    reset_owner_routing_scope(token)
    assert get_hermes_home().resolve(strict=False) == root.resolve(strict=False)


def test_failed_routed_create_restores_default_scope(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    (root / "profiles" / "trt").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    observed = {}

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=ROUTED_SKILL,
            category=None,
            create_fn=_creator(observed, success=False),
        )

    assert result["success"] is False
    assert token is None
    assert get_hermes_home().resolve(strict=False) == root.resolve(strict=False)


def test_named_profile_cannot_write_sideways(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    active = root / "profiles" / "growth"
    target = root / "profiles" / "trt"
    active.mkdir(parents=True)
    target.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active))
    observed = {}

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=ROUTED_SKILL,
            category=None,
            create_fn=_creator(observed),
        )

    assert result["success"] is False
    assert result["owner_profile"] == "trt"
    assert result["active_profile"] == "growth"
    assert "may not write skills sideways" in result["error"]
    assert token is None
    assert "home" not in observed


def test_owner_matching_named_profile_creates_locally(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    active = root / "profiles" / "trt"
    active.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(active))
    observed = {}

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=ROUTED_SKILL,
            category="ops",
            create_fn=_creator(observed),
        )

    assert result["success"] is True
    assert result["owner_profile"] == "trt"
    assert "routed_from_profile" not in result
    assert token is None
    assert observed["home"] == active.resolve(strict=False)
    assert observed["args"][2] == "ops"


def test_default_owned_skill_stays_on_default(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    content = ROUTED_SKILL.replace("owner_profile: trt", "owner_profile: default")
    observed = {}

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=content,
            category=None,
            create_fn=_creator(observed),
        )

    assert result["success"] is True
    assert result["owner_profile"] == "default"
    assert token is None
    assert observed["home"] == root.resolve(strict=False)


def test_unknown_owner_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    content = ROUTED_SKILL.replace("owner_profile: trt", "owner_profile: missing-profile")

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=content,
            category=None,
            create_fn=lambda *args, **kwargs: pytest.fail("create must not run"),
        )

    assert result["success"] is False
    assert "not registered" in result["error"]
    assert token is None


@pytest.mark.parametrize("bad_owner", ["../../tmp", "root", "bad/profile"])
def test_invalid_owner_is_rejected_before_profile_lookup(tmp_path, monkeypatch, bad_owner):
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    content = ROUTED_SKILL.replace("owner_profile: trt", f"owner_profile: {bad_owner}")

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(),
    ), patch(
        "hermes_cli.profiles.profile_exists",
        side_effect=AssertionError("invalid owner reached profile lookup"),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=content,
            category=None,
            create_fn=lambda *args, **kwargs: pytest.fail("create must not run"),
        )

    assert result["success"] is False
    assert "Could not resolve skill owner profile" in result["error"]
    assert token is None


def test_default_can_refuse_cross_profile_routing(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    (root / "profiles" / "trt").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(route_from_default=False),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=ROUTED_SKILL,
            category=None,
            create_fn=lambda *args, **kwargs: pytest.fail("create must not run"),
        )

    assert result["success"] is False
    assert "configured to refuse cross-profile creation" in result["error"]
    assert token is None


def test_required_owner_metadata_and_optional_compatibility(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=PLAIN_SKILL,
            category=None,
            create_fn=lambda *args, **kwargs: pytest.fail("create must not run"),
        )
    assert result["success"] is False
    assert "owner_profile" in result["error"]
    assert token is None

    observed = {}
    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value=_enabled_policy(require_owner_metadata=False),
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=PLAIN_SKILL,
            category=None,
            create_fn=_creator(observed),
        )
    assert result["success"] is True
    assert token is None
    assert observed["home"] == root.resolve(strict=False)


def test_disabled_policy_preserves_existing_create_behavior(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    observed = {}

    with patch(
        "tools.skill_owner_routing.owner_routing_policy",
        return_value={"enabled": False},
    ):
        result, token = create_skill_with_owner_routing(
            name="routed-skill",
            content=PLAIN_SKILL,
            category=None,
            create_fn=_creator(observed),
        )

    assert result["success"] is True
    assert token is None
    assert observed["home"] == root.resolve(strict=False)
