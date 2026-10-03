"""Integration tests for the owner-aware skill_manage registration adapter."""

from __future__ import annotations

import json

from hermes_constants import get_hermes_home
from tools import skill_manager_tool as manager
from tools import skill_owner_routing_tool as routed
from tools.registry import registry


SKILL = """\
---
name: routed-skill
description: Use when routing a learned skill.
metadata:
  hermes:
    owner_profile: trt
---

# Routed Skill

Do the routed thing.
"""


def _enabled_policy():
    return {
        "enabled": True,
        "require_owner_metadata": True,
        "route_from_default": True,
    }


def test_registered_skill_manage_exposes_owner_contract():
    entry = registry.get_entry("skill_manage")
    assert entry is not None
    assert entry.handler is not None
    assert "metadata.hermes.owner_profile" in entry.schema["description"]
    assert "may not write sideways" in entry.schema["description"]


def test_gate_runs_in_caller_scope_and_full_create_runs_in_owner_scope(
    tmp_path, monkeypatch
):
    root = tmp_path / "fleet"
    owner = root / "profiles" / "trt"
    owner.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    observed = {}

    def fake_gate(*args, **kwargs):
        observed["gate_home"] = get_hermes_home().resolve(strict=False)
        return None

    def fake_base(**kwargs):
        observed["base_home"] = get_hermes_home().resolve(strict=False)
        observed["base_bypass"] = manager._skill_gate_bypass.get()
        observed["base_kwargs"] = kwargs
        return json.dumps({"success": True, "message": "created", "path": "routed-skill"})

    monkeypatch.setattr(routed, "owner_routing_policy", _enabled_policy)
    monkeypatch.setattr(manager, "_apply_skill_write_gate", fake_gate)
    monkeypatch.setattr(manager, "skill_manage", fake_base)

    result = json.loads(
        routed.skill_manage(
            action="create",
            name="routed-skill",
            content=SKILL,
            task_id="task-1",
            session_id="session-1",
        )
    )

    assert observed["gate_home"] == root.resolve(strict=False)
    assert observed["base_home"] == owner.resolve(strict=False)
    assert observed["base_bypass"] is True
    assert observed["base_kwargs"]["task_id"] == "task-1"
    assert observed["base_kwargs"]["session_id"] == "session-1"
    assert result["success"] is True
    assert result["owner_profile"] == "trt"
    assert result["routed_from_profile"] == "default"
    assert get_hermes_home().resolve(strict=False) == root.resolve(strict=False)


def test_staged_create_never_switches_to_owner_or_invokes_base(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    (root / "profiles" / "trt").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    staged = json.dumps({"success": True, "staged": True, "pending_id": "abc123"})
    observed = {}

    def fake_gate(*args, **kwargs):
        observed["gate_home"] = get_hermes_home().resolve(strict=False)
        return staged

    def fail_base(**kwargs):
        raise AssertionError("staged create must not invoke the base manager")

    monkeypatch.setattr(routed, "owner_routing_policy", _enabled_policy)
    monkeypatch.setattr(manager, "_apply_skill_write_gate", fake_gate)
    monkeypatch.setattr(manager, "skill_manage", fail_base)

    assert routed.skill_manage(action="create", name="routed-skill", content=SKILL) == staged
    assert observed["gate_home"] == root.resolve(strict=False)
    assert get_hermes_home().resolve(strict=False) == root.resolve(strict=False)


def test_approved_pending_create_uses_same_owner_route(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    owner = root / "profiles" / "trt"
    owner.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    observed = {}

    def fake_gate(*args, **kwargs):
        observed["gate_bypass"] = manager._skill_gate_bypass.get()
        observed["gate_home"] = get_hermes_home().resolve(strict=False)
        return None

    def fake_base(**kwargs):
        observed["base_home"] = get_hermes_home().resolve(strict=False)
        observed["base_bypass"] = manager._skill_gate_bypass.get()
        return json.dumps({"success": True, "message": "created", "path": "routed-skill"})

    monkeypatch.setattr(routed, "owner_routing_policy", _enabled_policy)
    monkeypatch.setattr(manager, "_apply_skill_write_gate", fake_gate)
    monkeypatch.setattr(manager, "skill_manage", fake_base)

    result = json.loads(
        routed.apply_skill_pending(
            {
                "action": "create",
                "name": "routed-skill",
                "content": SKILL,
            }
        )
    )

    assert observed["gate_bypass"] is True
    assert observed["gate_home"] == root.resolve(strict=False)
    assert observed["base_bypass"] is True
    assert observed["base_home"] == owner.resolve(strict=False)
    assert result["owner_profile"] == "trt"
    assert get_hermes_home().resolve(strict=False) == root.resolve(strict=False)


def test_missing_content_preserves_base_manager_error_path(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    observed = {}

    def fake_gate(*args, **kwargs):
        return None

    def fake_base(**kwargs):
        observed["bypass"] = manager._skill_gate_bypass.get()
        return json.dumps({"success": False, "error": "content is required for 'create'."})

    monkeypatch.setattr(routed, "owner_routing_policy", _enabled_policy)
    monkeypatch.setattr(manager, "_apply_skill_write_gate", fake_gate)
    monkeypatch.setattr(manager, "skill_manage", fake_base)

    result = json.loads(routed.skill_manage(action="create", name="routed-skill"))
    assert result["error"] == "content is required for 'create'."
    assert observed["bypass"] is True


def test_disabled_policy_delegates_without_pre_gating(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    observed = {"gate": 0, "base": 0}

    def fake_gate(*args, **kwargs):
        observed["gate"] += 1
        return None

    def fake_base(**kwargs):
        observed["base"] += 1
        observed["home"] = get_hermes_home().resolve(strict=False)
        return json.dumps({"success": True})

    monkeypatch.setattr(routed, "owner_routing_policy", lambda: {"enabled": False})
    monkeypatch.setattr(manager, "_apply_skill_write_gate", fake_gate)
    monkeypatch.setattr(manager, "skill_manage", fake_base)

    assert json.loads(routed.skill_manage(action="create", name="plain", content=SKILL))["success"] is True
    # The adapter does not run its own gate when routing is disabled; the real
    # base manager owns that path exactly as before.
    assert observed["gate"] == 0
    assert observed["base"] == 1
    assert observed["home"] == root.resolve(strict=False)


def test_non_create_actions_are_identical_pass_through(tmp_path, monkeypatch):
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    observed = {}

    def fake_base(**kwargs):
        observed.update(kwargs)
        return json.dumps({"success": True, "message": "patched"})

    monkeypatch.setattr(manager, "skill_manage", fake_base)
    result = json.loads(
        routed.skill_manage(
            action="patch",
            name="existing",
            old_string="old",
            new_string="new",
            replace_all=True,
        )
    )

    assert result["message"] == "patched"
    assert observed["action"] == "patch"
    assert observed["old_string"] == "old"
    assert observed["new_string"] == "new"
    assert observed["replace_all"] is True
