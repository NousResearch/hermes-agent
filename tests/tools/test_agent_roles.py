"""Tests for bounded agent roles (``tools/agent_roles.py``).

Ported semantic from OpenAI Codex's agent-role system
(``codex-rs/core/src/agent/role.rs``): a role is a *bounded override* applied
to a delegated child. A role may customize instructions, point the child at a
model, or trim toolsets — but never raise the child above the parent's
authority.

The bounded-override invariant is enforced at three points, each tested here:
instructions are appended (never replace), toolsets are intersected (never
widen), and the model override only picks a string while credentials stay
inherited.
"""

from __future__ import annotations

from unittest.mock import patch

from tools.agent_roles import (
    AgentRole,
    apply_role_instructions,
    apply_role_model,
    apply_role_toolsets,
    get_agent_roles,
    resolve_role,
)


# ── get_agent_roles ────────────────────────────────────────────────────────


def test_get_agent_roles_parses_config():
    cfg = {
        "roles": {
            "explorer": {
                "instructions": "fast codebase explorer",
                "enabled_toolsets": ["terminal", "file"],
            },
            "reviewer": {
                "instructions": "security reviewer",
                "model": "gpt-5.6-sol",
            },
        }
    }
    with patch("tools.agent_roles._load_delegation_config", return_value=cfg):
        roles = get_agent_roles()
    assert set(roles) == {"explorer", "reviewer"}
    assert roles["explorer"].instructions == "fast codebase explorer"
    assert roles["explorer"].enabled_toolsets == ["terminal", "file"]
    assert roles["reviewer"].model == "gpt-5.6-sol"


def test_get_agent_roles_skips_malformed():
    cfg = {"roles": {"bad": "not-a-dict", "good": {"instructions": "ok"}}}
    with patch("tools.agent_roles._load_delegation_config", return_value=cfg):
        roles = get_agent_roles()
    assert "good" in roles
    assert "bad" not in roles


def test_get_agent_roles_empty():
    with patch("tools.agent_roles._load_delegation_config", return_value={}):
        assert get_agent_roles() == {}
    with patch("tools.agent_roles._load_delegation_config", return_value={"roles": []}):
        assert get_agent_roles() == {}


# ── resolve_role ───────────────────────────────────────────────────────────


def test_resolve_role_returns_configured_role():
    cfg = {"roles": {"explorer": {"instructions": "x"}}}
    with patch("tools.agent_roles._load_delegation_config", return_value=cfg):
        role = resolve_role("explorer")
    assert role is not None
    assert role.name == "explorer"


def test_resolve_role_none_for_builtins_and_unknown():
    with patch("tools.agent_roles._load_delegation_config", return_value={}):
        assert resolve_role(None) is None
        assert resolve_role("") is None
        assert resolve_role("leaf") is None
        assert resolve_role("orchestrator") is None
        assert resolve_role("nope") is None


# ── apply_role_instructions (append, never replace) ────────────────────────


def test_apply_role_instructions_appends():
    base = "You are a focused subagent."
    role = AgentRole(name="explorer", instructions="Be fast.")
    out = apply_role_instructions(base, role)
    assert out.startswith(base)
    assert "## Agent role: explorer" in out
    assert "Be fast." in out


def test_apply_role_instructions_noop_without_role_or_text():
    base = "You are a focused subagent."
    assert apply_role_instructions(base, None) == base
    assert apply_role_instructions(base, AgentRole(name="x")) == base


# ── apply_role_toolsets (intersect, never widen) ───────────────────────────


def test_apply_role_toolsets_intersects():
    child = ["terminal", "file", "web"]
    role = AgentRole(name="explorer", enabled_toolsets=["terminal", "file"])
    assert apply_role_toolsets(child, role) == ["terminal", "file"]


def test_apply_role_toolsets_cannot_widen():
    child = ["terminal"]
    role = AgentRole(name="x", enabled_toolsets=["terminal", "web", "delegation"])
    # The role lists MORE than the child has — intersection keeps child's set.
    assert apply_role_toolsets(child, role) == ["terminal"]


def test_apply_role_toolsets_noop_without_role_or_list():
    child = ["terminal", "file"]
    assert apply_role_toolsets(child, None) == child
    assert apply_role_toolsets(child, AgentRole(name="x")) == child


# ── apply_role_model (string pick, credentials inherited) ──────────────────


def test_apply_role_model_precedence_caller_wins():
    role = AgentRole(name="r", model="role-model")
    assert apply_role_model(role, "caller-model", "parent-model") == "caller-model"


def test_apply_role_model_role_over_parent():
    role = AgentRole(name="r", model="role-model")
    assert apply_role_model(role, None, "parent-model") == "role-model"


def test_apply_role_model_falls_back_to_parent():
    role = AgentRole(name="r", model=None)
    assert apply_role_model(role, None, "parent-model") == "parent-model"


def test_apply_role_model_none_without_role():
    assert apply_role_model(None, "caller", "parent") == "caller"
    assert apply_role_model(None, None, "parent") == "parent"
