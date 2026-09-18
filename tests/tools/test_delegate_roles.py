"""Named role definitions for delegate_task (issue #112369, layer 1).

Behaviour contracts, not snapshots:

* discovery is workspace-first then user-level, and a workspace role overrides a
  same-named user role;
* a role may only NARROW a spawn — a tools allowlist is intersected with what the
  parent actually holds, and an allowlist that resolves to nothing is refused
  instead of silently degrading to "inherit everything";
* ``spawn.can_delegate: false`` really removes the ability to spawn, and
  ``model`` really pins the child's route;
* a role that cannot be resolved (unknown name, missing/foreign schema, broken
  YAML, self-inconsistent fields) fails loudly — never a silent default.
"""

import json
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_child_system_prompt, delegate_task
from tools.delegate_tool_roles import (
    ROLE_SCHEMA,
    RoleDefinition,
    RoleDefinitionError,
    available_role_names,
    resolve_role,
    role_dirs,
)


def _role_yaml(name, *, charter="Verify; do not implement.", can_delegate=False, toolsets="terminal"):
    """A well-formed role file; callers tweak the pieces under test via textwrap-ish edits."""
    return (
        f"schema: {ROLE_SCHEMA}\n"
        f"name: {name}\n"
        f"description: Test role.\n"
        f"spawn:\n  can_delegate: {'true' if can_delegate else 'false'}\n"
        f"tools:\n  mode: allowlist\n  toolsets: [{toolsets}]\n"
        f"model:\n  provider: zai\n  model: glm-5.3\n"
        f"context:\n  context_files: false\n"
        f"prompt: |\n  You are a {{name}} role instance. {charter}\n"
    )


def _write_role(directory, name, body):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def user_roles():
    """User-level role dir = ``<HERMES_HOME>/roles`` (the conftest temp home)."""
    return Path(os.environ["HERMES_HOME"]) / "roles"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A repo-like workspace whose project root discovery resolves (``.git`` marker)."""
    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True)
    monkeypatch.setenv("TERMINAL_CWD", str(ws))
    return ws


@pytest.fixture
def ws_roles(workspace):
    return workspace / ".hermes" / "roles"


def _mock_parent(depth=0, toolsets=("terminal", "file")):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.enabled_toolsets = list(toolsets)
    parent.disabled_toolsets = []
    parent.providers_allowed = parent.providers_ignored = None
    parent.providers_order = parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _mock_child():
    child = MagicMock()
    child.run_conversation.return_value = {
        "final_response": "done", "completed": True, "api_calls": 1, "messages": [],
    }
    child._delegate_saved_tool_names = []
    child._credential_pool = None
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child.model = "test"
    return child


def _spawn(role=None, *, parent=None, credentials_cfg=None, tasks=None, **kwargs):
    """Run one real ``delegate_task`` with a mocked AIAgent; returns (child, call)."""
    parent = parent or _mock_parent()
    child = _mock_child()
    call = {"kwargs": {}, "cfg": [], "calls": []}

    def _creds(cfg, _parent):
        call["cfg"].append(dict(cfg))
        return {
            "provider": cfg.get("provider"), "base_url": cfg.get("base_url"), "api_key": cfg.get("api_key"),
            "api_mode": cfg.get("api_mode"), "model": cfg.get("model"), "request_overrides": cfg.get("request_overrides"),
        }

    with patch("run_agent.AIAgent", return_value=child) as mock_agent, \
         patch("tools.delegate_tool._load_config", return_value={"max_spawn_depth": 2, "provider": "openrouter", "model": "config/model"}), \
         patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_creds):
        out = delegate_task(goal=None if tasks else kwargs.pop("goal", "verify the remediation report"), tasks=tasks, role=role,
                            parent_agent=parent, credentials_cfg=credentials_cfg, **kwargs)
        call["kwargs"] = dict(mock_agent.call_args[1]) if mock_agent.call_args else {}
        call["calls"] = [dict(c[1]) for c in mock_agent.call_args_list]
        call["out"] = out
    return child, call


# ── 1. discovery + precedence ──────────────────────────────────────────────


def test_workspace_role_overrides_user_role_and_both_dirs_are_searched(user_roles, workspace, ws_roles):
    _write_role(user_roles, "reviewer", _role_yaml("reviewer", charter="USER charter."))
    _write_role(ws_roles, "reviewer", _role_yaml("reviewer", charter="WORKSPACE charter."))
    _write_role(user_roles, "scribe", _role_yaml("scribe", charter="SCRIBE charter."))

    assert role_dirs()[0] == ws_roles
    assert role_dirs()[1] == user_roles

    resolved = resolve_role("reviewer")
    assert resolved.definition.path == ws_roles / "reviewer.yaml"
    assert "WORKSPACE charter." in resolved.definition.charter()
    # A role only present at the user level still resolves.
    assert "SCRIBE charter." in resolve_role("scribe").definition.charter()
    assert set(available_role_names()) == {"reviewer", "scribe"}


# ── 2. tools narrowing (the key protection) ────────────────────────────────


def test_role_allowlist_is_intersected_with_parent_and_never_widens(ws_roles):
    _write_role(ws_roles, "narrow", _role_yaml("narrow", toolsets="terminal, web, browser"))
    parent = _mock_parent(toolsets=("terminal", "file"))

    _child, call = _spawn("narrow", parent=parent)

    assert call["kwargs"]["enabled_toolsets"] == ["terminal"]
    assert "web" not in call["kwargs"]["enabled_toolsets"]
    assert json.loads(call["out"])["results"]


def test_role_allowlist_with_no_parent_overlap_is_refused_not_widened(ws_roles):
    _write_role(ws_roles, "wide", _role_yaml("wide", toolsets="web, browser"))
    parent = _mock_parent(toolsets=("terminal", "file"))

    child, call = _spawn("wide", parent=parent)

    assert "error" in json.loads(call["out"])
    assert "'wide'" in call["out"] and "NARROW" in call["out"]
    assert call["kwargs"] == {}  # nothing was built


# ── 3. can_delegate maps onto the existing orchestrator/leaf semantics ─────


def test_role_without_can_delegate_stays_an_orchestrator_when_depth_allows(ws_roles):
    _write_role(ws_roles, "chief", _role_yaml("chief", can_delegate=True, toolsets="terminal, delegation"))
    parent = _mock_parent(toolsets=("terminal", "file", "delegation"))

    child, call = _spawn("chief", parent=parent)

    assert child._delegate_role == "orchestrator"
    assert "delegation" in call["kwargs"]["enabled_toolsets"]


def test_can_delegate_false_role_cannot_spawn_grandchildren(ws_roles):
    _write_role(ws_roles, "reviewer", _role_yaml("reviewer", can_delegate=False, toolsets="terminal, delegation"))
    parent = _mock_parent(toolsets=("terminal", "file", "delegation"))

    child, call = _spawn("reviewer", parent=parent)

    assert child._delegate_role == "leaf"
    assert child._delegate_role_name == "reviewer"
    assert "delegation" not in call["kwargs"]["enabled_toolsets"]


# ── 4. model pin ───────────────────────────────────────────────────────────


def test_role_model_pin_beats_delegation_config(ws_roles):
    _write_role(ws_roles, "reviewer", _role_yaml("reviewer"))

    _child, call = _spawn("reviewer")

    assert call["cfg"][-1]["provider"] == "zai"
    assert call["cfg"][-1]["model"] == "glm-5.3"
    assert call["kwargs"]["provider"] == "zai"
    assert call["kwargs"]["model"] == "glm-5.3"


def test_role_model_pin_loses_to_an_explicit_caller_route(ws_roles):
    _write_role(ws_roles, "reviewer", _role_yaml("reviewer"))

    _child, call = _spawn("reviewer", credentials_cfg={"provider": "vertex", "model": "gemini-x"})

    assert call["cfg"] == [{"provider": "vertex", "model": "gemini-x"}]
    assert call["kwargs"]["provider"] == "vertex"


# ── 5. charter + context_files wiring ──────────────────────────────────────


def test_role_charter_is_injected_and_context_files_can_be_skipped(workspace, ws_roles):
    _write_role(ws_roles, "reviewer", _role_yaml("reviewer", charter="Verify; do not implement."))
    (workspace / "AGENTS.md").write_text("WORKSPACE-CONVENTIONS-SENTINEL\n", encoding="utf-8")
    _write_role(
        ws_roles, "contextual",
        _role_yaml("contextual", charter="WITH CONTEXT.").replace("context_files: false", "context_files: true"),
    )

    _child, pinned = _spawn("reviewer")
    _child2, inherit = _spawn("contextual")

    prompt = pinned["kwargs"]["ephemeral_system_prompt"]
    assert "## Role Contract" in prompt and "reviewer" in prompt
    assert "Verify; do not implement." in prompt
    assert "{name}" not in prompt
    assert "WORKSPACE-CONVENTIONS-SENTINEL" not in prompt
    # Default (no context: key / context_files: true) keeps today's AGENTS.md injection.
    assert "WORKSPACE-CONVENTIONS-SENTINEL" in inherit["kwargs"]["ephemeral_system_prompt"]


# ── 6. broken / unknown roles fail loudly ──────────────────────────────────


@pytest.mark.parametrize(
    "body,needle",
    [
        ("name: broken\nprompt: hi\n", "schema"),
        (f"schema: hermes.role/v2\nname: broken\n", "hermes.role/v1"),
        ("schema: hermes.role/v1\nname: broken\nprompt: [unclosed\n", "YAML"),
        (f"schema: {ROLE_SCHEMA}\nname: other\n", "does not match"),
        (f"schema: {ROLE_SCHEMA}\nname: broken\ntools:\n  mode: everything\n", "mode"),
        (f"schema: {ROLE_SCHEMA}\nname: broken\ntools:\n  mode: allowlist\n", "toolsets"),
    ],
)
def test_broken_role_files_raise_a_named_error(ws_roles, body, needle):
    path = _write_role(ws_roles, "broken", body)

    with pytest.raises(RoleDefinitionError) as exc:
        resolve_role("broken")

    assert needle in str(exc.value)
    assert str(path) in str(exc.value)


def test_unknown_role_name_is_a_typed_error_not_a_silent_leaf(user_roles):
    _write_role(user_roles, "reviewer", _role_yaml("reviewer"))

    with pytest.raises(RoleDefinitionError) as exc:
        resolve_role("typo-role")

    assert "typo-role" in str(exc.value)
    assert "reviewer" in str(exc.value)  # tells the caller what it could have used

    _child, call = _spawn("typo-role")
    assert "error" in json.loads(call["out"])
    assert call["kwargs"] == {}


def test_builtin_roles_still_resolve_without_a_file(user_roles):
    assert resolve_role(None).definition is None
    assert resolve_role("").name == "leaf"
    assert resolve_role("Leaf").name == "leaf"
    assert resolve_role("orchestrator").definition is None


def test_per_task_role_applies_to_that_child_only(ws_roles):
    _write_role(ws_roles, "reviewer", _role_yaml("reviewer", toolsets="terminal"))
    parent = _mock_parent(toolsets=("terminal", "file"))

    _child, call = _spawn(tasks=[
        {"goal": "verify the remediation report", "role": "reviewer"},
        {"goal": "draft the remediation summary"},
    ], parent=parent)

    assert len(call["calls"]) == 2
    assert call["calls"][0]["enabled_toolsets"] == ["terminal"]  # the role's allowlist
    assert "file" in call["calls"][1]["enabled_toolsets"]  # the unroled sibling keeps the parent's tools
    assert call["calls"][0]["provider"] == "zai" and call["calls"][1]["provider"] == "openrouter"


def test_role_discovery_follows_the_active_profile_home(tmp_path, monkeypatch):
    """A→B→A: a role defined in one home must not resolve in another (no cached discovery)."""
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    _write_role(home_a / "roles", "reviewer", _role_yaml("reviewer", charter="HOME A charter."))

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert "HOME A charter." in resolve_role("reviewer").definition.charter()
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    with pytest.raises(RoleDefinitionError):
        resolve_role("reviewer")
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert "HOME A charter." in resolve_role("reviewer").definition.charter()


# ── 7. the prompt builder itself stays independently callable ──────────────


def test_prompt_builder_role_contract_is_a_marked_section():
    prompt = _build_child_system_prompt(
        "Verify the report.", role="leaf", role_charter="You are a reviewer. Verify; do not implement.",
    )

    assert "## Role Contract" in prompt
    assert "You are a reviewer. Verify; do not implement." in prompt
    # No charter → byte-identical to the pre-role prompt.
    assert "## Role Contract" not in _build_child_system_prompt("Verify the report.", role="leaf")


def test_role_definition_is_frozen_and_exposes_derived_flags(ws_roles):
    path = _write_role(ws_roles, "reviewer", _role_yaml("reviewer"))

    role = resolve_role("reviewer").definition

    assert isinstance(role, RoleDefinition)
    assert role.injects_context_files() is False
    assert role.tools_mode == "allowlist" and tuple(role.toolsets) == ("terminal",)
    assert role.can_delegate is False
    assert role.path == path
