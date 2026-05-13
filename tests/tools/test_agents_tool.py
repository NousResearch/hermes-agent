"""Tests for tools/agents_tool.py — native agent registry tools."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
from tools.agents_tool import agents_list, agent_view, assign_agent


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for the duration of a test."""
    hh = tmp_path / ".hermes"
    hh.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hh))
    return hh


@pytest.fixture
def project_root(tmp_path):
    """A fake git-backed project root."""
    pr = tmp_path / "my-project"
    pr.mkdir()
    (pr / ".git").mkdir()
    return pr


@pytest.fixture
def global_agents_dir(hermes_home):
    """Global agents directory."""
    d = hermes_home / "agents"
    d.mkdir()
    return d


@pytest.fixture
def project_agents_dir(project_root):
    """Project-local agents directory."""
    d = project_root / ".hermes" / "agents"
    d.mkdir(parents=True)
    return d


def make_agent(path: Path, name: str, description: str = "A test agent", **frontmatter):
    """Write a minimal valid agent file."""
    fm = {"schema_version": 1, "name": name, "description": description}
    fm.update(frontmatter)
    lines = ["---", *[f"{k}: {v}" for k, v in fm.items()], "---", "", "This is the agent prompt."]
    path.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration checks
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentsToolRegistration:
    """Verify tools register with correct toolset membership."""

    def test_agents_list_registered_in_agents_toolset(self):
        """agents_list must be registered in the 'agents' toolset."""
        from tools.registry import registry
        entry = registry.get_entry("agents_list")
        assert entry is not None, "agents_list not found in registry"
        assert entry.toolset == "agents", f"agents_list toolset is {entry.toolset}, expected 'agents'"

    def test_agent_view_registered_in_agents_toolset(self):
        """agent_view must be registered in the 'agents' toolset."""
        from tools.registry import registry
        entry = registry.get_entry("agent_view")
        assert entry is not None, "agent_view not found in registry"
        assert entry.toolset == "agents", f"agent_view toolset is {entry.toolset}, expected 'agents'"

    def test_assign_agent_registered_in_delegation_toolset(self):
        """assign_agent must be registered in the 'delegation' toolset."""
        from tools.registry import registry
        entry = registry.get_entry("assign_agent")
        assert entry is not None, "assign_agent not found in registry"
        assert entry.toolset == "delegation", f"assign_agent toolset is {entry.toolset}, expected 'delegation'"


class TestAgentsToolsetIntegration:
    """Verify toolset resolution includes the correct agents tools."""

    def test_resolve_toolset_agents_contains_list_and_view(self):
        """resolve_toolset('agents') must contain agents_list and agent_view."""
        from toolsets import resolve_toolset
        tools = resolve_toolset("agents")
        assert "agents_list" in tools, "agents_list missing from agents toolset"
        assert "agent_view" in tools, "agent_view missing from agents toolset"
        # assign_agent belongs to delegation, not agents
        assert "assign_agent" not in tools, "assign_agent should NOT be in agents toolset"

    def test_resolve_toolset_delegation_contains_assign_agent(self):
        """resolve_toolset('delegation') must contain assign_agent."""
        from toolsets import resolve_toolset
        tools = resolve_toolset("delegation")
        assert "assign_agent" in tools, "assign_agent missing from delegation toolset"

    def test_agents_toolset_does_not_contain_assign_agent(self):
        """The 'agents' toolset must be read-only — no delegation."""
        from toolsets import resolve_toolset
        tools = resolve_toolset("agents")
        assert "assign_agent" not in tools


# ─────────────────────────────────────────────────────────────────────────────
# agents_list
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentsList:
    """Tests for agents_list()."""

    def test_returns_json_with_success(self, global_agents_dir):
        """agents_list must return JSON with success=True."""
        result = agents_list(workdir=str(global_agents_dir))
        parsed = json.loads(result)
        assert parsed.get("success") is True

    def test_output_shape_excludes_prompt(self, global_agents_dir):
        """agents_list must NOT include full prompt bodies."""
        make_agent(global_agents_dir / "test-agent.md", "test-agent", description="A test agent")
        result = agents_list(workdir=str(global_agents_dir))
        parsed = json.loads(result)
        agents = parsed.get("agents", [])
        assert len(agents) >= 1
        for agent in agents:
            assert "prompt" not in agent, "agents_list must not include prompt bodies"
            assert "prompt" not in agent.get("description", "")

    def test_count_field_present(self, global_agents_dir):
        """agents_list must include a count field."""
        make_agent(global_agents_dir / "agent-a.md", "agent-a", description="Agent A")
        make_agent(global_agents_dir / "agent-b.md", "agent-b", description="Agent B")
        result = agents_list(workdir=str(global_agents_dir))
        parsed = json.loads(result)
        assert "count" in parsed
        assert parsed["count"] >= 2

    def test_empty_dir_returns_empty_list(self, global_agents_dir):
        """No agents in dir → empty list, not an error."""
        result = agents_list(workdir=str(global_agents_dir))
        parsed = json.loads(result)
        assert parsed.get("success") is True
        assert parsed.get("count", 0) == 0
        assert parsed.get("agents", []) == []

    def test_category_filter_no_match(self, global_agents_dir):
        """category filter with no matching agents returns empty list."""
        make_agent(global_agents_dir / "web-agent.md", "web-agent", description="Web agent",
                   tags=["web", "research"])
        result = agents_list(category="nonexistent-tag", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        assert parsed.get("count", 0) == 0

    def test_category_filter_matches(self, global_agents_dir):
        """category filter returns only matching agents."""
        make_agent(global_agents_dir / "web-agent.md", "web-agent", description="Web agent",
                   tags=["web"])
        make_agent(global_agents_dir / "code-agent.md", "code-agent", description="Code agent",
                   tags=["code"])
        result = agents_list(category="web", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        names = [a["name"] for a in parsed.get("agents", [])]
        assert "web-agent" in names
        assert "code-agent" not in names

    def test_disabled_agents_excluded_by_default(self, global_agents_dir):
        """Disabled agents are excluded from default output."""
        make_agent(global_agents_dir / "enabled-agent.md", "enabled-agent",
                   description="Enabled agent", enabled=True)
        make_agent(global_agents_dir / "disabled-agent.md", "disabled-agent",
                   description="Disabled agent", enabled=False)
        result = agents_list(workdir=str(global_agents_dir))
        parsed = json.loads(result)
        names = [a["name"] for a in parsed.get("agents", [])]
        assert "enabled-agent" in names
        assert "disabled-agent" not in names

    def test_disabled_agents_included_when_requested(self, global_agents_dir):
        """include_disabled=True includes disabled agents."""
        make_agent(global_agents_dir / "disabled-agent.md", "disabled-agent",
                   description="Disabled agent", enabled=False)
        result = agents_list(include_disabled=True, workdir=str(global_agents_dir))
        parsed = json.loads(result)
        names = [a["name"] for a in parsed.get("agents", [])]
        assert "disabled-agent" in names

    def test_list_summary_fields(self, global_agents_dir):
        """Each agent in list must have expected summary fields."""
        make_agent(global_agents_dir / "review-agent.md", "review-agent",
                   description="A code review agent", tags=["review"],
                   routing={"mode": "inherit", "model": "claude-3-5"})
        result = agents_list(workdir=str(global_agents_dir))
        parsed = json.loads(result)
        agent = parsed["agents"][0]
        # Must have these fields
        assert agent["name"] == "review-agent"
        assert agent["description"] == "A code review agent"
        assert agent["tags"] == ["review"]
        assert agent["source"] == "global"
        assert agent["enabled"] is True
        # Routing summary
        assert "routing" in agent
        assert agent["routing"]["mode"] == "inherit"
        # Must NOT have prompt
        assert "prompt" not in agent


# ─────────────────────────────────────────────────────────────────────────────
# agent_view
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentView:
    """Tests for agent_view()."""

    def test_returns_json_with_success(self, global_agents_dir):
        """agent_view must return JSON with success=True for a valid agent."""
        make_agent(global_agents_dir / "my-agent.md", "my-agent", description="My agent")
        result = agent_view(name="my-agent", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        assert parsed.get("success") is True

    def test_output_includes_prompt(self, global_agents_dir):
        """agent_view must include the full prompt."""
        make_agent(global_agents_dir / "prompt-agent.md", "prompt-agent",
                   description="Has a prompt",
                   tags=["test"])
        result = agent_view(name="prompt-agent", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        assert "prompt" in parsed.get("agent", {}), "agent_view must include prompt"
        assert "This is the agent prompt." in parsed["agent"]["prompt"]

    def test_output_includes_path_and_source(self, global_agents_dir):
        """agent_view must include path and source."""
        make_agent(global_agents_dir / "source-agent.md", "source-agent",
                   description="Source agent")
        result = agent_view(name="source-agent", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        agent = parsed.get("agent", {})
        assert "path" in agent
        assert "source" in agent
        assert agent["source"] == "global"

    def test_invalid_name_returns_error(self, global_agents_dir):
        """Non-existent agent name returns success=False."""
        result = agent_view(name="nonexistent-agent", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        assert parsed.get("success") is False, "agent_view should return success=False for unknown agent"
        assert "error" in parsed or "agent" in parsed

    def test_path_traversal_rejected(self, global_agents_dir):
        """agent_view must not allow path traversal via name."""
        # validate_agent_name should reject names with slashes/dots
        result = agent_view(name="../etc/passwd", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        # Should be rejected as invalid name (returns success=False)
        assert parsed.get("success") is False

    def test_disabled_agent_can_be_viewed(self, global_agents_dir):
        """Disabled agents can still be viewed (view is read-only)."""
        make_agent(global_agents_dir / "disabled-agent.md", "disabled-agent",
                   description="A disabled agent", enabled=False)
        result = agent_view(name="disabled-agent", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        assert parsed.get("success") is True
        assert parsed.get("agent", {}).get("enabled") is False

    def test_to_dict_includes_all_expected_fields(self, global_agents_dir):
        """agent_view returns full to_dict output."""
        make_agent(global_agents_dir / "full-agent.md", "full-agent",
                   description="Full agent",
                   tags=["test", "demo"],
                   routing={"mode": "hermes", "provider": "openai"},
                   tools={"mode": "restrict", "allow_toolsets": ["web"]},
                   delegation={"role": "orchestrator"})
        result = agent_view(name="full-agent", workdir=str(global_agents_dir))
        parsed = json.loads(result)
        agent = parsed.get("agent", {})
        assert agent["name"] == "full-agent"
        assert agent["description"] == "Full agent"
        assert agent["tags"] == ["test", "demo"]
        assert agent["routing"]["mode"] == "hermes"
        assert agent["routing"]["provider"] == "openai"
        assert agent["tools"]["mode"] == "restrict"
        assert agent["tools"]["allow_toolsets"] == ["web"]
        assert agent["delegation_role"] == "orchestrator"


# ─────────────────────────────────────────────────────────────────────────────
# assign_agent
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignAgent:
    """Tests for assign_agent()."""

    def test_without_parent_agent_returns_error(self, global_agents_dir):
        """assign_agent without parent_agent must return error JSON."""
        make_agent(global_agents_dir / "some-agent.md", "some-agent", description="Some agent")
        result = assign_agent(
            agent_name="some-agent",
            task="Do something",
            parent_agent=None,
        )
        parsed = json.loads(result)
        assert parsed.get("success") is False, "assign_agent without parent_agent should fail"
        assert "error" in parsed

    def test_disabled_agent_returns_error(self, global_agents_dir):
        """assign_agent with a disabled agent must return error."""
        make_agent(global_agents_dir / "disabled-delegation.md", "disabled-delegation",
                   description="Disabled agent", enabled=False)
        mock_parent = MagicMock()
        result = assign_agent(
            agent_name="disabled-delegation",
            task="Do something",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is False
        assert "error" in parsed

    def test_unknown_agent_returns_error(self, global_agents_dir):
        """assign_agent with unknown agent must return error."""
        mock_parent = MagicMock()
        result = assign_agent(
            agent_name="does-not-exist",
            task="Do something",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is False

    def test_assign_agent_calls_delegate_task_with_goal(self, global_agents_dir, monkeypatch):
        """assign_agent must call delegate_task with goal=task."""
        make_agent(global_agents_dir / "delegate-agent.md", "delegate-agent",
                   description="Delegatable agent")
        mock_parent = MagicMock()

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": [{"goal": kwargs.get("goal"), "output": "done"}]})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        result = assign_agent(
            agent_name="delegate-agent",
            task="Review the code",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        assert len(recorded_calls) == 1
        assert recorded_calls[0]["goal"] == "Review the code"

    def test_assign_agent_compiles_context_with_prompt(self, global_agents_dir, monkeypatch):
        """assign_agent context must include agent name, path, prompt, and user context."""
        make_agent(global_agents_dir / "context-agent.md", "context-agent",
                   description="Context test agent")
        mock_parent = MagicMock()

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="context-agent",
            task="Test task",
            context="User context here",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        assert len(recorded_calls) == 1
        compiled = recorded_calls[0]["context"]
        assert "Named agent: context-agent" in compiled
        assert "Agent instructions" in compiled
        assert "This is the agent prompt." in compiled
        assert "User context here" in compiled

    def test_assign_agent_toolsets_from_agent_allow_toolsets(self, global_agents_dir, monkeypatch):
        """When tools.mode==restrict, effective toolsets come from agent.tools.allow_toolsets."""
        make_agent(global_agents_dir / "restrict-agent.md", "restrict-agent",
                   description="Restricted agent",
                   tools={"mode": "restrict", "allow_toolsets": ["web", "terminal"]})
        mock_parent = MagicMock()

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="restrict-agent",
            task="Task",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["toolsets"] == ["web", "terminal"]

    def test_assign_agent_toolsets_none_when_no_restrict(self, global_agents_dir, monkeypatch):
        """When tools.mode != restrict, toolsets is None (not restricted)."""
        make_agent(global_agents_dir / "open-agent.md", "open-agent",
                   description="Open agent",
                   tools={"mode": "inherit"})  # inherit = no restriction
        mock_parent = MagicMock()

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="open-agent",
            task="Task",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        assert len(recorded_calls) == 1
        # When mode is not restrict, toolsets is None
        assert recorded_calls[0]["toolsets"] is None

    def test_assign_agent_role_leaf_by_default(self, global_agents_dir, monkeypatch):
        """Default role is 'leaf'."""
        make_agent(global_agents_dir / "role-agent.md", "role-agent",
                   description="Role test agent")
        mock_parent = MagicMock()

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="role-agent",
            task="Task",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["role"] == "leaf"

    def test_assign_agent_role_from_delegation_role(self, global_agents_dir, monkeypatch):
        """When agent has delegation_role, it is used."""
        make_agent(global_agents_dir / "orch-agent.md", "orch-agent",
                   description="Orchestrator agent",
                   delegation={"role": "orchestrator"})
        mock_parent = MagicMock()

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="orch-agent",
            task="Task",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["role"] == "orchestrator"

    def test_assign_agent_runtime_toolsets_override(self, global_agents_dir, monkeypatch):
        """Runtime toolsets override agent defaults."""
        make_agent(global_agents_dir / "runtime-agent.md", "runtime-agent",
                   description="Runtime override agent",
                   tools={"mode": "restrict", "allow_toolsets": ["web"]})
        mock_parent = MagicMock()

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="runtime-agent",
            task="Task",
            toolsets=["terminal", "file"],
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["toolsets"] == ["terminal", "file"]

    def test_assign_agent_runtime_role_override(self, global_agents_dir, monkeypatch):
        """Runtime role overrides agent delegation_role."""
        make_agent(global_agents_dir / "override-role.md", "override-role",
                   description="Override role agent",
                   delegation={"role": "leaf"})
        mock_parent = MagicMock()

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="override-role",
            task="Task",
            role="orchestrator",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["role"] == "orchestrator"

    def test_assign_agent_returns_success_with_metadata(self, global_agents_dir, monkeypatch):
        """assign_agent returns success=True with agent metadata and parsed result."""
        make_agent(global_agents_dir / "result-agent.md", "result-agent",
                   description="Result agent")
        mock_parent = MagicMock()

        def mock_delegate_task(**kwargs):
            return json.dumps({
                "success": True,
                "results": [{"goal": kwargs["goal"], "output": "Fixed the bug"}]
            })

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        result = assign_agent(
            agent_name="result-agent",
            task="Fix the bug",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is True
        assert "agent" in parsed
        assert parsed["agent"]["name"] == "result-agent"
        assert "result" in parsed
        assert "Fixed the bug" in str(parsed["result"])

    def test_assign_agent_returns_raw_when_not_json(self, global_agents_dir, monkeypatch):
        """Non-JSON delegate_task return is passed through as-is."""
        make_agent(global_agents_dir / "raw-agent.md", "raw-agent",
                   description="Raw agent")
        mock_parent = MagicMock()

        def mock_delegate_task(**kwargs):
            return "raw string result"

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        result = assign_agent(
            agent_name="raw-agent",
            task="Task",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is True
        assert parsed.get("result") == "raw string result"

    def test_assign_agent_passes_parent_agent(self, global_agents_dir, monkeypatch):
        """assign_agent must pass parent_agent to delegate_task."""
        make_agent(global_agents_dir / "parent-agent.md", "parent-agent",
                   description="Parent test agent")
        mock_parent = MagicMock()
        mock_parent._subagent_id = "parent-123"

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="parent-agent",
            task="Task",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        assert len(recorded_calls) == 1
        assert recorded_calls[0]["parent_agent"] is mock_parent


# ─────────────────────────────────────────────────────────────────────────────
# assign_agent — routing mode rejection (PR1 constraints)
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignAgentRoutingRejection:
    """Tests for routing mode / runner metadata rejection in PR1."""

    def test_hermes_routing_mode_passes_native_provider_model_to_delegate_task(self, global_agents_dir, monkeypatch):
        """Agents with routing.mode='hermes' must route children to their configured provider/model."""
        make_agent(global_agents_dir / "hermes-agent.md", "hermes-agent",
                   description="Hermes routing agent",
                   routing={"mode": "hermes", "provider": "kimi-coding", "model": "kimi-k2.6"})
        mock_parent = MagicMock()
        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)
        result = assign_agent(
            agent_name="hermes-agent",
            task="Do something",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is True
        assert len(recorded_calls) == 1
        assert recorded_calls[0]["provider"] == "kimi-coding"
        assert recorded_calls[0]["model"] == "kimi-k2.6"

    def test_assign_agent_passes_all_native_provider_routes(self, global_agents_dir, monkeypatch):
        """Kimi, MiniMax, and GLM/Z.AI agent routes are forwarded exactly to delegation."""
        routes = {
            "kimi-agent": ("kimi-coding", "kimi-k2.6"),
            "minimax-agent": ("minimax", "MiniMax-M2.7-highspeed"),
            "glm-agent": ("zai", "glm-5.1"),
        }
        for name, (provider, model) in routes.items():
            make_agent(global_agents_dir / f"{name}.md", name,
                       description=f"{provider} route",
                       routing={"mode": "hermes", "provider": provider, "model": model})

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)
        mock_parent = MagicMock()
        for name in routes:
            result = assign_agent(
                agent_name=name,
                task="Do something",
                parent_agent=mock_parent,
                workdir=str(global_agents_dir),
            )
            assert json.loads(result).get("success") is True

        assert [(c["provider"], c["model"]) for c in recorded_calls] == list(routes.values())

    def test_rejects_acp_routing_mode(self, global_agents_dir):
        """Agents with routing.mode='acp' must be rejected."""
        make_agent(global_agents_dir / "acp-agent.md", "acp-agent",
                   description="ACP routing agent",
                   routing={"mode": "acp"})
        mock_parent = MagicMock()
        result = assign_agent(
            agent_name="acp-agent",
            task="Do something",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is False
        assert "routing.mode='acp'" in parsed["error"]
        assert "Supported routing modes: inherit, hermes" in parsed["error"]

    def test_accepts_inherit_routing_mode(self, global_agents_dir, monkeypatch):
        """Agents with routing.mode='inherit' (default) must be accepted."""
        make_agent(global_agents_dir / "inherit-agent.md", "inherit-agent",
                   description="Inherit routing agent",
                   routing={"mode": "inherit"})

        def mock_delegate_task(**kwargs):
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        mock_parent = MagicMock()
        result = assign_agent(
            agent_name="inherit-agent",
            task="Do something",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is True

    def test_rejects_acp_command(self, global_agents_dir):
        """Agents with routing.acp_command must be rejected even with mode=inherit."""
        make_agent(global_agents_dir / "acpcmd-agent.md", "acpcmd-agent",
                   description="ACP command agent",
                   routing={"mode": "inherit", "acp_command": "some-cmd"})
        mock_parent = MagicMock()
        result = assign_agent(
            agent_name="acpcmd-agent",
            task="Do something",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is False
        assert "acp_command" in parsed["error"]
        assert "PR1" in parsed["error"]

    def test_rejects_unknown_runner_mode_inline(self, global_agents_dir):
        """Agents with unsupported runner.mode values must be rejected."""
        make_agent(global_agents_dir / "runner-agent.md", "runner-agent",
                   description="Inline runner agent",
                   routing={"mode": "inherit", "runner": {"mode": "subprocess"}})
        mock_parent = MagicMock()
        result = assign_agent(
            agent_name="runner-agent",
            task="Do something",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed.get("success") is False
        assert "runner.mode='subprocess'" in parsed["error"]
        assert "delegate_task, cli" in parsed["error"]


# ─────────────────────────────────────────────────────────────────────────────
# assign_agent — CLI runner execution
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignAgentCliRunner:
    """Tests for trusted CLI-backed named agents."""

    def test_cli_runner_invokes_trusted_configured_command(self, global_agents_dir, monkeypatch):
        (global_agents_dir / "cli-agent.md").write_text("""---
schema_version: 1
name: cli-agent
description: CLI-backed agent
runner:
  mode: cli
  name: claude-cli
  continue: off
---

You are a CLI agent.
""")
        mock_parent = MagicMock()
        mock_parent.session_id = "parent-session"
        recorded = {}

        class Completed:
            returncode = 0
            stdout = '{"session_id":"external-123","result":"ok"}\n'
            stderr = ""

        def fake_run(cmd, **kwargs):
            recorded["cmd"] = cmd
            recorded["kwargs"] = kwargs
            return Completed()

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "agent_runners": {
                "claude-cli": {
                    "type": "cli",
                    "command": "claude",
                    "args": ["-p", "--output-format", "stream-json"],
                    "resume_arg": "--resume",
                    "allowed_from_project_agents": True,
                }
            }
        })

        result = assign_agent(
            agent_name="cli-agent",
            task="Review the diff",
            context="Only inspect files.",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)

        assert parsed["success"] is True
        assert parsed["runner"]["mode"] == "cli"
        assert parsed["runner"]["name"] == "claude-cli"
        assert recorded["cmd"] == ["claude", "-p", "--output-format", "stream-json"]
        assert "Review the diff" in recorded["kwargs"]["input"]
        assert "You are a CLI agent." in recorded["kwargs"]["input"]
        assert parsed["external_session_id"] == "external-123"
        assert "command" not in parsed

    def test_cli_runner_auto_resume_uses_stored_external_session(self, global_agents_dir, monkeypatch):
        (global_agents_dir / "resume-agent.md").write_text("""---
schema_version: 1
name: resume-agent
description: Resume CLI agent
runner:
  mode: cli
  name: gemini-cli
  continue: auto
---

You resume external CLI sessions.
""")
        mock_parent = MagicMock()
        mock_parent.session_id = "parent-session"
        recorded = {}

        class Completed:
            returncode = 0
            stdout = "done"
            stderr = ""

        def fake_run(cmd, **kwargs):
            recorded["cmd"] = cmd
            return Completed()

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "agent_runners": {
                "gemini-cli": {
                    "type": "cli",
                    "command": "gemini",
                    "args": ["--output-format", "stream-json"],
                    "resume_arg": "--resume",
                    "allowed_from_project_agents": True,
                }
            }
        })
        from agent import agent_runner
        monkeypatch.setattr(agent_runner.CliSessionStore, "get", lambda self, **kw: "gemini-session-777")

        result = assign_agent(
            agent_name="resume-agent",
            task="Continue",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)

        assert parsed["success"] is True
        assert parsed["runner"]["resumed"] is True
        assert recorded["cmd"] == ["gemini", "--output-format", "stream-json", "--resume", "gemini-session-777"]

    def test_cli_runner_require_continuation_fails_without_stored_session(self, global_agents_dir, monkeypatch):
        (global_agents_dir / "require-agent.md").write_text("""---
schema_version: 1
name: require-agent
description: Require resume CLI agent
runner:
  mode: cli
  name: claude-cli
  continue: require
---

You require an existing external session.
""")
        mock_parent = MagicMock()
        mock_parent.session_id = "parent-session"
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "agent_runners": {
                "claude-cli": {
                    "type": "cli",
                    "command": "claude",
                    "args": ["-p"],
                    "resume_arg": "--resume",
                    "allowed_from_project_agents": True,
                }
            }
        })

        result = assign_agent(
            agent_name="require-agent",
            task="Continue",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)

        assert parsed["success"] is False
        assert "requires an existing CLI session" in parsed["error"]

    def test_cli_runner_session_persistence_failure_is_nonfatal(self, global_agents_dir, monkeypatch):
        """A successful CLI run still succeeds if persisting external session id fails."""
        (global_agents_dir / "store-fail-agent.md").write_text("""---
schema_version: 1
name: store-fail-agent
description: Store failure CLI agent
runner:
  mode: cli
  name: claude-cli
  continue: off
---

You are a CLI agent.
""")
        mock_parent = MagicMock()
        mock_parent.session_id = "parent-session"

        class Completed:
            returncode = 0
            stdout = '{"session_id":"external-123","result":"ok"}\n'
            stderr = ""

        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "agent_runners": {
                "claude-cli": {
                    "type": "cli",
                    "command": "claude",
                    "args": ["-p"],
                    "allowed_from_project_agents": True,
                }
            }
        })
        from agent import agent_runner
        monkeypatch.setattr(agent_runner.CliSessionStore, "set", lambda self, **kw: (_ for _ in ()).throw(OSError("disk full")))

        result = assign_agent(
            agent_name="store-fail-agent",
            task="Review",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)

        assert parsed["success"] is True
        assert parsed["external_session_id"] == "external-123"
        assert "warning" in parsed
        assert "Failed to persist CLI session" in parsed["warning"]

    def test_project_cli_runner_must_be_allowed_by_trusted_config(self, project_agents_dir, project_root, monkeypatch):
        (project_agents_dir / "project-cli.md").write_text("""---
schema_version: 1
name: project-cli
description: Project CLI agent
runner:
  mode: cli
  name: locked-cli
  continue: off
---

Project agent prompt.
""")
        mock_parent = MagicMock()
        mock_parent.session_id = "parent-session"
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "agent_runners": {
                "locked-cli": {
                    "type": "cli",
                    "command": "locked",
                    "args": [],
                    "allowed_from_project_agents": False,
                }
            }
        })

        result = assign_agent(
            agent_name="project-cli",
            task="Run",
            parent_agent=mock_parent,
            workdir=str(project_root),
        )
        parsed = json.loads(result)

        assert parsed["success"] is False
        assert "not allowed for project-local agents" in parsed["error"]


# ─────────────────────────────────────────────────────────────────────────────
# assign_agent — runtime trace
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignAgentRuntimeTrace:
    """Tests for runtime observability events emitted by assign_agent()."""

    def test_cli_runner_emits_runtime_trace_without_command_argv(self, global_agents_dir, monkeypatch):
        (global_agents_dir / "trace-cli.md").write_text("""---
schema_version: 1
name: trace-cli
description: Traceable CLI agent
runner:
  mode: cli
  name: trace-runner
  continue: off
---

You are a traceable CLI agent.
""")
        mock_parent = MagicMock()
        mock_parent.session_id = "trace-parent-session"

        class Completed:
            returncode = 0
            stdout = '{"session_id":"external-trace-123","result":"ok"}\n'
            stderr = ""

        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "agent_runners": {
                "trace-runner": {
                    "type": "cli",
                    "command": "trace-secret-command",
                    "args": ["--api-key", "should-not-appear"],
                    "allowed_from_project_agents": True,
                }
            }
        })

        result = assign_agent(
            agent_name="trace-cli",
            task="Trace this",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

        from agent.runtime_trace import read_runtime_events
        events = read_runtime_events(session_id="trace-parent-session")
        event_names = [event["event"] for event in events]
        assert "assign_agent.requested" in event_names
        assert "assign_agent.resolved" in event_names
        assert "assign_agent.dispatched" in event_names
        assert "assign_agent.completed" in event_names

        resolved = next(event for event in events if event["event"] == "assign_agent.resolved")
        assert resolved["data"]["agent"]["name"] == "trace-cli"
        assert resolved["data"]["routing"]["runner_mode"] == "cli"
        assert resolved["data"]["routing"]["runner_name"] == "trace-runner"

        completed = next(event for event in events if event["event"] == "assign_agent.completed")
        assert completed["data"]["success"] is True
        assert completed["data"]["runner"]["name"] == "trace-runner"
        assert completed["data"]["external_session_id"] == "external-trace-123"

        trace_text = (global_agents_dir.parent / "logs" / "runtime-trace.jsonl").read_text(encoding="utf-8")
        assert "trace-secret-command" not in trace_text
        assert "should-not-appear" not in trace_text

    def test_delegate_agent_emits_runtime_trace_with_effective_toolsets(self, global_agents_dir, monkeypatch):
        (global_agents_dir / "trace-delegate.md").write_text("""---
schema_version: 1
name: trace-delegate
description: Traceable delegate agent
tools:
  mode: restrict
  allow_toolsets: [file]
---

You are a traceable delegate agent.
""")
        mock_parent = MagicMock()
        mock_parent.session_id = "delegate-parent-session"

        monkeypatch.setattr(
            "tools.delegate_tool.delegate_task",
            lambda **kwargs: json.dumps({"success": True, "results": [{"summary": "ok"}]}),
        )

        result = assign_agent(
            agent_name="trace-delegate",
            task="Trace delegate",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )
        parsed = json.loads(result)
        assert parsed["success"] is True

        from agent.runtime_trace import read_runtime_events
        events = read_runtime_events(session_id="delegate-parent-session", agent_name="trace-delegate")
        resolved = next(event for event in events if event["event"] == "assign_agent.resolved")
        assert resolved["data"]["routing"]["runner_mode"] == "delegate_task"
        assert resolved["data"]["toolsets_effective"] == ["file"]

        completed = next(event for event in events if event["event"] == "assign_agent.completed")
        assert completed["data"]["dispatch"] == "delegate_task"
        assert completed["data"]["success"] is True


# ─────────────────────────────────────────────────────────────────────────────
# assign_agent — workdir resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignAgentWorkdirResolution:
    """Tests for workdir resolution from parent_agent attributes."""

    def test_explicit_workdir_takes_precedence(self, global_agents_dir, monkeypatch):
        """Explicit workdir is used even when parent_agent has cwd attributes."""
        make_agent(global_agents_dir / "wd-test.md", "wd-test",
                   description="Workdir test agent")
        mock_parent = MagicMock()
        mock_parent.terminal_cwd = "/should/not/be/used"
        mock_parent.cwd = "/also/not/used"

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="wd-test",
            task="Task",
            parent_agent=mock_parent,
            workdir=str(global_agents_dir),
        )

        # The agent should be found — explicit workdir was used
        assert len(recorded_calls) == 1

    def test_resolves_workdir_from_terminal_cwd_env(self, global_agents_dir, monkeypatch):
        """TERMINAL_CWD env var is used when no explicit workdir is given."""
        make_agent(global_agents_dir / "env-wd.md", "env-wd",
                   description="Env workdir agent")
        mock_parent = MagicMock()

        # Remove cwd attributes so only env var could help
        mock_parent._subdirectory_hints = None
        mock_parent.terminal_cwd = None
        mock_parent.cwd = None

        monkeypatch.setenv("TERMINAL_CWD", str(global_agents_dir))

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="env-wd",
            task="Task",
            parent_agent=mock_parent,
            # no explicit workdir
        )

        # Should succeed because TERMINAL_CWD pointed to global_agents_dir
        assert len(recorded_calls) == 1

    def test_resolves_workdir_from_parent_subdirectory_hints(self, global_agents_dir, monkeypatch):
        """parent_agent._subdirectory_hints.working_dir is used as fallback."""
        make_agent(global_agents_dir / "hint-wd.md", "hint-wd",
                   description="Hint workdir agent")

        # Create a mock subdirectory_hints object
        mock_hints = MagicMock()
        mock_hints.working_dir = str(global_agents_dir)

        mock_parent = MagicMock()
        mock_parent._subdirectory_hints = mock_hints
        mock_parent.terminal_cwd = None
        mock_parent.cwd = None

        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="hint-wd",
            task="Task",
            parent_agent=mock_parent,
            # no explicit workdir, TERMINAL_CWD unset
        )

        assert len(recorded_calls) == 1

    def test_resolves_workdir_from_parent_terminal_cwd(self, global_agents_dir, monkeypatch):
        """parent_agent.terminal_cwd is used when no other hint available."""
        make_agent(global_agents_dir / "term-cwd.md", "term-cwd",
                   description="Terminal cwd agent")
        mock_parent = MagicMock()
        mock_parent._subdirectory_hints = None
        mock_parent.terminal_cwd = str(global_agents_dir)
        mock_parent.cwd = None

        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="term-cwd",
            task="Task",
            parent_agent=mock_parent,
        )

        assert len(recorded_calls) == 1

    def test_resolves_workdir_from_parent_cwd(self, global_agents_dir, monkeypatch):
        """parent_agent.cwd is used as final fallback."""
        make_agent(global_agents_dir / "plain-cwd.md", "plain-cwd",
                   description="Plain cwd agent")
        mock_parent = MagicMock()
        mock_parent._subdirectory_hints = None
        mock_parent.terminal_cwd = None
        mock_parent.cwd = str(global_agents_dir)

        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        recorded_calls = []

        def mock_delegate_task(**kwargs):
            recorded_calls.append(kwargs)
            return json.dumps({"success": True, "results": []})

        monkeypatch.setattr("tools.delegate_tool.delegate_task", mock_delegate_task)

        assign_agent(
            agent_name="plain-cwd",
            task="Task",
            parent_agent=mock_parent,
        )

        assert len(recorded_calls) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignAgentSchema:
    """Tests for ASSIGN_AGENT_SCHEMA structure."""

    def test_schema_required_is_agent_name_and_task_only(self):
        """Schema required fields must be exactly agent_name and task (no parent_agent)."""
        from tools.agents_tool import ASSIGN_AGENT_SCHEMA
        required = ASSIGN_AGENT_SCHEMA["parameters"]["required"]
        assert "agent_name" in required
        assert "task" in required
        assert "parent_agent" not in required

    def test_schema_does_not_have_parent_agent_property(self):
        """Schema must not include parent_agent as a model-facing parameter."""
        from tools.agents_tool import ASSIGN_AGENT_SCHEMA
        properties = ASSIGN_AGENT_SCHEMA["parameters"]["properties"]
        assert "parent_agent" not in properties
