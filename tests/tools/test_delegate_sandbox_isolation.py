"""Per-subagent terminal sandbox isolation (delegate_task(sandbox=True)).

Feature issue #4271: parallel subagents share the parent's single terminal
backend — concurrent `cd`, env mutations, and writes to the same path
collide. This pins the opt-in fix:

* `delegate_task(sandbox=True)` (top-level or per-task) gives each sandboxed
  child its own container via `register_task_env_overrides()`; the child's
  task_id no longer collapses to the parent's container key.
* Non-sandboxed children keep the documented shared-parent-container
  contract (alias registration unchanged).
* local/ssh/vercel_sandbox backends cannot isolate: `sandbox=True` fails
  loudly (tool_error at entry, ValueError at spawn) instead of silently
  degrading to the shared sandbox.
* Overrides are cleared at child teardown so the registry cannot leak.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools import delegate_tool, terminal_tool
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _clear_child_sandbox,
    _register_child_sandbox_overrides,
    _seed_child_terminal_env,
    delegate_task,
)

_IMAGE = "nikolaik/python-nodejs:python3.11-nodejs20"


def _make_mock_parent(depth=0):
    """Mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
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
    return parent


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Reset the module-level registries and pin the config→env bridge."""
    before_overrides = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    with terminal_tool._container_alias_lock:
        before_aliases = dict(terminal_tool._container_aliases)
        terminal_tool._container_aliases.clear()
    with terminal_tool._session_cwd_lock:
        before_cwd = dict(terminal_tool._session_cwd)
        terminal_tool._session_cwd.clear()
    # The config→env bridge is one-shot; mark it done so tests control env vars.
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before_overrides)
    with terminal_tool._container_alias_lock:
        terminal_tool._container_aliases.clear()
        terminal_tool._container_aliases.update(before_aliases)
    with terminal_tool._session_cwd_lock:
        terminal_tool._session_cwd.clear()
        terminal_tool._session_cwd.update(before_cwd)


def _set_backend(monkeypatch, env_type: str, image: str = _IMAGE):
    monkeypatch.setenv("TERMINAL_ENV", env_type)
    monkeypatch.setenv(f"TERMINAL_{env_type.upper()}_IMAGE", image)


class TestSandboxSchema:
    """The model-facing schema exposes the opt-in flag."""

    def test_top_level_sandbox_property(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "sandbox" in props
        assert props["sandbox"]["type"] == "boolean"

    def test_per_task_sandbox_property(self):
        items = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        props = items["properties"]
        assert "sandbox" in props
        assert props["sandbox"]["type"] == "boolean"


class TestRegisterChildSandboxOverrides:
    """_register_child_sandbox_overrides registers isolation triggers."""

    def test_docker_registers_env_type_and_image(self, monkeypatch):
        _set_backend(monkeypatch, "docker", _IMAGE)
        _register_child_sandbox_overrides("subagent-1")
        assert terminal_tool._task_env_overrides["subagent-1"] == {
            "env_type": "docker",
            "docker_image": _IMAGE,
        }

    def test_modal_registers_modal_image(self, monkeypatch):
        _set_backend(monkeypatch, "modal", "hermes-modal")
        _register_child_sandbox_overrides("subagent-1")
        assert terminal_tool._task_env_overrides["subagent-1"] == {
            "env_type": "modal",
            "modal_image": "hermes-modal",
        }

    def test_daytona_registers_daytona_image(self, monkeypatch):
        _set_backend(monkeypatch, "daytona", "hermes-daytona")
        _register_child_sandbox_overrides("subagent-1")
        assert terminal_tool._task_env_overrides["subagent-1"] == {
            "env_type": "daytona",
            "daytona_image": "hermes-daytona",
        }

    def test_singularity_registers_singularity_image(self, monkeypatch):
        _set_backend(monkeypatch, "singularity", "docker://hermes-singularity")
        _register_child_sandbox_overrides("subagent-1")
        assert terminal_tool._task_env_overrides["subagent-1"] == {
            "env_type": "singularity",
            "singularity_image": "docker://hermes-singularity",
        }

    def test_local_backend_raises(self, monkeypatch):
        _set_backend(monkeypatch, "local")
        with pytest.raises(ValueError, match="container terminal backend"):
            _register_child_sandbox_overrides("subagent-1")

    def test_ssh_backend_raises(self, monkeypatch):
        _set_backend(monkeypatch, "ssh")
        with pytest.raises(ValueError, match="container terminal backend"):
            _register_child_sandbox_overrides("subagent-1")

    def test_vercel_sandbox_raises(self, monkeypatch):
        _set_backend(monkeypatch, "vercel_sandbox")
        with pytest.raises(ValueError, match="container terminal backend"):
            _register_child_sandbox_overrides("subagent-1")

    def test_override_triggers_isolation_keying(self, monkeypatch):
        """A registered override makes _resolve_container_task_id return the
        child's own task_id (its own container) instead of the parent key."""
        _set_backend(monkeypatch, "docker", _IMAGE)
        _register_child_sandbox_overrides("subagent-1")
        assert terminal_tool._resolve_container_task_id("subagent-1") == "subagent-1"
        assert terminal_tool.resolve_task_overrides("subagent-1")["docker_image"] == _IMAGE

    def test_parent_task_image_wins_over_process_config(self, monkeypatch):
        """A per-task image registered on the parent session (RL rollouts,
        ACP workspaces) is inherited by the sandboxed child."""
        _set_backend(monkeypatch, "docker", _IMAGE)
        terminal_tool.register_task_env_overrides(
            "tui:sess-a", {"docker_image": "custom/parent-image:latest"}
        )
        _register_child_sandbox_overrides("subagent-1", "tui:sess-a")
        assert terminal_tool._task_env_overrides["subagent-1"]["docker_image"] == (
            "custom/parent-image:latest"
        )

    def test_no_parent_task_falls_back_to_process_config(self, monkeypatch):
        _set_backend(monkeypatch, "docker", _IMAGE)
        _register_child_sandbox_overrides("subagent-1", None)
        assert terminal_tool._task_env_overrides["subagent-1"]["docker_image"] == _IMAGE


class TestSeedChildTerminalEnv:
    """Spawn wiring: sandbox children get overrides, others get the alias."""

    def test_shared_path_registers_alias_and_seeds_cwd(self, monkeypatch):
        calls = {"alias": 0, "sandbox": 0}
        monkeypatch.setattr(
            terminal_tool, "register_container_alias",
            lambda *a, **k: calls.__setitem__("alias", calls["alias"] + 1),
        )
        monkeypatch.setattr(
            terminal_tool, "get_session_cwd", lambda task_id: "/parent/dir",
        )
        monkeypatch.setattr(
            terminal_tool, "record_session_cwd", lambda *a, **k: None,
        )
        monkeypatch.setattr(
            delegate_tool, "_register_child_sandbox_overrides",
            lambda *a, **k: calls.__setitem__("sandbox", calls["sandbox"] + 1),
        )

        _seed_child_terminal_env("subagent-1", "tui:sess-a", sandbox=False)

        assert calls["alias"] == 1
        assert calls["sandbox"] == 0
        assert terminal_tool.get_session_cwd("subagent-1") == "/parent/dir"

    def test_sandbox_path_registers_overrides_not_alias(self, monkeypatch):
        calls = {"alias": 0, "sandbox": 0}
        monkeypatch.setattr(
            terminal_tool, "register_container_alias",
            lambda *a, **k: calls.__setitem__("alias", calls["alias"] + 1),
        )
        monkeypatch.setattr(
            terminal_tool, "get_session_cwd", lambda task_id: None,
        )
        monkeypatch.setattr(
            terminal_tool, "record_session_cwd", lambda *a, **k: None,
        )
        monkeypatch.setattr(
            delegate_tool, "_register_child_sandbox_overrides",
            lambda *a, **k: calls.__setitem__("sandbox", calls["sandbox"] + 1),
        )

        _seed_child_terminal_env("subagent-1", "tui:sess-a", sandbox=True)

        assert calls["sandbox"] == 1
        assert calls["alias"] == 0


class TestDelegateTaskEntryValidation:
    """sandbox=True fails fast on backends that cannot isolate."""

    def test_local_backend_tool_error(self, monkeypatch):
        _set_backend(monkeypatch, "local")
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="do the thing", sandbox=True, parent_agent=parent))
        assert "requires a container terminal backend" in result["error"]
        assert "local" in result["error"]

    def test_string_false_per_task_is_not_sandboxed(self, monkeypatch):
        """A model-emitted string 'false' must not coerce to True (bool('false')
        would); the shared truthy parser is the single owner of coercion."""
        _set_backend(monkeypatch, "local")
        parent = _make_mock_parent()
        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child
            result = json.loads(
                delegate_task(
                    tasks=[
                        {"goal": "First task with a longer self-contained goal", "sandbox": "false"},
                        {"goal": "Second task with a longer self-contained goal"},
                    ],
                    parent_agent=parent,
                )
            )
        # sandbox="false" is not a sandbox request -> entry validation must
        # NOT reject on backend; the children run normally.
        assert "error" not in result

    def test_ssh_backend_tool_error(self, monkeypatch):
        _set_backend(monkeypatch, "ssh")
        parent = _make_mock_parent()
        result = json.loads(delegate_task(goal="do the thing", sandbox=True, parent_agent=parent))
        assert "requires a container terminal backend" in result["error"]

    def test_per_task_sandbox_checked_too(self, monkeypatch):
        _set_backend(monkeypatch, "local")
        parent = _make_mock_parent()
        result = json.loads(
            delegate_task(
                tasks=[{"goal": "a"}, {"goal": "b", "sandbox": True}],
                parent_agent=parent,
            )
        )
        assert "requires a container terminal backend" in result["error"]

    def test_docker_backend_passes_validation(self, monkeypatch):
        """Validation passes on docker; the child runs via the mocked AIAgent."""
        _set_backend(monkeypatch, "docker", _IMAGE)
        parent = _make_mock_parent()
        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child
            result = json.loads(delegate_task(goal="sandboxed", sandbox=True, parent_agent=parent))
        assert "error" not in result


class TestClearChildSandbox:
    def test_clears_overrides_and_alias(self, monkeypatch):
        _set_backend(monkeypatch, "docker", _IMAGE)
        _register_child_sandbox_overrides("subagent-1")
        terminal_tool.register_container_alias("subagent-1", "tui:sess-a")
        assert "subagent-1" in terminal_tool._task_env_overrides

        _clear_child_sandbox("subagent-1")

        assert "subagent-1" not in terminal_tool._task_env_overrides
        assert "subagent-1" not in terminal_tool._container_aliases

    def test_noop_on_empty_task_id(self):
        _clear_child_sandbox(None)
        _clear_child_sandbox("")


class TestDispatchForwarding:
    """sandbox reaches delegate_task through the live model dispatch path."""

    def test_dispatch_forwards_sandbox(self):
        import run_agent

        captured = {}

        def fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return "{}"

        parent = _make_mock_parent()
        with patch("tools.delegate_tool.delegate_task", fake_delegate_task):
            run_agent.AIAgent._dispatch_delegate_task(
                parent,
                {"goal": "test", "sandbox": True, "tasks": [{"goal": "n", "sandbox": False}]},
            )

        assert captured["sandbox"] is True
        assert captured["tasks"][0]["sandbox"] is False
