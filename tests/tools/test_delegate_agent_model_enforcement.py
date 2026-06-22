import json
import threading
from unittest.mock import MagicMock, patch

from agent.agent_model_resolution import resolve_agent_model, resolve_job_model
from agent.handoff_telemetry import build_handoff_telemetry_event
from tools.delegate_tool import _build_child_agent, delegate_task


def _write_config(home):
    (home / "config.yaml").write_text(
        """\
model:
  default: gpt-default
  provider: openrouter
agents:
  models:
    devin:
      model: gpt-5.3-codex
      provider: openai-codex
      fallbacks:
        - provider: openrouter
          model: backup-model
    alice:
      model: gpt-4.1
      provider: openai
delegation:
  max_iterations: 50
  max_spawn_depth: 2
""".strip(),
        encoding="utf-8",
    )


def _parent():
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "parent-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "parent-model"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent._fallback_chain = [{"provider": "openrouter", "model": "parent-fallback"}]
    parent.enabled_toolsets = ["terminal", "file"]
    return parent


class FakeAIAgent:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.api_mode = kwargs.get("api_mode")
        self.base_url = kwargs.get("base_url")
        self.platform = kwargs.get("platform")
        self.enabled_toolsets = kwargs.get("enabled_toolsets")
        self._session_init_model_config = {}
        FakeAIAgent.instances.append(self)

    def run_conversation(self, user_message, task_id):
        return {"final_response": "ok", "completed": True, "api_calls": 1, "messages": []}

    def close(self):
        pass


def test_build_child_agent_uses_agents_models_devin(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)
    FakeAIAgent.instances = []

    with patch("run_agent.AIAgent", FakeAIAgent), patch(
        "tools.delegate_tool._resolve_child_credential_pool", return_value=None
    ):
        child = _build_child_agent(
            task_index=0,
            goal="do code",
            context=None,
            toolsets=["terminal", "file"],
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=_parent(),
            agent_id="DeViN",
        )

    assert child.model == "gpt-5.3-codex"
    assert child.provider == "openai-codex"
    assert child.kwargs["fallback_model"] == [{"provider": "openrouter", "model": "backup-model"}]
    assert child._agent_id == "devin"
    assert child._assigned_model == "gpt-5.3-codex"
    assert child._assigned_provider == "openai-codex"
    assert child._model_source == "agents.models.devin"
    assert child._session_init_model_config["effective_model"] == "gpt-5.3-codex"


def test_delegate_task_accepts_agent_id_and_preserves_legacy_without_it(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)
    FakeAIAgent.instances = []

    with patch("run_agent.AIAgent", FakeAIAgent), patch(
        "tools.delegate_tool._resolve_child_credential_pool", return_value=None
    ):
        result = json.loads(delegate_task(goal="do code", agent_id="devin", parent_agent=_parent()))
        legacy = json.loads(delegate_task(goal="do code", parent_agent=_parent()))

    assert result["results"][0]["status"] == "completed"
    assert FakeAIAgent.instances[0].model == "gpt-5.3-codex"
    assert FakeAIAgent.instances[0].provider == "openai-codex"
    assert legacy["results"][0]["status"] == "completed"
    assert FakeAIAgent.instances[1].model == "gpt-5.5" or FakeAIAgent.instances[1].model == "parent-model"


# ── B1: delegate_task sin agent_id → comportamiento legacy intacto ────────────


def test_build_child_agent_without_agent_id_uses_parent_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)
    FakeAIAgent.instances = []

    with patch("run_agent.AIAgent", FakeAIAgent), patch(
        "tools.delegate_tool._resolve_child_credential_pool", return_value=None
    ):
        child = _build_child_agent(
            task_index=0,
            goal="do code",
            context=None,
            toolsets=["terminal", "file"],
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=_parent(),
            agent_id=None,
        )

    # Without agent_id, the child should fall back to the parent's model/provider
    # (delegation.config_or_parent via resolve_agent_model with None agent_id)
    assert child.model == "parent-model"
    assert child.provider == "openrouter"
    assert child._model_source == "delegation.config_or_parent"
    assert child._agent_id is None
    assert child._assigned_model is None
    assert child._assigned_provider is None


# ── B2: delegate_task con agent_id inexistente → warnings ────────────────────


def test_build_child_agent_with_phantom_agent_id_falls_back_with_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)
    FakeAIAgent.instances = []

    with patch("run_agent.AIAgent", FakeAIAgent), patch(
        "tools.delegate_tool._resolve_child_credential_pool", return_value=None
    ):
        child = _build_child_agent(
            task_index=0,
            goal="do code",
            context=None,
            toolsets=["terminal", "file"],
            model=None,
            max_iterations=50,
            task_count=1,
            parent_agent=_parent(),
            agent_id="phantom",
        )

    # phantom not in agents.models → fallback to delegation.config_or_parent
    assert child.model == "parent-model"
    assert child.provider == "openrouter"
    assert child._model_source == "delegation.config_or_parent"
    assert child._agent_id == "phantom"
    assert len(child._model_resolution_warnings) >= 1
    assert any("phantom" in w for w in child._model_resolution_warnings)


# ── B3: resolve_agent_model("DEVIN") → case-insensitive ──────────────────────


def test_resolve_agent_model_case_insensitive(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)

    result = resolve_agent_model("DEVIN")
    assert result.assigned_model == "gpt-5.3-codex"
    assert result.assigned_provider == "openai-codex"
    assert result.effective_model == "gpt-5.3-codex"
    assert result.effective_provider == "openai-codex"
    assert result.model_source == "agents.models.devin"
    assert result.agent_id == "devin"


# ── B4: resolve_agent_model(None) → fallback sin crash ──────────────────────


def test_resolve_agent_model_none_returns_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)

    result = resolve_agent_model(
        None,
        fallback_model="fallback-model",
        fallback_provider="fallback-provider",
    )
    assert result.agent_id is None
    assert result.effective_model == "fallback-model"
    assert result.effective_provider == "fallback-provider"
    assert result.model_source == "fallback"
    assert len(result.warnings) >= 1
    assert "no agent_id provided" in result.warnings[0]


# ── B5: resolve_job_model con skill alice ────────────────────────────────────


def test_resolve_job_model_skill_alice_via_matrix(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)

    result = resolve_job_model(
        {"skill": "alice", "model": None, "provider": None},
        default_model="gpt-default",
        default_provider="openrouter",
    )
    assert result.agent_id == "alice"
    assert result.effective_model == "gpt-4.1"
    assert result.effective_provider == "openai"
    assert result.model_source == "agents.models.alice"


# ── B6: resolve_job_model con model explicito → precedence ────────────────────


def test_resolve_job_model_explicit_model_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)

    result = resolve_job_model(
        {"skill": "alice", "model": "explicit-model", "provider": None},
        default_model="gpt-default",
        default_provider="openrouter",
    )
    # Explicit model wins over matrix
    assert result.effective_model == "explicit-model"
    assert result.model_source == "job.model"
    # Provider falls back to default since job.provider is not set
    assert result.effective_provider == "openrouter"


# ── B7: resolve_job_model con no_agent → no fuerza AIAgent ──────────────────


def test_resolve_job_model_no_agent_resolves_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path)

    result = resolve_job_model(
        {"skill": "alice", "model": None, "provider": None, "no_agent": True},
        default_model="gpt-default",
        default_provider="openrouter",
    )
    # no_agent just means the scheduler skips AIAgent creation; the
    # resolution still runs and returns matrix values for telemetry
    assert result.agent_id == "alice"
    assert result.effective_model == "gpt-4.1"
    assert result.effective_provider == "openai"
    assert result.model_source == "agents.models.alice"


# ── B8: build_handoff_telemetry_event con nuevos campos ──────────────────────


def test_build_handoff_telemetry_event_includes_enforcement_fields():
    event = build_handoff_telemetry_event(
        trace_id="handoff-test-1234",
        subagent_id="sa-0-abcdef",
        parent_session_id="sess-1",
        parent_task_id="task-1",
        parent_subagent_id=None,
        task_index=0,
        status="completed",
        exit_reason="done",
        model="gpt-5.3-codex",
        provider="openai-codex",
        agent_id="devin",
        assigned_model="gpt-5.3-codex",
        assigned_provider="openai-codex",
        effective_model="gpt-5.3-codex",
        effective_provider="openai-codex",
        model_source="agents.models.devin",
        model_resolution_warnings=[],
        api_mode="codex_responses",
        api_calls=5,
        duration_seconds=10.5,
    )
    assert event["agent_id"] == "devin"
    assert event["assigned_model"] == "gpt-5.3-codex"
    assert event["assigned_provider"] == "openai-codex"
    assert event["effective_model"] == "gpt-5.3-codex"
    assert event["effective_provider"] == "openai-codex"
    assert event["model_source"] == "agents.models.devin"
    assert event["model_resolution_warnings"] == []


# ── C1: build_handoff_telemetry_event backward compatible ────────────────────


def test_build_handoff_telemetry_event_backward_compatible():
    """Call with only the original (pre-enforcement) kwargs → must not crash."""
    event = build_handoff_telemetry_event(
        trace_id="handoff-legacy-5678",
        subagent_id="sa-0-legacy",
        parent_session_id="sess-1",
        parent_task_id="task-1",
        parent_subagent_id=None,
        task_index=0,
        status="completed",
        exit_reason="done",
        model="gpt-default",
        provider="openrouter",
        api_mode="chat_completions",
        api_calls=3,
        duration_seconds=5.0,
    )
    # New fields should be None (backward compat)
    assert event["agent_id"] is None
    assert event["assigned_model"] is None
    assert event["assigned_provider"] is None
    assert event["model_source"] is None
    assert event["model_resolution_warnings"] == []
    # Core fields still present
    assert event["model"] == "gpt-default"
    assert event["provider"] == "openrouter"
    assert event["api_calls"] == 3
