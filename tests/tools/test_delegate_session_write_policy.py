import io
import json
import subprocess
import threading
from types import SimpleNamespace

import pytest

from agent.copilot_acp_client import CopilotACPClient
from agent.session_write_policy import (
    CapabilityGrant,
    SessionWritePolicy,
    SessionWritePolicyMode,
    get_current_session_write_policy,
)
from tools import delegate_tool


class _FakeReadable:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)


class _FakeACPProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = _FakeReadable(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n",
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {"sessionId": "fake-session"},
                    }
                )
                + "\n",
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"text": "fake-acp-response"},
                            }
                        },
                    }
                )
                + "\n",
                json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}) + "\n",
            ]
        )
        self.stderr = _FakeReadable([])

    def poll(self):
        return None

    def terminate(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


class FakeAgent:
    constructed = []
    run_calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeAgent.constructed.append(kwargs)
        self.session_id = "child-session"
        self.session_write_policy = kwargs.get("session_write_policy")
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.api_mode = kwargs.get("api_mode")
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self._session_init_model_config = {}
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.closed = False
        self._acp_client = None
        if kwargs.get("acp_command"):
            self._acp_client = CopilotACPClient(
                acp_command=kwargs["acp_command"],
                acp_args=list(kwargs.get("acp_args") or []),
            )

    def run_conversation(self, user_message, task_id=None, stream_callback=None):
        FakeAgent.run_calls.append(
            {
                "user_message": user_message,
                "task_id": task_id,
                "stream_callback": stream_callback,
            }
        )
        if self._acp_client is not None:
            try:
                response, _reasoning = self._acp_client._run_prompt(
                    user_message,
                    timeout_seconds=1,
                )
            except Exception as exc:
                return {
                    "final_response": "",
                    "completed": False,
                    "api_calls": 0,
                    "messages": [],
                    "error": str(exc),
                }
            return {
                "final_response": response,
                "completed": bool(response),
                "api_calls": 1,
                "messages": [],
            }
        return {
            "final_response": "child done",
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 1, "current_tool": None}

    def close(self):
        if self._acp_client is not None:
            self._acp_client.close()
        self.closed = True


def _reset_fake_agent():
    FakeAgent.constructed = []
    FakeAgent.run_calls = []


def _parent(policy, *, session_policy_attr=True, acp_command=None):
    parent = SimpleNamespace(
        session_id="parent-session",
        enabled_toolsets=["file", "terminal", "delegation"],
        valid_tool_names=[],
        model="parent-model",
        provider="parent-provider",
        base_url="https://parent.example/v1",
        api_mode="chat_completions",
        api_key="parent-key",
        _client_kwargs={"api_key": "parent-key"},
        acp_command=acp_command,
        acp_args=["--acp", "--stdio"] if acp_command else [],
        providers_allowed=["ProviderA"],
        providers_ignored=["ProviderB"],
        providers_order=["ProviderA"],
        provider_sort="throughput",
        provider_require_parameters=True,
        provider_data_collection="deny",
        request_overrides={"temperature": 0},
        openrouter_min_coding_score=42,
        _fallback_chain=[{"provider": "fallback", "model": "fallback-model"}],
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
        _current_turn_id="turn-id",
        _current_task_id="parent-task",
    )
    if session_policy_attr:
        parent.session_write_policy = policy
    else:
        parent.session_write_policy = object()
    return parent


@pytest.fixture(autouse=True)
def fake_agent_and_config(monkeypatch):
    _reset_fake_agent()
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {"max_iterations": 1})
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: None)
    yield
    _reset_fake_agent()


def _build(parent, **kwargs):
    child_model = kwargs.pop("model", None)
    return delegate_tool._build_child_agent(
        task_index=0,
        goal="do work",
        context=None,
        toolsets=None,
        model=child_model,
        max_iterations=1,
        task_count=1,
        parent_agent=parent,
        **kwargs,
    )


def test_build_child_agent_derives_child_policy_once_with_none(monkeypatch):
    parent_policy = SessionWritePolicy.normal("parent")
    calls = []
    original = SessionWritePolicy.derive_child

    def spy(self, requested=None):
        calls.append((self, requested))
        return original(self, requested)

    monkeypatch.setattr(SessionWritePolicy, "derive_child", spy)

    _build(_parent(parent_policy))

    assert calls == [(parent_policy, None)]
    assert FakeAgent.constructed[0]["session_write_policy"] is parent_policy


@pytest.mark.parametrize(
    "parent_policy",
    [
        SessionWritePolicy.normal("normal-parent"),
        SessionWritePolicy.deny_all("deny-parent"),
        SessionWritePolicy(
            session_id="allow-parent",
            mode=SessionWritePolicyMode.ALLOWLIST,
            allowed_roots=("/tmp/hermes-delegate-allow",),
            capability_grants=(
                CapabilityGrant(
                    kind="filesystem",
                    operation="write",
                    roots=("/tmp/hermes-delegate-allow",),
                ),
            ),
            protected=True,
        ),
    ],
)
def test_child_receives_semantically_identical_policy_and_preserves_identity(parent_policy):
    _build(_parent(parent_policy))

    constructed_policy = FakeAgent.constructed[0]["session_write_policy"]
    assert constructed_policy is parent_policy
    assert constructed_policy.mode is parent_policy.mode
    assert constructed_policy.allowed_roots == parent_policy.allowed_roots
    assert constructed_policy.capability_grants == parent_policy.capability_grants
    assert constructed_policy.protected is parent_policy.protected


def test_invalid_parent_policy_falls_back_to_current_policy_and_derives(monkeypatch):
    fallback_policy = SessionWritePolicy.deny_all("fallback-parent")
    calls = []
    original = SessionWritePolicy.derive_child

    def fake_current_session_write_policy(*, session_id, protected):
        assert session_id == "parent-session"
        assert protected is False
        return fallback_policy

    def spy(self, requested=None):
        calls.append((self, requested))
        return original(self, requested)

    monkeypatch.setattr(
        "agent.session_write_policy.get_current_session_write_policy",
        fake_current_session_write_policy,
    )
    monkeypatch.setattr(SessionWritePolicy, "derive_child", spy)

    _build(_parent(object(), session_policy_attr=False))

    assert calls == [(fallback_policy, None)]
    assert FakeAgent.constructed[0]["session_write_policy"] is fallback_policy


def test_default_in_process_delegation_does_not_pre_spawn_or_create_subprocess(monkeypatch):
    def fail_pre_spawn(*args, **kwargs):
        raise AssertionError("delegate_tool must not pre-spawn consult in-process delegation")

    def fail_popen(*args, **kwargs):
        raise AssertionError("delegate_tool must not create subprocesses in-process")

    monkeypatch.setattr("agent.session_write_policy.pre_spawn_consult", fail_pre_spawn)
    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    result = json.loads(
        delegate_tool.delegate_task(
            goal="default in-process run",
            parent_agent=_parent(SessionWritePolicy.normal("parent")),
        )
    )

    assert result["results"][0]["status"] == "completed"
    assert FakeAgent.run_calls[0]["user_message"] == "default in-process run"


def test_acp_command_and_args_forward_without_execution_or_pre_spawn(monkeypatch):
    def fail_pre_spawn(*args, **kwargs):
        raise AssertionError("delegate_tool must not pre-spawn consult ACP forwarding")

    def fail_popen(*args, **kwargs):
        raise AssertionError("delegate_tool must not execute ACP command")

    monkeypatch.setattr("agent.session_write_policy.pre_spawn_consult", fail_pre_spawn)
    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")

    _build(
        _parent(SessionWritePolicy.normal("parent")),
        override_acp_command="copilot",
        override_acp_args=["--acp", "--stdio"],
    )

    constructed = FakeAgent.constructed[0]
    assert constructed["provider"] == "copilot-acp"
    assert constructed["acp_command"] == "copilot"
    assert constructed["acp_args"] == ["--acp", "--stdio"]


def test_child_execution_binds_policy_through_real_acp_path(monkeypatch, tmp_path):
    env_builds: list[str] = []
    popen_calls: list[dict] = []

    monkeypatch.setattr(
        "agent.copilot_acp_client._build_subprocess_env",
        lambda: env_builds.append("env") or {"SAFE_ENV": "1"},
    )
    monkeypatch.setattr(
        "agent.copilot_acp_client.subprocess.Popen",
        lambda argv, **kwargs: popen_calls.append({"argv": argv, "kwargs": kwargs})
        or _FakeACPProcess(),
    )
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    parent = _parent(
        SessionWritePolicy.deny_all("parent-deny"),
        acp_command="fake-copilot",
    )

    child = _build(
        parent,
        override_acp_command="fake-copilot",
        override_acp_args=["--acp", "--stdio"],
    )
    assert child.session_write_policy is parent.session_write_policy

    result = delegate_tool._run_single_child(
        0,
        "deny child",
        child,
        parent,
    )

    assert result["status"] == "failed"
    assert "acp_subprocess_blocked_by_session_write_policy" in result["error"]
    assert env_builds == []
    assert popen_calls == []
    assert get_current_session_write_policy(session_id="after-child").mode is SessionWritePolicyMode.NORMAL


def test_child_execution_binds_normal_policy_and_restores_external_context(monkeypatch, tmp_path):
    env_builds: list[str] = []
    popen_calls: list[dict] = []

    monkeypatch.setattr(
        "agent.copilot_acp_client._build_subprocess_env",
        lambda: env_builds.append("env") or {"SAFE_ENV": "1"},
    )
    monkeypatch.setattr(
        "agent.copilot_acp_client.subprocess.Popen",
        lambda argv, **kwargs: popen_calls.append({"argv": argv, "kwargs": kwargs})
        or _FakeACPProcess(),
    )
    monkeypatch.setattr("agent.copilot_acp_client._acp_supported", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("shutil.which", lambda cmd: f"/usr/bin/{cmd}")
    parent = _parent(
        SessionWritePolicy.normal("parent-normal"),
        acp_command="fake-copilot",
    )
    child = _build(
        parent,
        override_acp_command="fake-copilot",
        override_acp_args=["--acp", "--stdio"],
    )
    sentinel = SessionWritePolicy.deny_all("external-sentinel")

    from agent.session_write_policy import session_write_policy_scope

    with session_write_policy_scope(sentinel):
        result = delegate_tool._run_single_child(
            0,
            "normal child",
            child,
            parent,
        )
        assert get_current_session_write_policy(session_id="inside-parent").mode is SessionWritePolicyMode.DENY_ALL

    assert result["status"] == "completed"
    assert env_builds == ["env"]
    assert len(popen_calls) == 1
    assert get_current_session_write_policy(session_id="after-child").mode is SessionWritePolicyMode.NORMAL


def test_child_policy_context_restores_after_child_exception(monkeypatch):
    parent = _parent(SessionWritePolicy.deny_all("parent-deny"))
    child = _build(parent)
    sentinel = SessionWritePolicy.normal("external-sentinel")

    def boom(**_kwargs):
        assert get_current_session_write_policy(session_id="child").mode is SessionWritePolicyMode.DENY_ALL
        raise RuntimeError("child failure")

    child.run_conversation = boom
    from agent.session_write_policy import session_write_policy_scope

    with session_write_policy_scope(sentinel):
        result = delegate_tool._run_single_child(0, "raises", child=child, parent_agent=parent)
        assert get_current_session_write_policy(session_id="restored").mode is SessionWritePolicyMode.NORMAL

    assert result["status"] == "error"
    assert get_current_session_write_policy(session_id="after-exception").mode is SessionWritePolicyMode.NORMAL
