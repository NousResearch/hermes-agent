from __future__ import annotations

import logging
from concurrent.futures import TimeoutError as FuturesTimeoutError
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


class _FakeChild:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        fallback_chain: list[dict[str, Any]] | None = None,
        api_calls: int = 0,
        progress_events: list[tuple[str, dict[str, Any]]] | None = None,
    ):
        self.provider = provider
        self.model = model
        self.base_url = f"https://{provider}.example.test/v1"
        self.api_mode = "chat_completions"
        self._fallback_chain = list(fallback_chain or [])
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._subagent_id = f"sa-{provider}-{model}".replace("/", "-")
        self._parent_subagent_id = None
        self.session_id = f"sess-{provider}-{model}".replace("/", "-")
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.max_iterations = 10
        self._api_calls = api_calls
        self.interrupted = False
        self.closed = False
        self.valid_tool_names = set()
        self.tools = []
        self.enabled_toolsets = ["file"]
        self._delegate_saved_tool_names = []
        self.tool_progress_callback = self._progress
        self._progress_events = progress_events if progress_events is not None else []

    def _progress(self, event: str, **kwargs):
        self._progress_events.append((event, kwargs))

    def get_activity_summary(self):
        return {
            "api_call_count": self._api_calls,
            "max_iterations": self.max_iterations,
            "current_tool": None,
            "seconds_since_activity": 0,
        }

    def interrupt(self):
        self.interrupted = True

    def close(self):
        self.closed = True


class _FakeFuture:
    def __init__(self, outcome, timeout_args):
        self._outcome = outcome
        self._timeout_args = timeout_args

    def result(self, timeout=None):
        self._timeout_args.append(timeout)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _install_fake_executor(monkeypatch, outcomes, timeout_args):
    from tools import delegate_tool

    class _FakeExecutor:
        def __init__(self, *args, **kwargs):
            self.shutdown_calls = []

        def submit(self, fn):
            assert outcomes, "test exhausted fake executor outcomes"
            return _FakeFuture(outcomes.pop(0), timeout_args)

        def shutdown(self, wait=False):
            self.shutdown_calls.append(wait)

    monkeypatch.setattr(delegate_tool, "ThreadPoolExecutor", _FakeExecutor)


def _parent():
    return SimpleNamespace(
        _current_task_id=None,
        _current_turn_id="turn-test",
        _touch_activity=lambda desc: None,
        _active_children=[],
        _active_children_lock=None,
        session_id="parent-session",
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )


def test_child_timeout_fails_over_to_second_provider(monkeypatch, caplog):
    from tools import delegate_tool

    progress_events: list[tuple[str, dict[str, Any]]] = []
    outcomes = [
        FuturesTimeoutError(),
        {
            "final_response": "done on fallback",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        },
    ]
    timeout_args = []
    _install_fake_executor(monkeypatch, outcomes, timeout_args)

    fallback_chain = [{"provider": "openai-codex", "model": "gpt-5.5"}]
    first = _FakeChild(
        provider="bedrock",
        model="us.anthropic.claude-opus-4-8",
        fallback_chain=fallback_chain,
        progress_events=progress_events,
    )
    built_routes = []

    def child_builder(route):
        built_routes.append(route)
        return _FakeChild(
            provider=route["provider"],
            model=route["model"],
            fallback_chain=fallback_chain,
            api_calls=1,
            progress_events=progress_events,
        )

    caplog.set_level(logging.WARNING, logger="tools.delegate_tool")

    result = delegate_tool._run_single_child(
        task_index=0,
        goal="do slow task",
        child=first,
        parent_agent=_parent(),
        child_timeout=30.0,
        child_builder=child_builder,
    )

    assert result["status"] == "completed"
    assert result["summary"] == "done on fallback"
    assert result["providers_tried"] == [
        "bedrock/us.anthropic.claude-opus-4-8",
        "openai-codex/gpt-5.5",
    ]
    assert result["attempt_count"] == 2
    assert first.interrupted is True
    assert built_routes == [{"provider": "openai-codex", "model": "gpt-5.5", "base_url": "", "api_key": "", "api_mode": "", "key_env": ""}]
    assert timeout_args == [30.0, pytest.approx(30.0)]

    assert (
        "Subagent 0 timed out on bedrock/us.anthropic.claude-opus-4-8"
        in caplog.text
    )
    assert "failing over to openai-codex/gpt-5.5 (attempt 2/2)" in caplog.text

    failover_events = [e for e in progress_events if e[0] == "subagent.failover"]
    assert len(failover_events) == 1
    assert failover_events[0][1]["to_provider"] == "openai-codex"
    assert failover_events[0][1]["to_model"] == "gpt-5.5"


def test_all_child_timeout_attempts_return_providers_tried_and_budget(monkeypatch):
    from tools import delegate_tool

    outcomes = [FuturesTimeoutError(), FuturesTimeoutError(), FuturesTimeoutError()]
    timeout_args = []
    _install_fake_executor(monkeypatch, outcomes, timeout_args)

    fallback_chain = [
        {"provider": "bedrock", "model": "us.anthropic.claude-opus-4-7"},
        {"provider": "openai-codex", "model": "gpt-5.5"},
    ]
    first = _FakeChild(
        provider="bedrock",
        model="us.anthropic.claude-opus-4-8",
        fallback_chain=fallback_chain,
        api_calls=2,
    )

    def child_builder(route):
        return _FakeChild(
            provider=route["provider"],
            model=route["model"],
            fallback_chain=fallback_chain,
            api_calls=2,
        )

    result = delegate_tool._run_single_child(
        task_index=0,
        goal="do slow task",
        child=first,
        parent_agent=_parent(),
        child_timeout=30.0,
        child_builder=child_builder,
    )

    assert result["status"] == "timeout"
    assert result["exit_reason"] == "timeout"
    assert result["providers_tried"] == [
        "bedrock/us.anthropic.claude-opus-4-8",
        "bedrock/us.anthropic.claude-opus-4-7",
        "openai-codex/gpt-5.5",
    ]
    assert result["attempt_count"] == 3
    assert result["timeout_budget_seconds"] == 90.0
    assert (
        "Timed out on bedrock/us.anthropic.claude-opus-4-8, "
        "bedrock/us.anthropic.claude-opus-4-7, openai-codex/gpt-5.5 "
        "(3 attempts)."
    ) in result["error"]
    assert len(timeout_args) == 3
    assert all(arg <= 30.0 for arg in timeout_args)


def test_non_timeout_exception_does_not_fail_over(monkeypatch):
    from tools import delegate_tool

    outcomes = [RuntimeError("boom")]
    timeout_args = []
    _install_fake_executor(monkeypatch, outcomes, timeout_args)

    first = _FakeChild(
        provider="bedrock",
        model="us.anthropic.claude-opus-4-8",
        fallback_chain=[{"provider": "openai-codex", "model": "gpt-5.5"}],
    )
    builder = MagicMock()

    result = delegate_tool._run_single_child(
        task_index=0,
        goal="do task",
        child=first,
        parent_agent=_parent(),
        child_timeout=30.0,
        child_builder=builder,
    )

    assert result["status"] == "error"
    assert result["exit_reason"] == "error"
    assert result["error"] == "boom"
    assert result["providers_tried"] == ["bedrock/us.anthropic.claude-opus-4-8"]
    builder.assert_not_called()
