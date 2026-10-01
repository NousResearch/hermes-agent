"""Focused switch/fallback regression proof for an already-open TUI session (#28).

The provider fakes are loopback-only, while the agent loop, turn-admission fallback sync,
error classifier, fallback activation, tool round, and usage attribution are production code.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

from agent.error_classifier import FailoverReason
from run_agent import AIAgent
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall
from tests.fakes.providers.chat_variants import CError, FakeChatVariantServer
from tui_gateway import server

PRIMARY_MODEL = "issue-28-primary"
FALLBACK_MODEL = "issue-28-fallback"
STALE_MODEL = "issue-28-stale"
TOOL_NAME = "issue_28_probe"
TOOL_RESULT = "issue-28-tool-result"
FINAL_RESPONSE = "issue-28-fallback-complete"
PROMPT = "Run the issue 28 fallback probe."
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return a deterministic regression-test canary.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}


class _StopAfterFallbackSync(Exception):
    pass


def _admit_turn(monkeypatch, config_path, agent) -> None:
    """Drive the production TUI turn-admission path through fallback sync."""
    session = {
        "agent": agent,
        "session_key": "issue-28-live-session",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
    }
    monkeypatch.setattr(server, "_active_config_path", lambda: config_path)
    monkeypatch.setattr(server, "_profile_runtime_scope_tokens", lambda _home: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    for name in (
        "_apply_pending_model_switch",
        "_sync_agent_model_with_config",
        "_sync_agent_compression_with_config",
    ):
        monkeypatch.setattr(server, name, lambda _sid, _session: None)

    def stop_after_sync(_sid, _session):
        raise _StopAfterFallbackSync()

    monkeypatch.setattr(server, "_sync_bot_capabilities", stop_after_sync)
    state = server._TurnRun(
        agent=None,
        one_turn_restore=None,
        terminal_callback=None,
        receipt_committed=False,
    )
    with pytest.raises(_StopAfterFallbackSync):
        server._prepare_turn_input("issue-28-ui", session, state, PROMPT, [])


def _build_agent(primary_url: str, fallback_chain: list[dict], statuses: list[tuple[str, str]]) -> AIAgent:
    with (
        patch("model_tools.get_tool_definitions", return_value=[TOOL_DEFINITION]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            api_key="issue-28-primary-key",
            base_url=primary_url,
            provider="openai",
            model=PRIMARY_MODEL,
            api_mode="chat_completions",
            max_iterations=4,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_chain,
            status_callback=lambda kind, message: statuses.append((kind, str(message))),
        )
    # A usage-limit wall is non-retryable, but keep every recovery ladder bounded if classification regresses.
    agent._api_max_retries = 1
    agent._auto_recovery_cycles = 0
    return agent


def _replacement_chain(primary_url: str, fallback_url: str) -> list[dict]:
    return [
        {
            "provider": "openai",
            "model": PRIMARY_MODEL,
            "base_url": primary_url,
        },
        {
            "provider": "custom",
            "model": FALLBACK_MODEL,
            "base_url": fallback_url,
            "key_env": "ISSUE28_FALLBACK_KEY",
        },
    ]


def _write_chain(config_path, chain: list[dict]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump({"fallback_providers": chain}), encoding="utf-8")


def _exercise(monkeypatch, root, *, already_live: bool) -> dict:
    statuses: list[tuple[str, str]] = []
    activations: list[dict] = []
    tool_calls: list[tuple[str, dict]] = []
    primary_error = CError(
        429,
        "Your account has reached its usage limit.",
        code="usage_limit_reached",
    )
    with (
        FakeChatVariantServer([primary_error]) as primary,
        FakeLLMServer(default_text="stale fallback must not run") as stale,
        FakeLLMServer(
            [
                ToolCall(TOOL_NAME, {"value": "issue-28"}),
                Text(FINAL_RESPONSE),
            ]
        ) as fallback,
    ):
        replacement = _replacement_chain(primary.base_url, fallback.base_url)
        config_path = root / "config.yaml"
        _write_chain(config_path, replacement)

        if already_live:
            original = [{
                "provider": "custom",
                "model": STALE_MODEL,
                "base_url": stale.base_url,
                "key_env": "ISSUE28_FALLBACK_KEY",
            }]
            agent = _build_agent(primary.base_url, original, statuses)
            agent._fallback_index = len(original)
            # Seed a stale unavailability memo in the production 3-tuple key shape
            # (_fallback_entry_key). The admission sync must replace the chain and
            # clear this memo because the configured chain changed.
            agent._unavailable_fallback_keys = {("custom", STALE_MODEL, stale.base_url.rstrip("/"))}
            _admit_turn(monkeypatch, config_path, agent)
            index_after_admission = agent._fallback_index
            chain_after_admission = list(agent._fallback_chain)
        else:
            from hermes_cli.config_effective import load_user_config_effective
            from hermes_cli.fallback_config import get_fallback_chain

            persisted = get_fallback_chain(
                load_user_config_effective(config_path, fail_closed=True)
            )
            agent = _build_agent(primary.base_url, persisted, statuses)
            index_after_admission = agent._fallback_index
            chain_after_admission = list(agent._fallback_chain)

        def execute_tool(name, arguments, *_args, **_kwargs):
            tool_calls.append((name, arguments))
            return TOOL_RESULT

        try:
            with (
                patch("model_tools.handle_function_call", side_effect=execute_tool),
                patch(
                    "hermes_cli.observability.shared_metrics_events.record_fallback",
                    side_effect=lambda **fields: activations.append(fields),
                ),
            ):
                result = agent.run_conversation(PROMPT)
            usage = server._get_usage(agent)
            messages = result["messages"]
            return {
                "result": result,
                "roles": [message.get("role") for message in messages if message.get("role") != "system"],
                "tool_results": [message.get("content") for message in messages if message.get("role") == "tool"],
                "primary_requests": primary.main_requests(),
                "stale_requests": stale.main_requests(),
                "fallback_requests": fallback.main_requests(),
                "activations": activations,
                "fallback_notices": [
                    text for _kind, text in statuses if "Model fallback:" in text
                ],
                "tool_calls": tool_calls,
                "usage_model": usage["model"],
                "provider_fallback_route": agent._provider_fallback_route,
                "chain_after_admission": chain_after_admission,
                "index_after_admission": index_after_admission,
                "replacement": replacement,
            }
        finally:
            agent.close()


def test_live_chain_replacement_matches_fresh_session_through_429_and_tool_round(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ISSUE28_FALLBACK_KEY", "issue-28-fallback-key")

    fresh = _exercise(monkeypatch, tmp_path / "fresh", already_live=False)
    live = _exercise(monkeypatch, tmp_path / "live", already_live=True)

    for outcome in (fresh, live):
        assert outcome["result"]["final_response"] == FINAL_RESPONSE
        assert outcome["roles"] == ["user", "assistant", "tool", "assistant"]
        assert outcome["tool_results"] == [TOOL_RESULT]
        assert outcome["tool_calls"] == [(TOOL_NAME, {"value": "issue-28"})]

        # One primary failure, then one fallback tool call and its conversational completion.
        # A self-fallback retry would add another primary request; a stale selection would hit stale.
        assert len(outcome["primary_requests"]) == 1
        assert outcome["stale_requests"] == []
        assert len(outcome["fallback_requests"]) == 2
        assert outcome["fallback_requests"][1]["messages"][-1]["role"] == "tool"
        assert TOOL_RESULT in outcome["fallback_requests"][1]["messages"][-1]["content"]

        # One activation is attributed from the failed primary to the configured fallback.
        assert len(outcome["activations"]) == 1
        activation = outcome["activations"][0]
        assert (activation["from_provider"], activation["to_provider"]) == ("openai", "custom")
        assert activation["reason"] is FailoverReason.billing
        assert outcome["provider_fallback_route"] == (FALLBACK_MODEL, "custom")
        assert outcome["usage_model"] == FALLBACK_MODEL
        assert len(outcome["fallback_notices"]) == 1
        assert PRIMARY_MODEL in outcome["fallback_notices"][0]
        assert FALLBACK_MODEL in outcome["fallback_notices"][0]

    assert live["chain_after_admission"] == live["replacement"]
    assert live["index_after_admission"] == 0
    assert fresh["index_after_admission"] == 0
    assert (live["result"]["final_response"], live["roles"], live["tool_results"]) == (
        fresh["result"]["final_response"],
        fresh["roles"],
        fresh["tool_results"],
    )


def _stateful_agent(chain: list[dict], *, active: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        _fallback_chain=list(chain),
        _fallback_model=chain[0] if chain else None,
        _fallback_index=len(chain),
        _fallback_activated=active,
        _rate_limited_until=float("inf") if active else 0,
        _unavailable_fallback_keys={"memoized-entry"},
    )


def test_invalid_config_is_ignored_but_valid_empty_config_clears_the_live_chain(
    monkeypatch, tmp_path
):
    previous = [{"provider": "custom", "model": STALE_MODEL}]
    agent = _stateful_agent(previous)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fallback_providers: [\n  - provider: {{{\n", encoding="utf-8")

    _admit_turn(monkeypatch, config_path, agent)

    assert agent._fallback_chain == previous
    assert agent._fallback_index == 1
    assert agent._unavailable_fallback_keys == {"memoized-entry"}

    config_path.write_text("fallback_providers: []\n", encoding="utf-8")
    _admit_turn(monkeypatch, config_path, agent)

    assert agent._fallback_chain == []
    assert agent._fallback_model is None
    assert agent._fallback_index == 0
    assert agent._unavailable_fallback_keys == set()


def test_active_fallback_cooldown_defers_a_persisted_chain_replacement(monkeypatch, tmp_path):
    active = [{"provider": "custom", "model": "active-fallback"}]
    replacement = [{"provider": "custom", "model": "replacement-fallback"}]
    agent = _stateful_agent(active, active=True)
    config_path = tmp_path / "config.yaml"
    _write_chain(config_path, replacement)

    _admit_turn(monkeypatch, config_path, agent)

    assert agent._fallback_chain == active
    assert agent._fallback_model == active[0]
    assert agent._fallback_index == 1
    assert agent._fallback_activated is True
    assert agent._unavailable_fallback_keys == {"memoized-entry"}
