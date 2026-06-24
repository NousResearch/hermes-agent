import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

moa = importlib.import_module("tools.mixture_of_agents_tool")


def test_moa_defaults_are_well_formed():
    # Invariants, not a catalog snapshot: the exact model list churns with
    # OpenRouter availability (see PR #6636 where gemini-3-pro-preview was
    # removed upstream). What we care about is that the defaults are present
    # and valid vendor/model slugs.
    assert isinstance(moa.REFERENCE_MODELS, list)
    assert len(moa.REFERENCE_MODELS) >= 1
    for m in moa.REFERENCE_MODELS:
        assert isinstance(m, str) and "/" in m and not m.startswith("/")
    assert isinstance(moa.AGGREGATOR_MODEL, str)
    assert "/" in moa.AGGREGATOR_MODEL


@pytest.mark.asyncio
async def test_reference_model_retry_warnings_avoid_exc_info_until_terminal_failure(monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=RuntimeError("rate limited"))
            )
        )
    )
    warn = MagicMock()
    err = MagicMock()

    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)
    monkeypatch.setattr(moa.logger, "warning", warn)
    monkeypatch.setattr(moa.logger, "error", err)

    model, message, success = await moa._run_reference_model_safe(
        "openai/gpt-5.4-pro", "hello", max_retries=2
    )

    assert model == "openai/gpt-5.4-pro"
    assert success is False
    assert "failed after 2 attempts" in message
    assert warn.call_count == 2
    assert all(call.kwargs.get("exc_info") is None for call in warn.call_args_list)
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_moa_top_level_error_logs_single_traceback_on_aggregator_failure(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(return_value=("anthropic/claude-opus-4.6", "ok", True)),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(side_effect=RuntimeError("aggregator boom")),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    err = MagicMock()
    monkeypatch.setattr(moa.logger, "error", err)

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "solve this",
            reference_models=["anthropic/claude-opus-4.6"],
        )
    )

    assert result["success"] is False
    assert "Error in MoA processing" in result["error"]
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


# ---------------------------------------------------------------------------
# _construct_aggregator_prompt
# ---------------------------------------------------------------------------


def test_construct_aggregator_prompt_enumerates_responses():
    prompt = moa._construct_aggregator_prompt("BASE", ["alpha", "beta"])
    assert "BASE" in prompt
    assert "1. alpha" in prompt
    assert "2. beta" in prompt


def test_construct_aggregator_prompt_empty_responses():
    prompt = moa._construct_aggregator_prompt("BASE", [])
    assert prompt == "BASE\n\n"


# ---------------------------------------------------------------------------
# _run_reference_model_safe — success and retry paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reference_model_success_returns_content(monkeypatch):
    fake_response = SimpleNamespace()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=fake_response)
            )
        )
    )
    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda r: "hello world")

    model, content, success = await moa._run_reference_model_safe(
        "anthropic/claude-opus-4.6", "hi", max_retries=2
    )
    assert model == "anthropic/claude-opus-4.6"
    assert content == "hello world"
    assert success is True


@pytest.mark.asyncio
async def test_reference_model_empty_content_retries_then_succeeds(monkeypatch):
    call_count = {"n": 0}

    async def fake_create(**kwargs):
        call_count["n"] += 1
        return SimpleNamespace()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)
        )
    )
    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)

    # First call: empty content → retry. Second call: real content.
    contents = iter(["", "real content"])
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda r: next(contents))
    monkeypatch.setattr(moa.asyncio, "sleep", AsyncMock())

    model, content, success = await moa._run_reference_model_safe(
        "anthropic/claude-opus-4.6", "hi", max_retries=3
    )
    assert success is True
    assert content == "real content"
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_reference_model_invalid_error_classified(monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=ValueError("invalid model spec"))
            )
        )
    )
    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)
    warn = MagicMock()
    monkeypatch.setattr(moa.logger, "warning", warn)
    monkeypatch.setattr(moa.asyncio, "sleep", AsyncMock())

    model, content, success = await moa._run_reference_model_safe(
        "bad/model", "hi", max_retries=2
    )
    assert success is False
    # The "invalid" branch logs a specific warning message.
    invalid_calls = [c for c in warn.call_args_list if "invalid" in c[0][0].lower()]
    assert len(invalid_calls) >= 1


@pytest.mark.asyncio
async def test_reference_model_unknown_error_classified(monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=RuntimeError("network down"))
            )
        )
    )
    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)
    warn = MagicMock()
    monkeypatch.setattr(moa.logger, "warning", warn)
    monkeypatch.setattr(moa.asyncio, "sleep", AsyncMock())

    model, content, success = await moa._run_reference_model_safe(
        "anthropic/claude-opus-4.6", "hi", max_retries=2
    )
    assert success is False
    unknown_calls = [c for c in warn.call_args_list if "unknown error" in c[0][0].lower()]
    assert len(unknown_calls) >= 1


# ---------------------------------------------------------------------------
# _run_aggregator_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregator_model_success(monkeypatch):
    fake_response = SimpleNamespace()
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=fake_response)
            )
        )
    )
    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda r: "aggregated result")

    result = await moa._run_aggregator_model("SYS", "USER")
    assert result == "aggregated result"


@pytest.mark.asyncio
async def test_aggregator_model_empty_content_retries_once(monkeypatch):
    call_count = {"n": 0}

    async def fake_create(**kwargs):
        call_count["n"] += 1
        return SimpleNamespace()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)
        )
    )
    monkeypatch.setattr(moa, "_get_openrouter_client", lambda: fake_client)

    contents = iter(["", "real aggregated"])
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda r: next(contents))

    result = await moa._run_aggregator_model("SYS", "USER")
    assert result == "real aggregated"
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# mixture_of_agents_tool — success, API key missing, insufficient references
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_moa_tool_success_path(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(return_value=("anthropic/claude-opus-4.6", "ref response", True)),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(return_value="final aggregated response"),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "solve this",
            reference_models=["anthropic/claude-opus-4.6"],
        )
    )
    assert result["success"] is True
    assert result["response"] == "final aggregated response"
    assert result["models_used"]["aggregator_model"] == moa.AGGREGATOR_MODEL


@pytest.mark.asyncio
async def test_moa_tool_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(
        await moa.mixture_of_agents_tool("solve this")
    )
    assert result["success"] is False
    assert "OPENROUTER_API_KEY" in result["error"]


@pytest.mark.asyncio
async def test_moa_tool_insufficient_references(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(return_value=("bad/model", "error", False)),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "solve this",
            reference_models=["bad/model"],
        )
    )
    assert result["success"] is False
    assert "Insufficient successful reference models" in result["error"]


@pytest.mark.asyncio
async def test_moa_tool_partial_failure_still_succeeds(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # One reference succeeds, one fails — still enough (MIN_SUCCESSFUL_REFERENCES=1).
    results = [
        ("model-a", "ref-a", True),
        ("model-b", "error", False),
    ]
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(side_effect=results),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(return_value="aggregated"),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "solve this",
            reference_models=["model-a", "model-b"],
        )
    )
    assert result["success"] is True
    assert result["response"] == "aggregated"


# ---------------------------------------------------------------------------
# check_moa_requirements + get_moa_configuration
# ---------------------------------------------------------------------------


def test_check_moa_requirements_delegates_to_openrouter(monkeypatch):
    monkeypatch.setattr(moa, "check_openrouter_api_key", lambda: True)
    assert moa.check_moa_requirements() is True


def test_check_moa_requirements_returns_false_when_no_key(monkeypatch):
    monkeypatch.setattr(moa, "check_openrouter_api_key", lambda: False)
    assert moa.check_moa_requirements() is False


def test_get_moa_configuration_returns_all_fields():
    config = moa.get_moa_configuration()
    assert config["reference_models"] == moa.REFERENCE_MODELS
    assert config["aggregator_model"] == moa.AGGREGATOR_MODEL
    assert config["reference_temperature"] == moa.REFERENCE_TEMPERATURE
    assert config["aggregator_temperature"] == moa.AGGREGATOR_TEMPERATURE
    assert config["min_successful_references"] == moa.MIN_SUCCESSFUL_REFERENCES
    assert config["total_reference_models"] == len(moa.REFERENCE_MODELS)
    assert "failure_tolerance" in config


# ---------------------------------------------------------------------------
# __main__ CLI demo block
# ---------------------------------------------------------------------------


def test_main_block_no_api_key_exits(monkeypatch):
    import runpy

    monkeypatch.setattr("tools.openrouter_client.check_api_key", lambda: False)
    exited = []
    monkeypatch.setattr(moa.sys, "exit", lambda code=0: exited.append(code))
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

    runpy.run_module("tools.mixture_of_agents_tool", run_name="__main__")
    assert exited == [1]


def test_main_block_with_api_key_prints_config(monkeypatch):
    import runpy

    monkeypatch.setattr("tools.openrouter_client.check_api_key", lambda: True)
    monkeypatch.setattr(moa.sys, "exit", lambda code=0: None)
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

    # _debug.active is False by default → covers the "disabled" branch.
    runpy.run_module("tools.mixture_of_agents_tool", run_name="__main__")


def test_main_block_debug_active_prints_session_id(monkeypatch):
    import runpy

    monkeypatch.setattr("tools.openrouter_client.check_api_key", lambda: True)
    monkeypatch.setattr(moa.sys, "exit", lambda code=0: None)
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

    # Patch DebugSession so _debug.active is True when the module re-executes.
    original_debug = moa.DebugSession

    class _FakeDebug:
        active = True
        session_id = "test-session"

        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr("tools.debug_helpers.DebugSession", _FakeDebug)
    monkeypatch.setattr(moa, "DebugSession", _FakeDebug)

    runpy.run_module("tools.mixture_of_agents_tool", run_name="__main__")
