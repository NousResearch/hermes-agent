import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

moa = importlib.import_module("tools.mixture_of_agents_tool")


def test_moa_defaults_track_current_direct_provider_stack():
    assert moa.REFERENCE_MODELS == [
        "minimax/MiniMax-M2.7-highspeed",
        "deepseek/deepseek-reasoner",
    ]
    assert moa.AGGREGATOR_MODEL == "xiaomi/mimo-v2-pro"


@pytest.mark.asyncio
async def test_reference_model_retry_warnings_avoid_exc_info_until_terminal_failure(monkeypatch):
    warn = MagicMock()
    err = MagicMock()

    monkeypatch.setattr(moa, "async_call_llm", AsyncMock(side_effect=RuntimeError("rate limited")))
    monkeypatch.setattr(moa.logger, "warning", warn)
    monkeypatch.setattr(moa.logger, "error", err)

    model, message, success = await moa._run_reference_model_safe(
        {"provider": "minimax", "model": "MiniMax-M2.7-highspeed"}, "hello", max_retries=2
    )

    assert model == "minimax/MiniMax-M2.7-highspeed"
    assert success is False
    assert "failed after 2 attempts" in message
    assert warn.call_count == 2
    assert all(call.kwargs.get("exc_info") is None for call in warn.call_args_list)
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_reference_model_empty_final_attempt_is_failure(monkeypatch):
    monkeypatch.setattr(moa, "async_call_llm", AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, reasoning=None, reasoning_content=None, reasoning_details=None))]
    )))

    model, message, success = await moa._run_reference_model_safe(
        {"provider": "minimax", "model": "MiniMax-M2.7-highspeed"}, "hello", max_retries=1
    )

    assert model == "minimax/MiniMax-M2.7-highspeed"
    assert success is False
    assert "empty reasoning-only content" in message


@pytest.mark.asyncio
async def test_moa_top_level_error_logs_single_traceback_on_aggregator_failure(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_run_reference_model_safe",
        AsyncMock(return_value=("minimax/MiniMax-M2.7-highspeed", "ok", True)),
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
    monkeypatch.setattr(moa, "_route_is_available", lambda route: True)

    err = MagicMock()
    monkeypatch.setattr(moa.logger, "error", err)

    result = json.loads(
        await moa.mixture_of_agents_tool(
            "solve this",
            reference_models=[{"provider": "minimax", "model": "MiniMax-M2.7-highspeed"}],
        )
    )

    assert result["success"] is False
    assert "Error in MoA processing" in result["error"]
    err.assert_called_once()
    assert err.call_args.kwargs.get("exc_info") is True


def test_check_moa_requirements_accepts_direct_provider_stack(monkeypatch):
    available = {
        "xiaomi/mimo-v2-pro",
        "minimax/MiniMax-M2.7-highspeed",
    }

    monkeypatch.setattr(
        moa,
        "_route_is_available",
        lambda route: moa._route_label(route) in available,
    )

    assert moa.check_moa_requirements() is True


def test_get_moa_configuration_reads_configured_route_overrides(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_load_moa_task_config",
        lambda: {
            "reference_models": [
                {"provider": "deepseek", "model": "deepseek-chat"},
            ],
            "aggregator_model": {
                "provider": "xiaomi",
                "model": "mimo-v2-flash",
            },
        },
    )

    config = moa.get_moa_configuration()

    assert config["reference_models"] == ["deepseek/deepseek-chat"]
    assert config["aggregator_model"] == "xiaomi/mimo-v2-flash"
