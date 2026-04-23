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
        "_run_reference_model_detailed",
        AsyncMock(return_value={
            "model": "minimax/MiniMax-M2.7-highspeed",
            "provider": "minimax",
            "success": True,
            "content": "ok",
            "error": "",
            "attempts": 1,
            "latency_seconds": 0.1,
            "output_chars": 2,
        }),
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


@pytest.mark.asyncio
async def test_moa_returns_full_reference_forensics(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_run_reference_model_detailed",
        AsyncMock(side_effect=[
            {
                "model": "minimax/MiniMax-M2.7-highspeed",
                "provider": "minimax",
                "success": True,
                "content": "MiniMax says choose crypto because momentum is higher right now.",
                "error": "",
                "attempts": 1,
                "latency_seconds": 1.25,
                "output_chars": 63,
            },
            {
                "model": "deepseek/deepseek-reasoner",
                "provider": "deepseek",
                "success": False,
                "content": "",
                "error": "deepseek failed",
                "attempts": 2,
                "latency_seconds": 2.5,
                "output_chars": 0,
            },
        ]),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(return_value="Final synthesized answer"),
    )
    monkeypatch.setattr(
        moa,
        "_run_moa_forensic_analysis",
        AsyncMock(return_value=(
            {
                "decision_trace": {
                    "model_proposals": {
                        "minimax/MiniMax-M2.7-highspeed": ["crypto momentum"],
                    },
                    "overlap": [],
                    "conflicts": [],
                    "final_candidates": ["crypto"],
                    "synthesis_summary": "MiMo kept the strongest growth thesis.",
                },
                "aggregator_influence_log": {
                    "kept_from_models": {
                        "minimax/MiniMax-M2.7-highspeed": ["crypto momentum"],
                    },
                    "discarded_or_deprioritized": ["weak macro hedges"],
                    "resolution_notes": ["No conflict because only one model succeeded."],
                    "influence_summary": "MiMo mostly followed MiniMax.",
                },
            },
            {
                "model": "xiaomi/mimo-v2-pro",
                "provider": "xiaomi",
                "success": True,
                "latency_seconds": 0.8,
                "output_chars": 240,
                "error": "",
            },
        )),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )
    monkeypatch.setattr(moa, "_route_is_available", lambda route: True)

    result = json.loads(await moa.mixture_of_agents_tool("compare assets"))

    assert result["success"] is True
    assert result["failed_models"] == ["deepseek/deepseek-reasoner"]
    assert result["failed_model_errors"] == {"deepseek/deepseek-reasoner": "deepseek failed"}
    assert result["reference_previews"] == {
        "minimax/MiniMax-M2.7-highspeed": "MiniMax says choose crypto because momentum is higher right now."
    }
    assert result["reference_outputs"] == {
        "minimax/MiniMax-M2.7-highspeed": "MiniMax says choose crypto because momentum is higher right now."
    }
    assert result["per_model_metrics"]["reference_models"]["minimax/MiniMax-M2.7-highspeed"]["attempts"] == 1
    assert result["per_model_metrics"]["reference_models"]["deepseek/deepseek-reasoner"]["error"] == "deepseek failed"
    assert result["per_model_metrics"]["aggregator"]["model"] == "xiaomi/mimo-v2-pro"
    assert result["per_model_metrics"]["forensic_analysis"]["success"] is True
    assert result["decision_trace"]["final_candidates"] == ["crypto"]
    assert result["aggregator_influence_log"]["kept_from_models"] == {
        "minimax/MiniMax-M2.7-highspeed": ["crypto momentum"]
    }
