import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

moa = importlib.import_module("tools.mixture_of_agents_tool")


def test_moa_defaults_track_current_direct_provider_stack():
    assert moa.REFERENCE_MODELS == [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
    ]
    assert moa.AGGREGATOR_MODEL == "xiaomi/mimo-v2-pro"


def test_moa_registry_requires_current_provider_keys():
    assert moa.registry.get_entry("mixture_of_agents").requires_env == [
        "XIAOMI_API_KEY",
        "OPENROUTER_API_KEY",
    ]


def test_construct_aggregator_prompt_keeps_model_labels():
    prompt = moa._construct_aggregator_prompt(
        "Base",
        [
            ("xiaomi/mimo-v2-pro (self-draft)", "MiMo first pass"),
            ("minimax/MiniMax-M2.7-highspeed", "MiniMax pass"),
        ],
    )

    assert "[xiaomi/mimo-v2-pro (self-draft)]" in prompt
    assert "[minimax/MiniMax-M2.7-highspeed]" in prompt
    assert "MiMo first pass" in prompt
    assert "MiniMax pass" in prompt


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
        "nvidia/nemotron-3-super-120b-a12b:free",
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


def test_legacy_paid_moa_config_upgrades_to_free_defaults(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_load_moa_task_config",
        lambda: {
            "reference_models": [
                {"provider": "minimax", "model": "MiniMax-M2.7-highspeed"},
                {"provider": "deepseek", "model": "deepseek-reasoner"},
            ],
        },
    )

    config = moa.get_moa_configuration()

    assert config["reference_models"] == [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
    ]


def test_forensic_analysis_placeholder_detector_flags_template_echo():
    assert moa._forensic_analysis_has_placeholders({
        "decision_trace": {
            "model_proposals": {"model_label": ["proposal", "..."]},
            "overlap": ["shared idea"],
            "conflicts": [],
            "final_candidates": ["final pick"],
            "synthesis_summary": "short summary",
        },
        "aggregator_influence_log": {
            "kept_from_models": {"actual_model_name": ["kept point"]},
            "discarded_or_deprioritized": [],
            "resolution_notes": [],
            "influence_summary": "concrete summary",
        },
    }) is True


def test_fallback_forensic_analysis_uses_raw_outputs():
    result = moa._fallback_forensic_analysis(
        {
            "minimax/MiniMax-M2.7-highspeed": "Winner: AKG\nReason one\nReason two",
            "deepseek/deepseek-reasoner": "Winner: Lithium\nExtra detail",
        },
        "Winner: AKG\nFinal answer body",
    )

    assert result["decision_trace"]["model_proposals"]["minimax/MiniMax-M2.7-highspeed"][0] == "AKG"
    assert result["decision_trace"]["final_candidates"][0] == "AKG"
    assert result["aggregator_influence_log"]["kept_from_models"]["minimax/MiniMax-M2.7-highspeed"][0] == "AKG"


def test_extract_json_object_handles_trailing_duplicate_payloads():
    payload = moa._extract_json_object(
        '```json\n{"decision_trace":{"model_proposals":{},"overlap":[],"conflicts":[],"final_candidates":[],"synthesis_summary":"ok"},"aggregator_influence_log":{"kept_from_models":{},"discarded_or_deprioritized":[],"resolution_notes":[],"influence_summary":"ok"}}\n```\n{"ignored":true}'
    )

    assert payload["decision_trace"]["synthesis_summary"] == "ok"
    assert payload["aggregator_influence_log"]["influence_summary"] == "ok"


def test_forensic_analysis_empty_detector_flags_blank_schema():
    assert moa._forensic_analysis_is_empty(moa._empty_forensic_analysis()) is True


@pytest.mark.asyncio
async def test_moa_forensic_analysis_retries_invalid_reply_once(monkeypatch):
    valid_json = json.dumps({
        "decision_trace": {
            "model_proposals": {"minimax/MiniMax-M2.7-highspeed": ["akg"]},
            "overlap": ["akg"],
            "conflicts": [],
            "final_candidates": ["akg"],
            "synthesis_summary": "kept overlap",
        },
        "aggregator_influence_log": {
            "kept_from_models": {"minimax/MiniMax-M2.7-highspeed": ["akg"]},
            "discarded_or_deprioritized": ["lithium"],
            "resolution_notes": ["used overlap"],
            "influence_summary": "clean",
        },
    })
    llm = AsyncMock(side_effect=["no json here", valid_json])
    monkeypatch.setattr(moa, "async_call_llm", llm)
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda response: response)

    parsed, metrics = await moa._run_moa_forensic_analysis(
        {"provider": "xiaomi", "model": "mimo-v2-pro"},
        "compare compounds",
        {"minimax/MiniMax-M2.7-highspeed": "AKG"},
        "Final answer: AKG",
    )

    assert llm.await_count == 2
    assert parsed["decision_trace"]["final_candidates"] == ["akg"]
    assert parsed["aggregator_influence_log"]["influence_summary"] == "clean"
    assert metrics["success"] is True


@pytest.mark.asyncio
async def test_moa_forensic_analysis_retries_empty_schema_once(monkeypatch):
    valid_json = json.dumps({
        "decision_trace": {
            "model_proposals": {"deepseek/deepseek-reasoner": ["refuse"]},
            "overlap": [],
            "conflicts": ["minimax picked B"],
            "final_candidates": ["B"],
            "synthesis_summary": "picked B",
        },
        "aggregator_influence_log": {
            "kept_from_models": {"minimax/MiniMax-M2.7-highspeed": ["B"]},
            "discarded_or_deprioritized": ["A"],
            "resolution_notes": ["preferred upside"],
            "influence_summary": "clean",
        },
    })
    llm = AsyncMock(side_effect=[json.dumps(moa._empty_forensic_analysis()), valid_json])
    monkeypatch.setattr(moa, "async_call_llm", llm)
    monkeypatch.setattr(moa, "extract_content_or_reasoning", lambda response: response)

    parsed, metrics = await moa._run_moa_forensic_analysis(
        {"provider": "xiaomi", "model": "mimo-v2-pro"},
        "compare compounds",
        {"minimax/MiniMax-M2.7-highspeed": "B"},
        "Final answer: B",
    )

    assert llm.await_count == 2
    assert parsed["decision_trace"]["final_candidates"] == ["B"]
    assert metrics["success"] is True


@pytest.mark.asyncio
async def test_moa_returns_full_reference_forensics(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_run_reference_model_detailed",
        AsyncMock(side_effect=[
            {
                "model": "xiaomi/mimo-v2-pro (self-draft)",
                "provider": "xiaomi",
                "success": True,
                "content": "MiMo self-draft says crypto has the clearest asymmetric upside.",
                "error": "",
                "attempts": 1,
                "latency_seconds": 0.9,
                "output_chars": 63,
            },
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

    result = json.loads(await moa.mixture_of_agents_tool("compare assets", enable_forensic_analysis=True))

    assert result["success"] is True
    assert result["models_used"]["reference_models"] == [
        "xiaomi/mimo-v2-pro (self-draft)",
        "minimax/MiniMax-M2.7-highspeed",
    ]
    assert result["failed_models"] == ["deepseek/deepseek-reasoner"]
    assert result["failed_model_errors"] == {"deepseek/deepseek-reasoner": "deepseek failed"}
    assert result["reference_previews"] == {
        "xiaomi/mimo-v2-pro (self-draft)": "MiMo self-draft says crypto has the clearest asymmetric upside.",
        "minimax/MiniMax-M2.7-highspeed": "MiniMax says choose crypto because momentum is higher right now."
    }
    assert result["reference_outputs"] == {
        "xiaomi/mimo-v2-pro (self-draft)": "MiMo self-draft says crypto has the clearest asymmetric upside.",
        "minimax/MiniMax-M2.7-highspeed": "MiniMax says choose crypto because momentum is higher right now."
    }
    assert result["per_model_metrics"]["reference_models"]["xiaomi/mimo-v2-pro (self-draft)"]["provider"] == "xiaomi"
    assert result["per_model_metrics"]["reference_models"]["minimax/MiniMax-M2.7-highspeed"]["attempts"] == 1
    assert result["per_model_metrics"]["reference_models"]["deepseek/deepseek-reasoner"]["error"] == "deepseek failed"
    assert result["per_model_metrics"]["aggregator"]["model"] == "xiaomi/mimo-v2-pro"
    assert result["per_model_metrics"]["forensic_analysis"]["success"] is True
    assert result["decision_trace"]["final_candidates"] == ["crypto"]
    assert result["aggregator_influence_log"]["kept_from_models"] == {
        "minimax/MiniMax-M2.7-highspeed": ["crypto momentum"]
    }


@pytest.mark.asyncio
async def test_moa_v2_requires_successful_external_reference(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_run_reference_model_detailed",
        AsyncMock(side_effect=[
            {
                "model": "xiaomi/mimo-v2-pro (self-draft)",
                "provider": "xiaomi",
                "success": True,
                "content": "MiMo self-draft",
                "error": "",
                "attempts": 1,
                "latency_seconds": 0.9,
                "output_chars": 15,
            },
            {
                "model": "minimax/MiniMax-M2.7-highspeed",
                "provider": "minimax",
                "success": False,
                "content": "",
                "error": "minimax failed",
                "attempts": 2,
                "latency_seconds": 1.2,
                "output_chars": 0,
            },
            {
                "model": "deepseek/deepseek-reasoner",
                "provider": "deepseek",
                "success": False,
                "content": "",
                "error": "deepseek failed",
                "attempts": 2,
                "latency_seconds": 1.5,
                "output_chars": 0,
            },
        ]),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(return_value="should not run"),
    )
    monkeypatch.setattr(
        moa,
        "_run_moa_forensic_analysis",
        AsyncMock(return_value=(moa._empty_forensic_analysis(), {})),
    )
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )
    monkeypatch.setattr(moa, "_route_is_available", lambda route: True)

    result = json.loads(await moa.mixture_of_agents_tool("compare assets"))

    assert result["success"] is False
    assert "Insufficient successful external reference models" in result["error"]
    assert result["reference_outputs"] == {
        "xiaomi/mimo-v2-pro (self-draft)": "MiMo self-draft"
    }


@pytest.mark.asyncio
async def test_moa_skips_extra_forensic_llm_call_by_default(monkeypatch):
    monkeypatch.setattr(
        moa,
        "_run_reference_model_detailed",
        AsyncMock(side_effect=[
            {
                "model": "xiaomi/mimo-v2-pro (self-draft)",
                "provider": "xiaomi",
                "success": True,
                "content": "MiMo self-draft says buy index funds.",
                "error": "",
                "attempts": 1,
                "latency_seconds": 0.9,
                "output_chars": 38,
            },
            {
                "model": "minimax/MiniMax-M2.7-highspeed",
                "provider": "minimax",
                "success": True,
                "content": "MiniMax says buy index funds.",
                "error": "",
                "attempts": 1,
                "latency_seconds": 1.2,
                "output_chars": 29,
            },
            {
                "model": "deepseek/deepseek-reasoner",
                "provider": "deepseek",
                "success": False,
                "content": "",
                "error": "deepseek failed",
                "attempts": 2,
                "latency_seconds": 1.5,
                "output_chars": 0,
            },
        ]),
    )
    monkeypatch.setattr(
        moa,
        "_run_aggregator_model",
        AsyncMock(return_value="Final synthesized answer"),
    )
    forensic = AsyncMock(return_value=(moa._empty_forensic_analysis(), {}))
    monkeypatch.setattr(moa, "_run_moa_forensic_analysis", forensic)
    monkeypatch.setattr(
        moa,
        "_debug",
        SimpleNamespace(log_call=MagicMock(), save=MagicMock(), active=False),
    )
    monkeypatch.setattr(moa, "_route_is_available", lambda route: True)

    result = json.loads(await moa.mixture_of_agents_tool("compare assets"))

    forensic.assert_not_awaited()
    assert result["success"] is True
    assert result["per_model_metrics"]["forensic_analysis"]["skipped"] is True
    assert result["decision_trace"]["final_candidates"]
