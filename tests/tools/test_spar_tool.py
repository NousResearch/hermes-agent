import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools import spar_tool as spar


def test_spar_registry_requires_current_provider_keys():
    assert spar.registry.get_entry("spar").requires_env == [
        "XIAOMI_API_KEY",
        "OPENROUTER_API_KEY",
    ]


def _llm_text(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                )
            )
        ]
    )


def test_filter_spar_review_auto_approves_cosmetic_only_feedback():
    review = spar.SparReview(
        approved=False,
        summary="Only naming polish remains.",
        issues=["Variable naming could be cleaner.", "Formatting is inconsistent."],
        fix="Rename the local helper for readability.",
    )

    filtered = spar.filter_spar_review(review)

    assert filtered.approved is True
    assert filtered.issues == []
    assert filtered.fix is None


def test_parse_spar_review_extracts_json_body():
    review = spar.parse_spar_review(
        'noise before {"approved": false, "summary": "Missing edge case.", "issues": ["Missing null handling"], "fix": "Handle null input."} noise after'
    )

    assert review.approved is False
    assert review.summary == "Missing edge case."
    assert review.issues == ["Missing null handling"]
    assert review.fix == "Handle null input."


def test_parse_spar_review_coerces_scalar_issues():
    review = spar.parse_spar_review(
        '{"approved": false, "summary": "Missing edge case.", "issues": "Missing null handling", "fix": "Handle null input."}'
    )

    assert review.issues == ["Missing null handling"]


def test_parse_spar_review_skips_invalid_prefix_and_extracts_fenced_json():
    review = spar.parse_spar_review(
        'bad prefix {not json}\n```json\n{"approved": true, "summary": "Looks good.", "issues": []}\n```\n'
    )

    assert review.approved is True
    assert review.summary == "Looks good."
    assert review.issues == []


@pytest.mark.asyncio
async def test_spar_call_route_retries_transient_api_failure(monkeypatch):
    monkeypatch.setattr(spar.asyncio, "sleep", AsyncMock())
    call = AsyncMock(side_effect=[RuntimeError("upstream timeout"), _llm_text("ok")])
    monkeypatch.setattr(spar, "async_call_llm", call)

    result = await spar._call_route(
        {"provider": "xiaomi", "model": "mimo-v2.5-pro"},
        [{"role": "user", "content": "hello"}],
        task="spar",
        temperature=0.0,
    )

    assert result == "ok"
    assert call.await_count == 2


@pytest.mark.asyncio
async def test_spar_tool_material_rejection_triggers_single_fix_round(monkeypatch):
    monkeypatch.setattr(
        spar,
        "async_call_llm",
        AsyncMock(
            side_effect=[
                _llm_text("initial answer"),
                _llm_text('{"approved": false, "summary": "Misses requirement.", "issues": ["Missing the requested output"], "fix": "Include the requested output."}'),
                _llm_text('{"approved": false, "summary": "Judge also sees the gap.", "issues": ["Missing the requested output"], "fix": "Include the requested output."}'),
                _llm_text("fixed answer"),
                _llm_text('{"approved": true, "summary": "Complete now.", "issues": []}'),
                _llm_text('{"approved": true, "summary": "Judge agrees now.", "issues": []}'),
            ]
        ),
    )

    result = json.loads(await spar.spar_tool("Do the task"))

    assert result["approved"] is True
    assert result["final_response"] == "fixed answer"
    phases = [entry["phase"] for entry in result["trace"]]
    assert phases == ["build.1", "spar.review.1", "judge.review.1", "spar.fix.1", "spar.review.2", "judge.review.2"]


@pytest.mark.asyncio
async def test_spar_tool_accepts_candidate_without_running_builder(monkeypatch):
    monkeypatch.setattr(
        spar,
        "async_call_llm",
        AsyncMock(
            side_effect=[
                _llm_text('{"approved": true, "summary": "Looks good.", "issues": []}'),
                _llm_text('{"approved": true, "summary": "Judge agrees.", "issues": []}'),
            ]
        ),
    )

    result = json.loads(await spar.spar_tool("Do the task", candidate_response="draft"))

    assert result["approved"] is True
    assert result["final_response"] == "draft"
    phases = [entry["phase"] for entry in result["trace"]]
    assert phases == ["candidate.input", "spar.review.1", "judge.review.1"]


@pytest.mark.asyncio
async def test_spar_tool_returns_rejection_after_fix_budget_is_exhausted(monkeypatch):
    monkeypatch.setattr(
        spar,
        "async_call_llm",
        AsyncMock(
            side_effect=[
                _llm_text("initial answer"),
                _llm_text('{"approved": false, "summary": "Still wrong.", "issues": ["Wrong output"], "fix": "Return the expected payload."}'),
                _llm_text('{"approved": false, "summary": "Judge still rejects.", "issues": ["Wrong output"], "fix": "Return the expected payload."}'),
                _llm_text("attempted fix"),
                _llm_text('{"approved": false, "summary": "Still wrong.", "issues": ["Wrong output"], "fix": "Return the expected payload."}'),
                _llm_text('{"approved": false, "summary": "Judge still rejects.", "issues": ["Wrong output"], "fix": "Return the expected payload."}'),
            ]
        ),
    )

    result = json.loads(await spar.spar_tool("Do the task"))

    assert result["approved"] is False
    assert result["issues"] == ["Wrong output"]
    assert result["final_response"] == "attempted fix"


def test_check_spar_requirements_uses_builder_and_reviewer_routes(monkeypatch):
    available = {
        "xiaomi/mimo-v2.5-pro",
        "nvidia/nemotron-3-super-120b-a12b",
    }
    monkeypatch.setattr(spar, "_default_builder_route", lambda: {"provider": "xiaomi", "model": "mimo-v2.5-pro", "label": "xiaomi/mimo-v2.5-pro"})
    monkeypatch.setattr(spar, "_route_is_available", lambda route: spar._route_label(route) in available)

    assert spar.check_spar_requirements() is True


@pytest.mark.asyncio
async def test_spar_tool_flags_reviewer_judge_disagreement(monkeypatch):
    monkeypatch.setattr(
        spar,
        "async_call_llm",
        AsyncMock(
            side_effect=[
                _llm_text("initial answer"),
                _llm_text('{"approved": true, "summary": "Reviewer approves.", "issues": []}'),
                _llm_text('{"approved": false, "summary": "Judge rejects.", "issues": ["Missing requirement"], "fix": "Add the missing requirement."}'),
            ]
        ),
    )

    result = json.loads(await spar.spar_tool("Do the task", max_fix_rounds=0))

    assert result["approved"] is True
    assert result["disagreement"] is True
    assert result["judge_verdict"]["approved"] is False


@pytest.mark.asyncio
async def test_spar_tool_records_failure_scar_on_final_rejection(monkeypatch):
    monkeypatch.setattr(
        spar,
        "async_call_llm",
        AsyncMock(
            side_effect=[
                _llm_text("initial answer"),
                _llm_text('{"approved": false, "summary": "Still wrong.", "issues": ["Wrong output"], "fix": "Return the expected payload."}'),
                _llm_text('{"approved": false, "summary": "Judge still rejects.", "issues": ["Wrong output"], "fix": "Return the expected payload."}'),
                _llm_text("attempted fix"),
                _llm_text('{"approved": false, "summary": "Still wrong.", "issues": ["Wrong output"], "fix": "Return the expected payload."}'),
                _llm_text('{"approved": false, "summary": "Judge still rejects.", "issues": ["Wrong output"], "fix": "Return the expected payload."}'),
            ]
        ),
    )
    recorder = MagicMock()
    monkeypatch.setattr(spar, "record_failure", recorder)

    result = json.loads(await spar.spar_tool("Do the task"))

    assert result["approved"] is False
    assert recorder.call_count == 1
