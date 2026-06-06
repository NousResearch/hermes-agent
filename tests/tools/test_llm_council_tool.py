import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

council = importlib.import_module("tools.llm_council_tool")


def test_llm_council_defaults_are_well_formed():
    assert isinstance(council.DEFAULT_COUNCIL_MEMBERS, list)
    assert len(council.DEFAULT_COUNCIL_MEMBERS) >= council.MIN_SUCCESSFUL_MEMBERS
    for member in council.DEFAULT_COUNCIL_MEMBERS:
        assert member["role"]
        assert "/" in member["model"]
        assert member["persona"]
    assert "/" in council.DEFAULT_JUDGE_MODEL


def test_parse_vote_best_effort():
    parsed = council._parse_vote("Vote: Proposal 2\nConfidence: 0.73\nCritique: solid")
    assert parsed == {"vote": "Proposal 2", "confidence": 0.73}

    parsed = council._parse_vote("Vote: Combination\nConfidence: 7")
    assert parsed == {"vote": "Combination", "confidence": 1.0}

    parsed = council._parse_vote("no structured vote")
    assert parsed == {"vote": "unparsed", "confidence": None}


def test_normalize_custom_models_keeps_roles_and_caps():
    members = council._normalize_council_members([
        "a/model-1",
        "b/model-2",
        "c/model-3",
        "d/model-4",
        "e/model-5",
        "f/model-6",
    ])
    assert [m["model"] for m in members] == [
        "a/model-1",
        "b/model-2",
        "c/model-3",
        "d/model-4",
        "e/model-5",
    ]
    assert members[0]["role"] == "Architect"
    assert members[1]["role"] == "Skeptic"


@pytest.mark.asyncio
async def test_llm_council_success_flow(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(council, "_debug", SimpleNamespace(log_call=MagicMock(), save=MagicMock()))

    async def fake_call(model, messages, temperature, max_tokens=8000, max_retries=2):
        system = messages[0]["content"]
        if "neutral judge" in system:
            return model, "Final judged answer", True
        if "Critique the council proposals" in system:
            role = system.split(" in an LLM council", 1)[0].replace("You are the ", "")
            return model, f"Vote: Proposal 1\nConfidence: 0.80\nCritique: {role} agrees\nNeeded fixes: none", True
        role = system.split(" in an LLM council", 1)[0].replace("You are the ", "")
        return model, f"Summary: {role} proposal\nRecommended answer: do it", True

    monkeypatch.setattr(council, "_call_model_safe", fake_call)

    result = json.loads(await council.llm_council_tool("Should we ship?", include_transcript=True))

    assert result["success"] is True
    assert result["response"] == "Final judged answer"
    assert result["council_size"] == 3
    assert len(result["votes"]) == 3
    assert all(v["vote"] == "Proposal 1" for v in result["votes"])
    assert "transcript" in result


@pytest.mark.asyncio
async def test_llm_council_requires_minimum_successful_members(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(council, "_debug", SimpleNamespace(log_call=MagicMock(), save=MagicMock()))

    async def fake_proposal(member, user_prompt):
        return {
            "role": member["role"],
            "model": member["model"],
            "content": "only one succeeded" if member["role"] == "Architect" else "failed",
            "success": member["role"] == "Architect",
        }

    monkeypatch.setattr(council, "_run_member_proposal", fake_proposal)
    monkeypatch.setattr(council, "_run_member_critique", AsyncMock())
    monkeypatch.setattr(council, "_run_judge", AsyncMock())

    result = json.loads(await council.llm_council_tool("Hard question"))

    assert result["success"] is False
    assert "Insufficient successful council members" in result["error"]
    council._run_member_critique.assert_not_called()
    council._run_judge.assert_not_called()


def test_llm_council_registry_shape():
    entry = council.registry.get_entry("llm_council")
    assert entry is not None
    assert entry.toolset == "moa"
    assert entry.is_async is True
    assert entry.schema["name"] == "llm_council"
