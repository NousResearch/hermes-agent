"""Tests for the council tool subsystem.

Tests persona loading, council query/evaluate/gate handlers,
DPO extraction, tool registration, and toolset membership.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure repo root on path
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


# =========================================================================
# Persona tests
# =========================================================================


class TestCouncilPersonas:
    """Tests for tools/council_personas.py."""

    def test_default_personas_loaded(self):
        from tools.council_personas import DEFAULT_PERSONAS

        assert len(DEFAULT_PERSONAS) == 5
        assert set(DEFAULT_PERSONAS.keys()) == {
            "advocate", "skeptic", "oracle", "contrarian", "arbiter"
        }

    def test_persona_fields(self):
        from tools.council_personas import DEFAULT_PERSONAS

        for name, persona in DEFAULT_PERSONAS.items():
            assert persona.name == name
            assert len(persona.tradition) > 0
            assert len(persona.system_prompt) > 50
            assert isinstance(persona.scoring_weights, dict)
            assert len(persona.scoring_weights) > 0
            assert isinstance(persona.tags, list)
            assert len(persona.tags) > 0

    def test_get_persona(self):
        from tools.council_personas import get_persona

        advocate = get_persona("advocate")
        assert advocate is not None
        assert advocate.name == "advocate"

        # Case insensitive
        skeptic = get_persona("Skeptic")
        assert skeptic is not None
        assert skeptic.name == "skeptic"

        # Non-existent
        assert get_persona("nonexistent") is None

    def test_list_personas(self):
        from tools.council_personas import list_personas

        names = list_personas()
        assert len(names) == 5
        assert "advocate" in names
        assert "arbiter" in names

    def test_scoring_weights_sum(self):
        from tools.council_personas import DEFAULT_PERSONAS

        for name, persona in DEFAULT_PERSONAS.items():
            total = sum(persona.scoring_weights.values())
            assert abs(total - 1.0) < 0.01, (
                f"{name} scoring weights sum to {total}, expected ~1.0"
            )

    def test_load_custom_personas_no_file(self):
        from tools.council_personas import load_custom_personas

        merged = load_custom_personas("/nonexistent/path.yaml")
        assert len(merged) == 5  # defaults only

    def test_load_custom_personas_with_yaml(self, tmp_path):
        from tools.council_personas import load_custom_personas

        config = tmp_path / "config.yaml"
        config.write_text(
            "council:\n"
            "  personas:\n"
            "    red_team:\n"
            "      tradition: 'Adversarial security'\n"
            "      system_prompt: 'You are a red team analyst'\n"
            "      tags: ['security']\n"
        )
        merged = load_custom_personas(str(config))
        assert "red_team" in merged
        assert merged["red_team"].tradition == "Adversarial security"
        assert len(merged) == 6  # 5 defaults + 1 custom


# =========================================================================
# PersonaResponse / CouncilVerdict dataclass tests
# =========================================================================


class TestCouncilDataclasses:
    """Tests for dataclass construction."""

    def test_persona_response(self):
        from tools.council_personas import PersonaResponse

        resp = PersonaResponse(
            persona_name="skeptic",
            content="This is flawed because...",
            confidence=0.8,
            dissents=True,
            key_points=["point 1", "point 2"],
            sources=["https://example.com"],
        )
        assert resp.persona_name == "skeptic"
        assert resp.dissents is True
        assert len(resp.key_points) == 2

    def test_council_verdict(self):
        from tools.council_personas import CouncilVerdict, PersonaResponse

        verdict = CouncilVerdict(
            question="test question",
            responses={"arbiter": PersonaResponse(
                persona_name="arbiter", content="synthesis",
                confidence=0.75, dissents=False,
            )},
            arbiter_synthesis="synthesis",
            confidence_score=75,
            conflict_detected=False,
        )
        assert verdict.confidence_score == 75
        assert verdict.conflict_detected is False
        assert len(verdict.dpo_pairs) == 0
        assert len(verdict.sources) == 0


# =========================================================================
# Response parsing tests
# =========================================================================


class TestResponseParsing:
    """Tests for response text parsing in council_tool.py."""

    def test_parse_confidence_decimal(self):
        from tools.council_tool import _parse_confidence

        assert abs(_parse_confidence("CONFIDENCE: 0.85") - 0.85) < 0.01

    def test_parse_confidence_percentage(self):
        from tools.council_tool import _parse_confidence

        assert abs(_parse_confidence("My confidence is 75%") - 0.75) < 0.01

    def test_parse_confidence_missing(self):
        from tools.council_tool import _parse_confidence

        assert abs(_parse_confidence("No confidence here") - 0.5) < 0.01

    def test_parse_dissent_true(self):
        from tools.council_tool import _parse_dissent

        assert _parse_dissent("DISSENT: true") is True
        assert _parse_dissent("DISSENT: True") is True

    def test_parse_dissent_false(self):
        from tools.council_tool import _parse_dissent

        assert _parse_dissent("DISSENT: false") is False
        assert _parse_dissent("no dissent info") is False

    def test_parse_key_points(self):
        from tools.council_tool import _parse_key_points

        text = (
            "Some text\n"
            "- First important point about the topic\n"
            "- Second point with more detail here\n"
            "* Third asterisk point is also valid\n"
            "- Short\n"  # too short, should be skipped
        )
        points = _parse_key_points(text)
        assert len(points) == 3

    def test_extract_sources(self):
        from tools.council_tool import _extract_sources

        text = "See https://example.com and https://data.gov/stats for more."
        sources = _extract_sources(text)
        assert len(sources) == 2
        assert "https://example.com" in sources

    def test_build_persona_response(self):
        from tools.council_tool import _build_persona_response

        raw = (
            "Analysis here.\n"
            "- Key point one about the topic discussed\n"
            "- Another point with supporting detail\n"
            "CONFIDENCE: 0.7\n"
            "DISSENT: true\n"
            "Source: https://example.com/evidence\n"
        )
        resp = _build_persona_response("skeptic", raw)
        assert resp.persona_name == "skeptic"
        assert abs(resp.confidence - 0.7) < 0.01
        assert resp.dissents is True
        assert len(resp.key_points) >= 2
        assert len(resp.sources) >= 1


# =========================================================================
# DPO extraction tests
# =========================================================================


class TestDPOExtraction:
    """Tests for DPO pair extraction."""

    def test_dpo_with_dissenter(self):
        from tools.council_personas import PersonaResponse
        from tools.council_tool import _extract_dpo_pairs

        responses = {
            "advocate": PersonaResponse(
                persona_name="advocate", content="Strong case for",
                confidence=0.9, dissents=False,
            ),
            "skeptic": PersonaResponse(
                persona_name="skeptic", content="Fatal flaw found",
                confidence=0.8, dissents=True,
            ),
            "arbiter": PersonaResponse(
                persona_name="arbiter", content="Synthesis here",
                confidence=0.75, dissents=False,
            ),
        }
        pairs = _extract_dpo_pairs("test q", responses)
        assert len(pairs) >= 1
        assert pairs[0]["chosen_persona"] == "arbiter"
        assert pairs[0]["rejected_persona"] == "skeptic"

    def test_dpo_no_dissenters(self):
        from tools.council_personas import PersonaResponse
        from tools.council_tool import _extract_dpo_pairs

        responses = {
            "advocate": PersonaResponse(
                persona_name="advocate", content="All agree",
                confidence=0.9, dissents=False,
            ),
            "arbiter": PersonaResponse(
                persona_name="arbiter", content="Synthesis",
                confidence=0.85, dissents=False,
            ),
        }
        pairs = _extract_dpo_pairs("test q", responses)
        # No dissenters = no primary pair
        assert len(pairs) == 0

    def test_dpo_empty_responses(self):
        from tools.council_tool import _extract_dpo_pairs

        pairs = _extract_dpo_pairs("test q", {})
        assert pairs == []


# =========================================================================
# Tool handler tests (mocked LLM)
# =========================================================================


class TestCouncilHandlers:
    """Tests for council tool handlers with mocked LLM calls."""

    @pytest.fixture(autouse=True)
    def mock_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")

    @pytest.mark.asyncio
    async def test_council_query_missing_question(self):
        from tools.council_tool import council_query_handler

        result = json.loads(await council_query_handler({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_council_query_success(self):
        from tools.council_tool import council_query_handler

        mock_response = (
            "Analysis content here.\n"
            "- Key point about the topic\n"
            "- Another important observation\n"
            "CONFIDENCE: 0.7\n"
            "DISSENT: false\n"
        )

        with patch("tools.council_tool._llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = json.loads(
                await council_query_handler({"question": "Is X better than Y?"})
            )

        assert result["success"] is True
        assert "confidence_score" in result
        assert "arbiter_synthesis" in result
        assert "persona_responses" in result
        assert isinstance(result["dpo_pairs"], list)

    @pytest.mark.asyncio
    async def test_council_evaluate_missing_content(self):
        from tools.council_tool import council_evaluate_handler

        result = json.loads(await council_evaluate_handler({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_council_evaluate_success(self):
        from tools.council_tool import council_evaluate_handler

        mock_response = (
            "Evaluation content.\n"
            "- This is well-researched\n"
            "- Could improve depth\n"
            "CONFIDENCE: 0.65\n"
            "DISSENT: false\n"
        )

        with patch("tools.council_tool._llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = json.loads(
                await council_evaluate_handler({
                    "content": "Some research output to evaluate",
                    "question": "Is the analysis thorough?",
                })
            )

        assert result["success"] is True
        assert "confidence_score" in result
        assert "persona_feedback" in result

    @pytest.mark.asyncio
    async def test_council_gate_missing_action(self):
        from tools.council_tool import council_gate_handler

        result = json.loads(await council_gate_handler({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_council_gate_success(self):
        from tools.council_tool import council_gate_handler

        mock_response = (
            "Safety analysis.\n"
            "- Low risk action\n"
            "CONFIDENCE: 0.8\n"
            "DISSENT: false\n"
        )

        with patch("tools.council_tool._llm_call", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = json.loads(
                await council_gate_handler({
                    "action": "Deploy to staging",
                    "risk_level": "low",
                })
            )

        assert result["success"] is True
        assert "allowed" in result
        assert isinstance(result["allowed"], bool)
        assert "confidence" in result
        assert "reasoning" in result


# =========================================================================
# Registration tests
# =========================================================================


class TestCouncilRegistration:
    """Tests for tool registration and toolset membership."""

    def test_tools_registered(self):
        from tools.registry import registry

        # Force import to trigger registration
        import tools.council_tool  # noqa: F401

        all_names = registry.get_all_tool_names()
        assert "council_query" in all_names
        assert "council_evaluate" in all_names
        assert "council_gate" in all_names

    def test_toolset_membership(self):
        from tools.registry import registry

        import tools.council_tool  # noqa: F401

        assert registry.get_toolset_for_tool("council_query") == "council"
        assert registry.get_toolset_for_tool("council_evaluate") == "council"
        assert registry.get_toolset_for_tool("council_gate") == "council"

    def test_check_fn_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from tools.council_tool import check_council_requirements

        assert check_council_requirements() is True

    def test_check_fn_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("NOUS_API_KEY", raising=False)
        from tools.council_tool import check_council_requirements

        assert check_council_requirements() is False

    def test_toolset_resolution(self):
        from toolsets import resolve_toolset

        tools = resolve_toolset("council")
        assert "council_query" in tools
        assert "council_evaluate" in tools
        assert "council_gate" in tools

    def test_council_query_in_core_tools(self):
        from toolsets import _HERMES_CORE_TOOLS

        assert "council_query" in _HERMES_CORE_TOOLS

    def test_schema_structure(self):
        from tools.council_tool import (
            COUNCIL_QUERY_SCHEMA,
            COUNCIL_EVALUATE_SCHEMA,
            COUNCIL_GATE_SCHEMA,
        )

        for schema in [COUNCIL_QUERY_SCHEMA, COUNCIL_EVALUATE_SCHEMA, COUNCIL_GATE_SCHEMA]:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert schema["parameters"]["type"] == "object"
            assert "properties" in schema["parameters"]
            assert "required" in schema["parameters"]


# =========================================================================
# API config tests
# =========================================================================


class TestAPIConfig:
    """Tests for API configuration resolution."""

    def test_openrouter_priority(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        from tools.council_tool import _get_api_config

        config = _get_api_config()
        assert config["api_key"] == "or-key"
        assert "openrouter" in config["base_url"]

    def test_nous_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("NOUS_API_KEY", "nous-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from tools.council_tool import _get_api_config

        config = _get_api_config()
        assert config["api_key"] == "nous-key"
        assert "nousresearch" in config["base_url"]

    def test_openai_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NOUS_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        from tools.council_tool import _get_api_config

        config = _get_api_config()
        assert config["api_key"] == "oa-key"

    def test_no_keys(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("NOUS_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from tools.council_tool import _get_api_config

        config = _get_api_config()
        assert config == {}
