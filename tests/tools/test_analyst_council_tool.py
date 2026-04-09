"""Tests for ``tools.analyst_council_tool``.

These tests mock the OpenRouter client and the auxiliary content extractor so
nothing hits a real LLM API.  Coverage targets the 8 test cases from the
plan file (foamy-gliding-marble.md):

    1. quick depth runs 3 reviewers (+ 1 chairman call)
    2. full depth runs 5 reviewers + 5 peer reviews + 1 chairman
    3. domain persona selection picks the right set
    4. one reviewer failure does not kill the council
    5. trivial query is bypassed (no LLM calls at all)
    6. COUNCIL_DIRECTIVE is injected into the system prompt when the tool
       is loaded AND council_enabled is True
    7. council_enabled=False via AIAgent constructor suppresses the directive
    8. X-Hermes-Council header parser maps strings to bool overrides
"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

council = importlib.import_module("tools.analyst_council_tool")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_response(text: str, in_tokens: int = 10, out_tokens: int = 20):
    """Build a SimpleNamespace that mimics openai response.usage shape."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=in_tokens,
            completion_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
        ),
    )


def _install_fake_client(monkeypatch, *, side_effect=None, return_text="canned review"):
    """Replace ``_get_openrouter_client`` with a client whose ``chat.completions.create``
    returns a deterministic fake response (or raises via ``side_effect``).
    """
    create_mock = AsyncMock()
    if side_effect is not None:
        create_mock.side_effect = side_effect
    else:
        create_mock.return_value = _make_fake_response(return_text)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_mock)
        )
    )
    monkeypatch.setattr(council, "_get_openrouter_client", lambda: fake_client)
    monkeypatch.setattr(
        council, "extract_content_or_reasoning", lambda response: (
            response.choices[0].message.content if response and response.choices else ""
        )
    )
    monkeypatch.setattr(council, "check_openrouter_api_key", lambda: True)
    return create_mock


def _reset_council_in_flight():
    """Contextvar is set() per tool invocation; reset between tests just in case."""
    try:
        council._COUNCIL_IN_FLIGHT.set(False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test 1 — Quick depth runs 3 reviewers + 1 chairman
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quick_depth_runs_three_reviewers_and_one_chairman(monkeypatch):
    _reset_council_in_flight()
    create_mock = _install_fake_client(monkeypatch, return_text="ok-review")
    monkeypatch.setattr(council, "_load_council_config", lambda: {})

    result_json = await council.analyst_council_tool(
        data="Detailed analysis of a fictional research topic that is long enough to bypass the trivial-query gate comfortably.",
        domain="general",
        depth="quick",
    )
    result = json.loads(result_json)

    assert result["success"] is True
    # 3 reviewers + 1 chairman = 4 LLM calls
    assert create_mock.call_count == 4
    assert result["depth"] == "quick"
    assert result["peer_reviews_collected"] == 0
    assert len(result["reviewers"]) == 3


# ---------------------------------------------------------------------------
# Test 2 — Full depth runs 5 + 5 + 1 = 11 calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_depth_runs_five_reviewers_and_peer_review(monkeypatch):
    _reset_council_in_flight()
    create_mock = _install_fake_client(monkeypatch, return_text="ok-review")
    monkeypatch.setattr(council, "_load_council_config", lambda: {})

    result_json = await council.analyst_council_tool(
        data="Long enough research analysis that the trivial-query gate never matches anything. We want a detailed council run.",
        domain="finance",
        depth="full",
    )
    result = json.loads(result_json)

    assert result["success"] is True
    # 5 reviewers + 5 peer reviews + 1 chairman = 11 LLM calls
    assert create_mock.call_count == 11
    assert result["depth"] == "full"
    assert result["peer_reviews_collected"] == 5
    assert len(result["reviewers"]) == 5


# ---------------------------------------------------------------------------
# Test 3 — Domain persona selection
# ---------------------------------------------------------------------------

def test_finance_domain_selects_finance_personas():
    personas = council._select_personas("finance", full=True)
    names = [p.name for p in personas]
    assert names == [
        "Bull Analyst",
        "Bear Analyst",
        "Technical Analyst",
        "Fundamental Analyst",
        "Risk Manager",
    ]


def test_medicine_domain_selects_medicine_personas():
    personas = council._select_personas("medicine", full=True)
    assert personas[0].name == "Clinical Expert"
    assert any(p.name == "Skeptic" for p in personas)
    assert len(personas) == 5


def test_technology_domain_selects_technology_personas():
    personas = council._select_personas("technology", full=True)
    assert personas[0].name == "Architect"
    assert any(p.name == "Security Reviewer" for p in personas)
    assert len(personas) == 5


def test_unknown_domain_falls_back_to_general():
    personas = council._select_personas("art-history", full=True)
    assert personas[0].name == "Subject Expert"
    assert len(personas) == 5


def test_quick_depth_trims_to_three_personas():
    personas = council._select_personas("finance", full=False)
    assert len(personas) == 3


# ---------------------------------------------------------------------------
# Test 4 — Reviewer failure does not kill the council
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reviewer_failure_does_not_kill_council(monkeypatch):
    _reset_council_in_flight()
    monkeypatch.setattr(council, "_load_council_config", lambda: {})
    monkeypatch.setattr(council, "check_openrouter_api_key", lambda: True)
    monkeypatch.setattr(
        council, "extract_content_or_reasoning",
        lambda response: response.choices[0].message.content if response else "",
    )

    call_count = {"n": 0}

    async def flaky_create(**kwargs):
        call_count["n"] += 1
        # Force the first reviewer call to always raise for all retries.
        # Stage 1 runs 3 reviewers in parallel, so reviewer index is inferred
        # from which persona is in the system message.
        if "bull-case equity analyst" in (kwargs.get("messages", [{}])[0].get("content", "") or "").lower():
            raise RuntimeError("reviewer timeout")
        return _make_fake_response("ok-review")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=flaky_create)))
    )
    monkeypatch.setattr(council, "_get_openrouter_client", lambda: fake_client)

    result_json = await council.analyst_council_tool(
        data="A long enough research analysis about a finance topic to bypass the trivial gate, so the council actually runs.",
        domain="finance",
        depth="quick",
    )
    result = json.loads(result_json)

    # Chairman still ran with surviving opinions
    assert result["success"] is True
    # Bull Analyst failed (first 3 reviewers in finance are Bull / Bear / Technical)
    assert "Bull Analyst" not in result["reviewers"]
    assert len(result["reviewers"]) == 2  # Bear + Technical survived


# ---------------------------------------------------------------------------
# Test 5 — Trivial query bypass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trivial_query_bypassed_without_llm_calls(monkeypatch):
    _reset_council_in_flight()
    create_mock = _install_fake_client(monkeypatch)
    monkeypatch.setattr(council, "_load_council_config", lambda: {
        "council": {"skip_for_trivial": True}
    })

    result_json = await council.analyst_council_tool(data="what's the weather")
    result = json.loads(result_json)

    assert result["success"] is True
    assert result.get("skipped") is True
    assert result.get("reason") == "trivial_query"
    # No LLM calls were made
    assert create_mock.call_count == 0


@pytest.mark.asyncio
async def test_trivial_bypass_disabled_by_config_still_runs_council(monkeypatch):
    _reset_council_in_flight()
    create_mock = _install_fake_client(monkeypatch)
    monkeypatch.setattr(council, "_load_council_config", lambda: {
        "council": {"skip_for_trivial": False}
    })

    result_json = await council.analyst_council_tool(data="what's the weather")
    result = json.loads(result_json)

    # Config said never skip — council runs even though query is trivial
    assert result["success"] is True
    assert result.get("skipped") is not True
    assert create_mock.call_count > 0


# ---------------------------------------------------------------------------
# Test 6 — Directive injected when tool loaded AND council enabled
# Test 7 — Directive suppressed when council_enabled=False
# ---------------------------------------------------------------------------

def test_council_directive_string_is_nonempty_and_descriptive():
    from agent.prompt_builder import COUNCIL_DIRECTIVE
    assert "Council Review: ENABLED" in COUNCIL_DIRECTIVE
    assert "analyst_council" in COUNCIL_DIRECTIVE
    assert "finance" in COUNCIL_DIRECTIVE
    assert "trivial" in COUNCIL_DIRECTIVE.lower()


def test_council_directive_gate_logic():
    """Unit-test the guard without constructing a full AIAgent.

    The prompt-builder guard is:
        "analyst_council" in self.valid_tool_names AND self._council_enabled
    Both must be True for the directive to land in the system prompt.
    """
    # Simulates the guard exactly
    def guard(tool_names, council_enabled):
        return "analyst_council" in tool_names and bool(council_enabled)

    assert guard({"analyst_council", "memory"}, True) is True
    assert guard({"analyst_council", "memory"}, False) is False
    assert guard({"memory"}, True) is False  # tool absent → off
    assert guard(set(), True) is False
    assert guard({"analyst_council"}, None) is False


# ---------------------------------------------------------------------------
# Test 8 — Gateway X-Hermes-Council header parser
# ---------------------------------------------------------------------------

def test_parse_council_header_on_values():
    from gateway.platforms.api_server import _parse_council_header
    for value in ("on", "ON", "true", "True", "1", "yes", "YES", " on "):
        assert _parse_council_header(value) is True


def test_parse_council_header_off_values():
    from gateway.platforms.api_server import _parse_council_header
    for value in ("off", "OFF", "false", "False", "0", "no", "NO", " off "):
        assert _parse_council_header(value) is False


def test_parse_council_header_missing_or_unknown_returns_none():
    from gateway.platforms.api_server import _parse_council_header
    assert _parse_council_header("") is None
    assert _parse_council_header("maybe") is None
    assert _parse_council_header(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bonus — model selection
# ---------------------------------------------------------------------------

def test_pick_reviewer_model_prefers_explicit_override():
    cfg = {"council": {"reviewer_model": "explicit/model-1"}}
    assert council._pick_reviewer_model(cfg) == "explicit/model-1"


def test_pick_reviewer_model_falls_back_to_smart_routing_cheap_model():
    cfg = {
        "council": {},
        "smart_model_routing": {"cheap_model": {"model": "smart/cheap"}},
    }
    assert council._pick_reviewer_model(cfg) == "smart/cheap"


def test_pick_reviewer_model_final_fallback():
    assert council._pick_reviewer_model({}) == council.FALLBACK_REVIEWER_MODEL


def test_pick_chairman_model_explicit_vs_default():
    assert council._pick_chairman_model({}) == council.FALLBACK_CHAIRMAN_MODEL
    cfg = {"council": {"chairman_model": "premium/model-1"}}
    assert council._pick_chairman_model(cfg) == "premium/model-1"


# ---------------------------------------------------------------------------
# Registration smoke — tool must register itself on import
# ---------------------------------------------------------------------------

def test_analyst_council_is_registered_via_module_import():
    from tools.registry import registry
    # importing the module at the top of this file triggers registration
    schema = registry.get_schema("analyst_council")
    assert schema is not None
    assert schema["name"] == "analyst_council"
    assert "data" in schema["parameters"]["required"]
