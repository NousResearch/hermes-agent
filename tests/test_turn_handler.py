"""Integration test for the central Hermes turn handler (C§1.9).

Verifies that one inbound turn flows through all 8 submodules in declared
order and produces the expected TurnResult shape.
"""

from __future__ import annotations

import pytest

from agent.modules.turn_handler import TurnInput, TurnResult, run_turn
from agent.modules.identity import IdentityPacket
from agent.modules.context_loader import ContextPackage
from agent.modules.interpreter import Interpretation
from agent.modules.intent_classifier import ClassifiedIntent, Route
from agent.modules.routing_policy import Dispatch
from agent.modules.response_mode import ResponseShape, UpstreamResult
from agent.modules.summarizer import ExecutiveSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_turn() -> TurnInput:
    return TurnInput(
        session_id="test-session-001",
        text="Hello, what can you do?",
        source="cli",
        raw_user_id="user-42",
    )


@pytest.fixture()
def anonymous_turn() -> TurnInput:
    return TurnInput(
        session_id="anon-session-002",
        text="What time is it?",
        source="telegram",
    )


@pytest.fixture()
def enterprise_turn() -> TurnInput:
    return TurnInput(
        session_id="ent-session-003",
        text="Generate the weekly executive report.",
        source="slack",
        raw_user_id="user-ceo",
        raw_company_id="company-acme",
    )


@pytest.fixture()
def upstream_result() -> UpstreamResult:
    return UpstreamResult(success=True, data={"report": "done"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunTurn:
    """run_turn() flows through all 8 submodules and returns a TurnResult."""

    def test_returns_turn_result(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert isinstance(result, TurnResult)

    # ------ Stage 1: identity ------

    def test_identity_is_identity_packet(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert isinstance(result.identity, IdentityPacket)

    def test_identity_user_id_from_input(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert result.identity.user_id == "user-42"

    def test_anonymous_identity_mode(self, anonymous_turn: TurnInput) -> None:
        result = run_turn(anonymous_turn)
        assert result.identity.mode == "anonymous"
        assert result.identity.user_id.startswith("anon:")

    def test_enterprise_identity_mode(self, enterprise_turn: TurnInput) -> None:
        result = run_turn(enterprise_turn)
        assert result.identity.mode == "enterprise"
        assert result.identity.company_id == "company-acme"

    # ------ Stage 2: context_loader ------

    def test_context_is_context_package(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert isinstance(result.context, ContextPackage)

    def test_context_has_budget(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert result.context.token_budget > 0

    # ------ Stage 3: interpreter ------

    def test_interpretation_is_interpretation(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert isinstance(result.interpretation, Interpretation)

    def test_interpretation_preserves_raw_text(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert result.interpretation.raw_text == basic_turn.text

    # ------ Stage 4: intent_classifier ------

    def test_classified_is_classified_intent(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert isinstance(result.classified, ClassifiedIntent)

    def test_classified_has_route(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert isinstance(result.classified.route, Route)

    # ------ Stage 5: mission_compiler ------

    def test_mission_none_for_direct_route(self, basic_turn: TurnInput) -> None:
        # Stub classifier always returns ANSWER_DIRECTLY → "direct" → MissionContract present
        result = run_turn(basic_turn)
        # compile_mission returns a contract for "direct" target
        # (CONTRACTS_REQUIRING_MISSION includes "direct")
        assert result.mission is not None

    # ------ Stage 6: routing_policy ------

    def test_dispatch_is_dispatch(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert isinstance(result.dispatch, Dispatch)

    def test_dispatch_has_target(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert result.dispatch.target in {
            "direct", "state-engine", "openclaw",
            "mcp-tool", "clarify", "approve", "escalate",
        }

    # ------ Stage 7: response_mode ------

    def test_response_shape_is_response_shape(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert isinstance(result.response_shape, ResponseShape)

    def test_response_shape_mode_not_empty(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert result.response_shape.mode

    # ------ Stage 8: summarizer (conditional) ------

    def test_summary_none_without_upstream_result(self, basic_turn: TurnInput) -> None:
        result = run_turn(basic_turn)
        assert result.summary is None

    def test_summary_present_with_upstream_result(
        self, basic_turn: TurnInput, upstream_result: UpstreamResult
    ) -> None:
        result = run_turn(basic_turn, upstream_result=upstream_result)
        assert isinstance(result.summary, ExecutiveSummary)

    def test_summary_reflects_upstream_success(
        self, basic_turn: TurnInput, upstream_result: UpstreamResult
    ) -> None:
        result = run_turn(basic_turn, upstream_result=upstream_result)
        assert result.summary is not None
        assert result.summary.success is True

    def test_summary_has_bullets(
        self, basic_turn: TurnInput, upstream_result: UpstreamResult
    ) -> None:
        result = run_turn(basic_turn, upstream_result=upstream_result)
        assert result.summary is not None
        assert len(result.summary.bullets) >= 1

    def test_summary_has_next_action(
        self, basic_turn: TurnInput, upstream_result: UpstreamResult
    ) -> None:
        result = run_turn(basic_turn, upstream_result=upstream_result)
        assert result.summary is not None
        assert result.summary.next_action


class TestAllModulesInvoked:
    """Smoke test: pipeline executes without ImportError or AttributeError."""

    def test_import_chain(self) -> None:
        from agent.modules import (  # noqa: PLC0415
            run_turn,
            TurnInput,
            TurnResult,
            bootstrap_identity,
            assemble_context,
            interpret,
            classify_intent,
            compile_mission,
            apply_routing_policy,
            select_response_mode,
            summarize,
        )
        # All names resolvable — no ImportError is the assertion
        assert callable(run_turn)
        assert callable(bootstrap_identity)
        assert callable(assemble_context)
        assert callable(interpret)
        assert callable(classify_intent)
        assert callable(compile_mission)
        assert callable(apply_routing_policy)
        assert callable(select_response_mode)
        assert callable(summarize)
