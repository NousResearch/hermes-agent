"""Zero-delivery `type` verdict: web inputs behind trusted-event checks
swallow synthetic keystrokes entirely (delivered 0 of N). Retrying the
same delivery rung cannot help; the actionable next step is the AX
set_value path, which bypasses synthetic-event filtering."""

from __future__ import annotations

from tools.computer_use.backend import ActionResult
from tools.computer_use.tool import _classify_action_result


def _result(delivered):
    return ActionResult(
        ok=True,
        action="type",
        message="type_text incomplete",
        code="type_text_incomplete",
        meta={"delivered_chars": delivered, "requested_chars": 11},
    )


def test_zero_delivery_recommends_set_value():
    verdict = _classify_action_result(_result(0))
    assert verdict["decision"] == "escalate"
    assert verdict["recommended"] == "set_value"
    assert "set_value" in verdict["hint"]


def test_partial_delivery_keeps_generic_ladder():
    verdict = _classify_action_result(_result(7))
    assert verdict.get("recommended") != "set_value"


def test_synthesis_budget_refusal_keeps_driver_recommendation():
    """`type_text_synthesis_budget_exceeded` also reports delivered 0 (the
    budget declined to emit at all), but the target is not dropping
    keystrokes — re-routing it to set_value would replace the driver's own
    `chunk` retry instruction with a different write path."""
    verdict = _classify_action_result(
        ActionResult(
            ok=True,
            action="type",
            message="type_text synthesis budget exceeded",
            code="type_text_synthesis_budget_exceeded",
            meta={"delivered_chars": 0, "requested_chars": 6500},
            escalation={"recommended": "chunk", "reason": "requested run exceeds the synthesis budget"},
        )
    )
    assert verdict["decision"] == "escalate"
    assert verdict["recommended"] == "chunk"
    assert "set_value" not in verdict["hint"]
