from __future__ import annotations

from datetime import datetime, timezone

from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.finance import build_torben_finance_radar_adapter


def _ratatosk_run(*, score: float = 0.82, can_place_order_directly: bool = False) -> dict:
    return {
        "status": "llm_completed",
        "phase": "open",
        "external_mutations": 0,
        "orders_submitted": 0,
        "llm_error": None,
        "llm_run": {
            "run_id": "llm-test-001",
            "cron_tick_id": "ratatosk-robinhood-v01-open:2026-06-26T09:45:00-04:00",
            "phase": "open",
            "token_budget": 2500,
            "order_tools_available": False,
        },
        "llm_result": {
            "market_regime": "risk-on but confirmation required",
            "no_trade_reason": "candidate is for review only; no order tools exposed",
            "watchlist": ["SPY"],
            "signals": [
                {
                    "symbol": "SPY",
                    "confidence": score,
                    "can_trade_directly": False,
                    "summary": "Broad-market confirmation signal.",
                }
            ],
            "candidates": [
                {
                    "symbol": "SPY",
                    "score": score,
                    "direction": "long",
                    "instrument_type": "equity",
                    "candidate_type": "watchlist_only",
                    "can_place_order_directly": can_place_order_directly,
                    "eligible_for_pretrade_guard": False,
                    "research_note": "SPY has a plausible setup but still needs price and risk confirmation.",
                    "constraints": ["cash-only", "no-margin", "no-shorts"],
                }
            ],
        },
    }


def test_finance_radar_stages_high_score_candidate_without_broker_mutation(tmp_path):
    payload = build_torben_finance_radar_adapter(
        _ratatosk_run(),
        ledger=ActionLedger(tmp_path / "actions.json"),
        state_path=tmp_path / "finance-state.json",
        now=datetime(2026, 6, 26, 14, 0, tzinfo=timezone.utc),
    )

    assert payload["wakeAgent"] is True
    assert payload["selected_count"] == 1
    assert payload["public_actions_taken"] == 0
    assert payload["external_mutations"] == 0
    assert payload["orders_submitted"] == 0
    assert payload["broker_orders_submitted"] == 0
    assert "No order was placed" in payload["text"]
    action = payload["actions"][0]
    assert action["handle"] == "FIN-20260626-001"
    assert action["scope"] == "fin"
    assert action["status"] == "staged"
    state = action["executor_state"]
    assert state["mutation_type"] == "broker_order_candidate"
    assert state["mutation_status"] == "stage_only_not_ordered"
    assert state["provider"] == "ratatosk_robinhood_v01"
    assert state["order_tools_available"] is False
    assert state["orders_submitted"] == 0
    assert state["external_mutations"] == 0
    assert "TBC-DECIDE-LIVE-FINANCE" in state["execution_blocked_until"]


def test_finance_radar_silent_for_below_threshold_watchlist(tmp_path):
    payload = build_torben_finance_radar_adapter(
        _ratatosk_run(score=0.58),
        ledger=ActionLedger(tmp_path / "actions.json"),
        state_path=tmp_path / "finance-state.json",
        now=datetime(2026, 6, 26, 14, 0, tzinfo=timezone.utc),
    )

    assert payload["wakeAgent"] is False
    assert payload["reason"] == "no fresh Ratatosk candidate reached the 0.70 review threshold"
    assert payload["selected_count"] == 0
    assert payload["text"] == ""
    assert payload["llm_judge"]["invoked"] is True
    assert payload["llm_judge"]["order_tools_available"] is False
    assert payload["external_mutations"] == 0
    assert payload["broker_orders_submitted"] == 0


def test_finance_radar_dedupes_delivered_candidate(tmp_path):
    ledger = ActionLedger(tmp_path / "actions.json")
    state_path = tmp_path / "finance-state.json"
    now = datetime(2026, 6, 26, 14, 0, tzinfo=timezone.utc)

    first = build_torben_finance_radar_adapter(
        _ratatosk_run(),
        ledger=ledger,
        state_path=state_path,
        now=now,
    )
    second = build_torben_finance_radar_adapter(
        _ratatosk_run(),
        ledger=ledger,
        state_path=state_path,
        now=now,
    )

    assert first["wakeAgent"] is True
    assert second["wakeAgent"] is False
    assert second["suppressed_duplicate_count"] == 1


def test_finance_radar_rejects_direct_order_candidate_even_when_score_is_high(tmp_path):
    payload = build_torben_finance_radar_adapter(
        _ratatosk_run(score=0.91, can_place_order_directly=True),
        ledger=ActionLedger(tmp_path / "actions.json"),
        state_path=tmp_path / "finance-state.json",
        now=datetime(2026, 6, 26, 14, 0, tzinfo=timezone.utc),
    )

    assert payload["wakeAgent"] is False
    assert payload["selected_count"] == 0
    assert payload["external_mutations"] == 0
    assert payload["broker_orders_submitted"] == 0


def test_finance_radar_fails_loud_if_ratatosk_reports_stage_only_mutation(tmp_path):
    run = _ratatosk_run(score=0.91)
    run["external_mutations"] = 1
    run["orders_submitted"] = 1

    payload = build_torben_finance_radar_adapter(
        run,
        ledger=ActionLedger(tmp_path / "actions.json"),
        state_path=tmp_path / "finance-state.json",
        now=datetime(2026, 6, 26, 14, 0, tzinfo=timezone.utc),
    )

    assert payload["wakeAgent"] is True
    assert payload["status"] == "fail_closed"
    assert payload["reason"] == "ratatosk_reported_stage_only_mutation"
    assert payload["actions"] == []
    assert payload["external_mutations"] == 1
    assert payload["broker_orders_submitted"] == 1
    assert "I did not stage a FIN review card" in payload["text"]
