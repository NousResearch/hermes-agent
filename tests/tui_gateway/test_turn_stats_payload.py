"""Per-turn usage stats on message.complete (prompt_turn._turn_stats_delta / payload)."""
from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace

from tui_gateway.method_ctx import rebind
from tui_gateway import prompt_turn


def _usage(**kwargs):
    base = {
        "input": 0, "output": 0, "reasoning": 0,
        "cache_read": 0, "cache_write": 0, "calls": 0,
    }
    base.update(kwargs)
    return base


def _agent(**kwargs):
    defaults = dict(
        model="grok-4", provider="xai",
        session_input_tokens=0, session_output_tokens=0,
        session_reasoning_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_api_calls=0,
        session_estimated_cost_usd=0.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _st(agent, usage_before=None):
    return prompt_turn._TurnRun(
        agent, None, None, receipt_committed=True, usage_before=usage_before)


def test_turn_stats_delta_positive_diff():
    before = _usage(input=10, output=4, reasoning=1, cache_read=8, cache_write=2, calls=1)
    after = _usage(input=30, output=9, reasoning=3, cache_read=18, cache_write=5, calls=3)
    assert prompt_turn._turn_stats_delta(before, after) == {
        "input": 20, "output": 5, "reasoning": 2,
        "cache_read": 10, "cache_write": 3, "calls": 2,
    }


def test_turn_stats_delta_drops_negative_and_zero():
    before = _usage(input=100, output=10, reasoning=5, cache_read=40, cache_write=8, calls=4)
    after = _usage(input=20, output=10, reasoning=8, cache_read=0, cache_write=8, calls=4)
    # compression reset: input/cache_read negative → dropped; output/write/calls zero → dropped
    assert prompt_turn._turn_stats_delta(before, after) == {"reasoning": 3}


def test_build_turn_stats_payload_delta_and_cost(monkeypatch):
    monkeypatch.setattr(prompt_turn.time, "time", lambda: 1_700_000_007.4)
    agent = _agent(model="grok-4", provider="xai", session_estimated_cost_usd=0.42)
    before = _usage(input=10, output=2, calls=1)
    before["_cost_usd"] = 0.10
    after = _usage(input=40, output=12, calls=3, cache_read=5)
    session = {
        "agent": agent,
        "inflight_turn": {"started_at": 1_700_000_000.0, "assistant": ""},
    }
    stats = prompt_turn._build_turn_stats(session, _st(agent, before), after)
    assert stats is not None
    assert stats["input"] == 30
    assert stats["output"] == 10
    assert stats["calls"] == 2
    assert stats["cache_read"] == 5
    assert "reasoning" not in stats
    assert "cache_write" not in stats
    assert stats["duration_s"] == 7
    assert abs(stats["cost_usd"] - 0.32) < 1e-9
    assert stats["model"] == "grok-4"
    assert stats["provider"] == "xai"


def test_build_turn_stats_omits_negative_cost_and_empty_delta_without_duration():
    agent = _agent(session_estimated_cost_usd=0.01)
    before = _usage(input=50)
    before["_cost_usd"] = 0.50
    after = _usage(input=10)  # negative input dropped
    session = {"agent": agent}  # no inflight → no duration_s
    assert prompt_turn._build_turn_stats(session, _st(agent, before), after) is None


def test_build_turn_stats_skipped_when_compute_host_active():
    agent = _agent()
    before = _usage(input=1)
    after = _usage(input=9)
    session = {
        "agent": agent,
        "_compute_host_active": True,
        "inflight_turn": {"started_at": 1_700_000_000.0},
    }
    assert prompt_turn._build_turn_stats(session, _st(agent, before), after) is None


def test_complete_turn_payload_attaches_turn_stats_and_skips_compute_host(monkeypatch):
    monkeypatch.setattr(prompt_turn.time, "time", lambda: 1_700_000_004.4)
    agent = _agent(model="m", provider="p", session_estimated_cost_usd=1.5)
    before = _usage(input=5, output=1, calls=1)
    before["_cost_usd"] = 1.0
    after = _usage(input=15, output=6, calls=2, cache_read=4)

    persisted = []

    def _persist(session, st, turn_stats):
        persisted.append(dict(turn_stats))

    cleared = []

    g = dict(prompt_turn.__dict__)
    g.update({
        "_get_usage": lambda _a: dict(after),
        "_is_bot_mode_session": lambda _s: False,
        "render_message": lambda *_a, **_k: None,
        "_clear_inflight_turn": lambda s: cleared.append(s.pop("inflight_turn", None)),
        "_fail_inflight_turn": lambda *_a, **_k: None,
        "_retire_turn_marker": lambda *_a, **_k: None,
        "_persist_turn_stats": _persist,
    })
    complete = rebind(prompt_turn._complete_turn_payload, g)

    session = {
        "agent": agent,
        "history_lock": threading.Lock(),
        "session_key": "s1",
        "inflight_turn": {"started_at": 1_700_000_000.0, "assistant": "hi"},
    }
    st = _st(agent, before)
    st.result = {"final_response": "ok"}
    payload, raw, status = complete(session, st, None, 80)
    assert status == "complete"
    assert raw == "ok"
    assert "turn_stats" in payload
    assert payload["turn_stats"]["input"] == 10
    assert payload["turn_stats"]["output"] == 5
    assert payload["turn_stats"]["calls"] == 1
    assert payload["turn_stats"]["cache_read"] == 4
    assert payload["turn_stats"]["duration_s"] == 4
    assert abs(payload["turn_stats"]["cost_usd"] - 0.5) < 1e-9
    assert payload["turn_stats"]["model"] == "m"
    assert payload["turn_stats"]["provider"] == "p"
    assert persisted and persisted[0]["input"] == 10
    assert cleared  # duration was read before clear

    host_session = {
        "agent": agent,
        "_compute_host_active": True,
        "history_lock": threading.Lock(),
        "session_key": "s1",
        "inflight_turn": {"started_at": 1_700_000_000.0},
    }
    host_st = _st(agent, before)
    host_st.result = {"final_response": "ok"}
    host_payload, _, _ = complete(host_session, host_st, None, 80)
    assert "turn_stats" not in host_payload


def test_persist_turn_stats_skips_when_the_turn_wrote_no_assistant_row():
    """An error / interrupt turn must not stamp its stats onto the previous turn's message."""
    writes = []

    class _DB:
        def __init__(self, row_id):
            self._row_id = row_id

        def latest_message_row_id(self, _session_id, role="user", **_kwargs):
            return self._row_id

        def update_message_display_metadata(self, session_id, row_id, key, value):
            writes.append((session_id, row_id, key, value))

    def _db_opener(row_id):
        @contextlib.contextmanager
        def _open(_session):
            yield _DB(row_id)
        return _open

    agent = _agent()
    stats = {"input": 10}
    session = {"session_key": "s1"}

    g = dict(prompt_turn.__dict__)
    g["_session_db"] = _db_opener(9)
    persist = rebind(prompt_turn._persist_turn_stats, g)

    fresh_row = _st(agent)
    fresh_row.assistant_row_before = 7
    persist(session, fresh_row, stats)
    assert writes == [("s1", 9, "turn_stats", stats)]

    writes.clear()
    same_row = _st(agent)
    same_row.assistant_row_before = 9
    persist(session, same_row, stats)
    assert writes == []


def test_message_complete_contract_accepts_turn_stats():
    """`turn_stats` rides a payload model that forbids unknown keys: an undeclared field is a
    contract violation at emit time, warned past in production and raised under test isolation."""
    import pytest
    from pydantic import ValidationError

    from tui_gateway.contracts.events import MessageCompletePayload

    MessageCompletePayload.model_validate({
        "text": "done",
        "status": "complete",
        "usage": {
            "model": "m", "input": 40, "output": 12, "cache_read": 5,
            "cache_ttl_s": 300, "cache_refreshed_at": 1_700_000_000.0,
        },
        "turn_stats": {
            "duration_s": 4, "input": 30, "output": 10, "reasoning": 2,
            "cache_read": 5, "cache_write": 1, "calls": 2, "cost_usd": 0.5,
            "model": "m", "provider": "p",
        },
    })

    # The guard only means something while the model stays closed.
    with pytest.raises(ValidationError):
        MessageCompletePayload.model_validate({"text": "done", "not_a_declared_field": 1})

