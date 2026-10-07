"""Managed-turn usage is an opt-in, turn-scoped observation; never session totals."""
from types import SimpleNamespace
import threading

import pytest
from pydantic import ValidationError
from tui_gateway.managed_turn_usage import ManagedTurnUsageObserver, managed_turn_usage_scope
from tui_gateway import server
from tui_gateway.contracts.registry import EVENTS


def usage(*, input_tokens, output_tokens, cache_read_tokens=0, cache_write_tokens=0):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
                           reasoning_tokens=0, prompt_tokens=input_tokens + cache_read_tokens,
                           request_count=1, total_tokens=input_tokens + output_tokens + cache_read_tokens)


def test_moa_managed_usage_refuses_multi_model_folding_before_model():
    agent = SimpleNamespace(api_mode="chat_completions", provider="moa")
    with pytest.raises(RuntimeError, match="MoA"):
        with managed_turn_usage_scope("a" * 32, 42, "ui", agent, lambda *_: None):
            pytest.fail("model must not start")
    assert getattr(agent, "_managed_turn_usage_callback", None) is None


def test_codex_app_server_and_missing_receipt_refuse_managed_usage_before_model():
    agent = SimpleNamespace(api_mode="codex_app_server")
    with pytest.raises(RuntimeError, match="Codex app-server"):
        with managed_turn_usage_scope("a" * 32, 42, "ui", agent, lambda *_: None):
            pytest.fail("model must not start")
    assert getattr(agent, "_managed_turn_usage_callback", None) is None
    with pytest.raises(RuntimeError, match="committed user row"):
        with managed_turn_usage_scope("a" * 32, None, "ui", SimpleNamespace(), lambda *_: None):
            pytest.fail("model must not start")


def test_managed_usage_event_contract_requires_correlated_complete_shape():
    contract = EVENTS["managed_turn.usage"].payload
    good = {"managed_turn_key": "a" * 32, "user_row_id": 42, "model": "model-a",
            "observed_calls": 1, "observed_usage_complete": True, "coverage": "accounted_responses_only",
            "usage": {"calls": 1, "input": 7, "output": 2, "total": 9,
                      "cache_read": 0, "cache_write": 0, "reasoning": 0}}
    contract.model_validate(good)
    for invalid in ({**good, "user_row_id": 0}, {**good, "observed_calls": 0},
                    {**good, "usage": {**good["usage"], "extra": "not owned"}},
                    {**good, "usage": {**good["usage"], "total": -1}}):
        with pytest.raises(ValidationError):
            contract.model_validate(invalid)


def test_managed_turn_hooks_call_usage_before_completion_then_clears_for_normal_turn(monkeypatch):
    events = []
    callbacks = []
    class Agent:
        model = "model-a"
        def run_conversation(self, _text, **_kwargs):
            callback = getattr(self, "_managed_turn_usage_callback", None)
            if callback:
                callbacks.append(callback)
                callback(usage(input_tokens=7, output_tokens=2), model=self.model,
                         served_model=self.model, raw_usage_complete=True)
                callback(None, model=self.model, served_model=self.model, raw_usage_complete=False)
            return {"final_response": "done", "messages": []}
    agent = Agent()
    session = {"session_key": "stored", "agent": agent, "history_lock": threading.Lock(),
               "_submit_user_row": {"_row_id": 42, "content": "hello"}}
    monkeypatch.setattr(server, "_is_bot_mode_session", lambda _session: False)
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_adopt_submit_user_row", lambda *args: None)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *args: (threading.Event(), SimpleNamespace(join=lambda: None)))
    monkeypatch.setattr(server, "_emit", lambda typ, sid, payload=None: events.append((typ, sid, payload)))
    st = server._TurnRun(agent, None, None, True)
    server._invoke_agent("ui", session, st, "hello", "hello", None, [], None, None,
                         text="hello", managed_turn_key="a" * 32)
    before = len(events)
    callbacks[0](usage(input_tokens=900, output_tokens=900), model="model-a",
                 served_model="model-a", raw_usage_complete=True)
    assert len(events) == before
    assert [(typ, payload["observed_calls"]) for typ, _, payload in events if typ == "managed_turn.usage"] == [
        ("managed_turn.usage", 1), ("managed_turn.usage", 2)]
    assert all(sid == "ui" and payload["user_row_id"] == 42 and payload["managed_turn_key"] == "a" * 32
               for typ, sid, payload in events if typ == "managed_turn.usage")
    assert getattr(agent, "_managed_turn_usage_callback", None) is None
    st2 = server._TurnRun(agent, None, None, True)
    server._invoke_agent("ui", session, st2, "next", "next", None, [], None, None, text="next")
    assert len(events) == before


def test_managed_usage_counts_only_observed_turn_calls_and_keeps_missing_usage_unknown():
    events = []
    observer = ManagedTurnUsageObserver("a" * 32, 42, "ui-session",
                                        lambda sid, payload: events.append((sid, payload)))
    observer.record(usage(input_tokens=6, output_tokens=2), model="model-a", served_model="model-a", raw_usage_complete=True)
    observer.record(None, model="model-a", served_model="model-a", raw_usage_complete=False)
    observer.record(usage(input_tokens=4, output_tokens=3, cache_read_tokens=1),
                    model="model-a", served_model="model-a", raw_usage_complete=True)
    observer.close()
    observer.record(usage(input_tokens=900, output_tokens=900), model="model-a",
                    served_model="model-a", raw_usage_complete=True)
    assert len(events) == 3
    assert [payload["observed_calls"] for _, payload in events] == [1, 2, 3]
    assert [payload["observed_usage_complete"] for _, payload in events] == [True, False, False]
    assert all(payload["coverage"] == "accounted_responses_only" for _, payload in events)
    assert events[-1] == ("ui-session", {
        "managed_turn_key": "a" * 32, "user_row_id": 42, "model": "model-a",
        "observed_calls": 3, "observed_usage_complete": False, "coverage": "accounted_responses_only",
        "usage": {"calls": 2, "input": 10, "output": 5, "cache_read": 1,
                  "cache_write": 0, "reasoning": 0, "total": 16},
    })


def test_unverified_served_model_or_partial_provider_usage_never_counts_as_complete():
    for served, raw_complete in (("different-model", True), ("model-a", False), (None, True)):
        events = []
        observer = ManagedTurnUsageObserver("c" * 32, 44, "ui", lambda _sid, payload: events.append(payload))
        observer.record(usage(input_tokens=10, output_tokens=2), model="model-a",
                        served_model=served, raw_usage_complete=raw_complete)
        assert events[0]["observed_usage_complete"] is False
        assert events[0]["usage"]["calls"] == 0
        assert events[0]["usage"]["total"] == 0


def test_model_change_and_invalid_usage_never_become_complete():
    events = []
    observer = ManagedTurnUsageObserver("b" * 32, 43, "ui-session", lambda _sid, value: events.append(value))
    observer.record(usage(input_tokens=2, output_tokens=1), model="model-a", served_model="model-a", raw_usage_complete=True)
    observer.record(usage(input_tokens=5, output_tokens=1), model="model-b", served_model="model-b", raw_usage_complete=True)
    observer.record(usage(input_tokens=-1, output_tokens=1), model="model-a", served_model="model-a", raw_usage_complete=True)
    observer.record(usage(input_tokens=1, output_tokens=1, cache_write_tokens=100),
                    model="model-a", served_model="model-a", raw_usage_complete=True)
    assert events[-1]["observed_usage_complete"] is False
    assert events[-1]["model"] == "model-a"
    assert events[-1]["observed_calls"] == 4
    assert events[-1]["usage"]["total"] == 3  # no cross-model or malformed addition
