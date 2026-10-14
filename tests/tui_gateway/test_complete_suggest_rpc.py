"""complete.suggest: deterministic candidates are free and always win; the aux-model prediction runs
only when opted in AND nothing deterministic fits, never carries an echoed role label, and is
dropped if a new turn moved history."""

from __future__ import annotations

import types

import agent.auxiliary_client as aux
import tui_gateway.server as srv


def _reply(text):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])


def test_suggest_gating_and_prediction_fallback(monkeypatch):
    calls = []
    replies = ["USER: Open the PR too"]  # an echoed role label is stripped
    session = {"history": [], "history_version": 7, "history_lock": None, "agent": None}
    monkeypatch.setitem(srv._sessions, "s1", session)

    def fake_llm(**kw):
        calls.append(kw["task"])
        return _reply(replies[-1])

    monkeypatch.setattr(aux, "call_llm", fake_llm)

    def run(history, **display):
        session["history"] = history
        monkeypatch.setattr(srv, "_load_cfg", lambda: {"display": display})
        return srv._methods["complete.suggest"](1, {"session_id": "s1"})["result"]

    ask = [{"role": "assistant", "content": "Want me to delete it?"}, {"role": "user", "content": "no"},
           {"role": "assistant", "content": [{"type": "text", "text": "Done. Want me to push?"}]}]
    done = [{"role": "user", "content": "fix it"}, {"role": "assistant", "content": "Fixed and pushed."}]

    assert run(ask, composer_suggestions=False, composer_predictions=True)["candidates"] == []
    assert run(ask, composer_predictions=True) == {
        "candidates": [{"text": "Yes, go ahead", "kind": "confirm"}], "history_version": 7}
    assert run(done)["candidates"] == [] and calls == []  # predictions are opt-in

    assert run(done, composer_predictions=True)["candidates"] == [{"text": "Open the PR too", "kind": "prediction"}]
    assert calls == ["composer_predictions"]

    replies.append("NONE")
    assert run(done, composer_predictions=True)["candidates"] == []
    replies.append("Open the PR too")

    def stale_llm(**kw):
        session["history_version"] += 1  # a new turn landed mid-prediction
        return _reply(replies[-1])

    monkeypatch.setattr(aux, "call_llm", stale_llm)
    assert run(done, composer_predictions=True)["candidates"] == []
