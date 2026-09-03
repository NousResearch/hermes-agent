"""Preview actions fail fast when the Desktop renderer bridge is unavailable."""

import json
import threading

import pytest

import tui_gateway.server as server


@pytest.fixture
def session(monkeypatch):
    record = {}
    monkeypatch.setitem(server._sessions, "s1", record)
    return record


@pytest.fixture
def bridge(monkeypatch):
    calls = []

    def fake_block(event, sid, payload, timeout=None, **_kw):
        answer = fake_block.answers.pop(0) if fake_block.answers else ""
        calls.append({"event": event, "sid": sid, "payload": payload, "timeout": timeout})
        return answer

    fake_block.answers = []
    fake_block.calls = calls
    monkeypatch.setattr(server, "_block", fake_block)
    return fake_block


def test_first_preview_action_uses_short_bridge_probe(session, bridge):
    bridge.answers = [json.dumps({"success": True})]

    server._preview_action_request("s1", {"action": "elements"})

    assert bridge.calls[0]["event"] == "preview.act.request"
    assert bridge.calls[0]["timeout"] == server._PREVIEW_ACTION_PROBE_TIMEOUT_S
    assert server._PREVIEW_ACTION_PROBE_TIMEOUT_S < server._PREVIEW_ACTION_TIMEOUT_S


def test_answered_preview_bridge_gets_full_action_deadline(session, bridge):
    bridge.answers = [json.dumps({"success": True}), json.dumps({"success": True})]

    server._preview_action_request("s1", {"action": "elements"})
    server._preview_action_request("s1", {"action": "click", "selector": "h1"})

    assert bridge.calls[1]["timeout"] == server._PREVIEW_ACTION_TIMEOUT_S


def test_unanswered_preview_probe_returns_actionable_error(session, bridge):
    result = json.loads(server._preview_action_request("s1", {"action": "elements"}))

    assert result["success"] is False
    assert "desktop" in result["error"].lower()
    assert "session" in result["error"].lower()


def test_repeated_preview_actions_short_circuit_after_unanswered_probe(session, bridge):
    server._preview_action_request("s1", {"action": "elements"})

    for payload in (
        {"action": "elements"},
        {"action": "annotate", "selector": "h1", "label": "Heading"},
    ):
        assert json.loads(server._preview_action_request("s1", payload))["success"] is False

    assert len(bridge.calls) == 1


def test_unknown_session_fails_without_creating_or_probing(bridge):
    server._sessions.pop("missing", None)

    result = server._preview_action_request("missing", {"action": "elements"})

    assert json.loads(result)["success"] is False
    assert "missing" not in server._sessions
    assert bridge.calls == []


def test_unanswered_bridge_reprobes_after_cooldown(session, bridge, monkeypatch):
    now = 100.0
    monkeypatch.setattr(server.time, "monotonic", lambda: now)

    server._preview_action_request("s1", {"action": "elements"})
    now += server._PREVIEW_ACTION_REPROBE_COOLDOWN_S - 0.1
    server._preview_action_request("s1", {"action": "elements"})
    assert len(bridge.calls) == 1

    now += 0.1
    bridge.answers = [json.dumps({"success": True})]
    result = server._preview_action_request("s1", {"action": "elements"})

    assert json.loads(result)["success"] is True
    assert len(bridge.calls) == 2
    assert session["preview_action_bridge"] == "answered"


def test_only_one_concurrent_caller_owns_expired_cooldown_reprobe(session, monkeypatch):
    now = 100.0
    session["preview_action_bridge"] = "unanswered"
    session["preview_action_bridge_retry_at"] = now
    monkeypatch.setattr(server.time, "monotonic", lambda: now)
    entered = threading.Event()
    release = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def blocking_probe(*_args, **_kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        entered.set()
        assert release.wait(2)
        return "", True

    monkeypatch.setattr(server, "_block", blocking_probe)
    results: list[str | None] = [None] * 5
    threads = [
        threading.Thread(
            target=lambda i=i: results.__setitem__(
                i, server._preview_action_request("s1", {"action": "elements"})
            )
        )
        for i in range(5)
    ]

    for thread in threads:
        thread.start()
    assert entered.wait(2)
    for thread in threads:
        thread.join(0.2)
    blocked_callers = sum(thread.is_alive() for thread in threads)

    release.set()
    for thread in threads:
        thread.join(2)

    assert not any(thread.is_alive() for thread in threads)
    assert blocked_callers == 1
    assert call_count == 1
    assert all(result is not None and json.loads(result)["success"] is False for result in results)


def test_cancelled_cooldown_reprobe_releases_ownership(session, monkeypatch):
    now = 100.0
    session["preview_action_bridge"] = "unanswered"
    session["preview_action_bridge_retry_at"] = now
    monkeypatch.setattr(server.time, "monotonic", lambda: now)
    calls = 0

    def cancel_then_answer(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return json.dumps({"success": True}), False

    monkeypatch.setattr(server, "_block", cancel_then_answer)

    with pytest.raises(KeyboardInterrupt):
        server._preview_action_request("s1", {"action": "elements"})
    result = server._preview_action_request("s1", {"action": "elements"})

    assert json.loads(result)["success"] is True
    assert calls == 2
    assert "preview_action_bridge_reprobe" not in session


def test_interrupted_preview_probe_does_not_poison_session(session, monkeypatch):
    calls = 0

    def interrupt_then_answer(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "", False
        return json.dumps({"success": True}), False

    monkeypatch.setattr(server, "_block", interrupt_then_answer)

    first = server._preview_action_request("s1", {"action": "elements"})
    second = server._preview_action_request("s1", {"action": "elements"})

    assert json.loads(first)["success"] is False
    assert json.loads(second)["success"] is True
    assert calls == 2
    assert session["preview_action_bridge"] == "answered"


def test_slow_action_does_not_condemn_proven_preview_bridge(session, bridge):
    bridge.answers = [json.dumps({"success": True})]
    server._preview_action_request("s1", {"action": "elements"})
    server._preview_action_request("s1", {"action": "click", "selector": "h1"})

    bridge.answers = [json.dumps({"success": True})]
    result = server._preview_action_request("s1", {"action": "elements"})

    assert json.loads(result)["success"] is True
    assert len(bridge.calls) == 3


def test_concurrent_probe_timeout_cannot_overwrite_answered_bridge(session, monkeypatch):
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    call_lock = threading.Lock()
    call_count = 0

    def ordered_block(*_args, **_kwargs):
        nonlocal call_count
        with call_lock:
            index = call_count
            call_count += 1
        entered[index].set()
        assert release[index].wait(2)
        return json.dumps({"success": True}) if index == 0 else ""

    monkeypatch.setattr(server, "_block", ordered_block)
    results: list[str | None] = [None, None]
    threads = [
        threading.Thread(
            target=lambda i=i: results.__setitem__(
                i, server._preview_action_request("s1", {"action": "elements"})
            )
        )
        for i in range(2)
    ]
    threads[0].start()
    assert entered[0].wait(2)
    threads[1].start()
    assert entered[1].wait(2)

    release[0].set()
    threads[0].join(2)
    release[1].set()
    threads[1].join(2)

    assert not any(thread.is_alive() for thread in threads)
    assert server._sessions["s1"]["preview_action_bridge"] == "answered"
