"""Coalesced gateway approvals are keyed on the executed command, not its redacted display (#117815)."""

import threading

from tests.waiting import wait_until
from tools import approval as mod
from tools import approval_gateway_wait as wait_mod

SESSION_KEY = "approval-identity"
# Both requests render identically: redaction already replaced what differs between them.
DISPLAYED = {"command": "deploy --token ***", "description": "d", "pattern_key": "dangerous",
             "pattern_keys": ["dangerous"]}


def _run_two(monkeypatch, first: str, second: str):
    mod._gateway_queues.clear()
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda name, **kw: None)
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 60)
    notified: list[str] = []
    follows: list[str] = []
    real_follow = wait_mod._await_coalesced_leader

    def follow(session_key, leader, payload):
        follows.append(leader.data["request_id"])
        return real_follow(session_key, leader, payload)

    monkeypatch.setattr(wait_mod, "_await_coalesced_leader", follow)
    results: list[dict] = []

    def run(identity):
        results.append(wait_mod._await_gateway_decision(
            SESSION_KEY, notified.append, dict(DISPLAYED), identity=identity))

    threads = [threading.Thread(target=run, args=(identity,), daemon=True) for identity in (first, second)]
    threads[0].start()
    wait_until(lambda: len(notified) == 1, what="the first prompt")
    threads[1].start()
    wait_until(lambda: len(notified) == 2 or len(follows) == 1, what="the second request to settle")
    prompts = len(notified)
    mod.resolve_gateway_approval(SESSION_KEY, "session", resolve_all=True)
    for thread in threads:
        thread.join(10)
    mod._gateway_queues.clear()
    return prompts, results


def test_commands_that_render_alike_each_get_their_own_prompt(monkeypatch):
    prompts, results = _run_two(monkeypatch, "deploy --token aaa", "deploy --token bbb")
    assert prompts == 2
    assert not any(r.get("coalesced") for r in results)


def test_the_same_executed_command_still_coalesces(monkeypatch):
    prompts, results = _run_two(monkeypatch, "deploy --token aaa", "deploy --token aaa")
    assert prompts == 1
    assert sorted(bool(r.get("coalesced")) for r in results) == [False, True]
