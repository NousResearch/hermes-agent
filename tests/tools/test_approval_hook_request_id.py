"""Gateway approval hooks carry the request_id the gateway card carries.

An observer (a notifier plugin, a second client) keys its state on the id it can answer with
through ``approval.respond``. The same id must appear on ``pre_approval_request``,
``post_approval_response`` and the generic ``on_human_input_*`` pair for one prompt, and a
coalesced follower must report the leader's id, since its own entry is never enqueued.
"""
import threading
import time
from unittest.mock import patch

import pytest

from tools import approval as mod
from tools import approval_context

SESSION_KEY = "test-approval-hook-request-id"
COMMAND = "rm -rf .git"


@pytest.fixture
def gateway_session(monkeypatch):
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    for k in ("HERMES_CRON_SESSION", "HERMES_YOLO_MODE", "HERMES_INTERACTIVE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"mode": "manual", "timeout": 60})
    hooks = []
    with patch("hermes_cli.lifecycle.invoke_hook",
               side_effect=lambda name, **kw: hooks.append((name, kw)) or []):
        yield hooks
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()


def _named(hooks, name, **match):
    return [kw for n, kw in hooks if n == name and all(kw.get(k) == v for k, v in match.items())]


def test_one_request_id_across_pre_post_and_human_input_hooks(gateway_session):
    notified = []

    def notify(data):
        notified.append(data)
        mod.resolve_gateway_approval(SESSION_KEY, "deny")

    mod.register_gateway_notify(SESSION_KEY, notify)
    try:
        result = mod.check_all_command_guards(COMMAND, "local")
    finally:
        mod.unregister_gateway_notify(SESSION_KEY)

    assert result["approved"] is False
    assert len(notified) == 1
    rid = notified[0]["request_id"]
    assert rid
    assert [kw["request_id"] for kw in _named(gateway_session, "pre_approval_request")] == [rid]
    assert [kw["request_id"] for kw in _named(gateway_session, "post_approval_response")] == [rid]
    assert [kw["request_id"] for kw in _named(gateway_session, "on_human_input_request", kind="approval")] == [rid]
    assert [kw["request_id"] for kw in _named(gateway_session, "on_human_input_resolved", kind="approval")] == [rid]


def test_coalesced_follower_reports_the_leaders_request_id(gateway_session):
    notified = threading.Event()
    leader_data = {}

    def notify(data):
        leader_data.update(data)
        notified.set()

    mod.register_gateway_notify(SESSION_KEY, notify)
    results = {}
    try:
        leader = threading.Thread(target=lambda: results.__setitem__("leader", mod.check_all_command_guards(COMMAND, "local")))
        leader.start()
        assert notified.wait(timeout=10), "leader approval was never enqueued"
        follower = threading.Thread(target=lambda: results.__setitem__("follower", mod.check_all_command_guards(COMMAND, "local")))
        follower.start()  # identical command -> coalesces onto the leader, no second card
        deadline = time.monotonic() + 10
        while not _named(gateway_session, "pre_approval_request", coalesced=True):
            assert time.monotonic() < deadline, "follower never coalesced onto the leader"
            time.sleep(0.05)
        mod.resolve_gateway_approval(SESSION_KEY, "deny")
        leader.join(timeout=10)
        follower.join(timeout=10)
    finally:
        mod.unregister_gateway_notify(SESSION_KEY)

    assert not leader.is_alive() and not follower.is_alive()
    rid = leader_data["request_id"]
    assert [kw["request_id"] for kw in _named(gateway_session, "pre_approval_request", coalesced=True)] == [rid]
    assert [kw["request_id"] for kw in _named(gateway_session, "post_approval_response", coalesced=True)] == [rid]
    assert {kw["request_id"] for kw in _named(gateway_session, "post_approval_response")} == {rid}
