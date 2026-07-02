"""Regression tests for #21563 — cross-process approval handshake.

The gateway's pending-approval state lives in-process (``_gateway_queues`` +
``threading.Event``), so an external supervisor process — e.g. the MCP bridge
(``hermes mcp serve``), a separate stdio subprocess — could neither see nor
resolve approvals: ``permissions_list_open`` always returned zero approvals
and ``permissions_respond`` reported success without ever unblocking the
waiting agent thread.

The fix mirrors every pending gateway approval to
``<HERMES_HOME>/approvals/pending/<id>.json`` while the agent thread waits,
and consumes decisions from ``<HERMES_HOME>/approvals/responses/<id>.json``
on each tick of the wait loop. These tests drive the real wait loop on a
real thread against real files — no mocked-out IPC.
"""

import json
import os
import tempfile
import threading
import time

import pytest


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir so the handshake dirs are isolated."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def _clean_approval_state():
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    yield
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()


def _short_timeout(monkeypatch, seconds):
    from tools import approval as mod
    monkeypatch.setattr(
        mod, "_get_approval_config",
        lambda: {"mode": "manual", "gateway_timeout": seconds,
                 "timeout": seconds},
    )


APPROVAL_DATA = {
    "command": "rm -rf build/",
    "description": "Recursive delete of the build directory",
    "pattern_key": "rm_rf",
    "pattern_keys": ["rm_rf"],
}

SESSION_KEY = "external-resolution-session"


def _start_wait(notify_cb=None, session_key=SESSION_KEY):
    """Run _await_gateway_decision on a worker thread; return (thread, box)."""
    from tools import approval as mod
    box = {}

    def _run():
        box["result"] = mod._await_gateway_decision(
            session_key, notify_cb or (lambda data: None),
            dict(APPROVAL_DATA),
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, box


def _assert_decision(result, *, resolved, choice, reason=None):
    assert result == {"resolved": resolved, "choice": choice, "reason": reason}


def _wait_for(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _pending_dir(home):
    return home / "approvals" / "pending"


def _responses_dir(home):
    return home / "approvals" / "responses"


def _pending_files(home):
    return sorted(_pending_dir(home).glob("*.json"))


def _write_response(home, approval_id, payload):
    """Atomically place a response file, like a supervisor would."""
    responses = _responses_dir(home)
    responses.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(responses), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        if isinstance(payload, dict):
            json.dump(payload, fh)
        else:
            fh.write(payload)
    os.replace(tmp, responses / f"{approval_id}.json")


class TestPublishLifecycle:
    def test_pending_record_published_while_waiting(
            self, tmp_path, monkeypatch):
        """A blocked approval is mirrored with the full decision context."""
        from tools import approval as mod
        _short_timeout(monkeypatch, 30)

        thread, box = _start_wait()
        assert _wait_for(lambda: _pending_files(tmp_path)), \
            "pending approval was never mirrored"

        record = json.loads(_pending_files(tmp_path)[0].read_text())
        assert record["id"] == _pending_files(tmp_path)[0].stem
        assert record["session_key"] == SESSION_KEY
        assert record["command"] == APPROVAL_DATA["command"]
        assert record["description"] == APPROVAL_DATA["description"]
        assert record["pattern_keys"] == ["rm_rf"]
        assert record["surface"] == "gateway"
        assert record["expires_at"] > record["created_at"]

        # In-process resolution (the /approve path) retracts the mirror.
        assert mod.resolve_gateway_approval(SESSION_KEY, "once") == 1
        thread.join(timeout=5)
        assert not thread.is_alive()
        _assert_decision(box["result"], resolved=True, choice="once")
        assert _wait_for(lambda: not _pending_files(tmp_path)), \
            "mirror not retracted after in-process resolution"

    def test_notify_failure_publishes_nothing(self, tmp_path, monkeypatch):
        """No user notification → no external mirror (nothing to decide on)."""
        _short_timeout(monkeypatch, 30)

        def _broken_notify(data):
            raise RuntimeError("platform send failed")

        thread, box = _start_wait(notify_cb=_broken_notify)
        thread.join(timeout=5)
        assert box["result"]["notify_failed"] is True
        assert not _pending_files(tmp_path)

    def test_timeout_retracts_mirror(self, tmp_path, monkeypatch):
        """An unanswered approval leaves no stale pending file behind."""
        _short_timeout(monkeypatch, 1)

        thread, box = _start_wait()
        assert _wait_for(lambda: _pending_files(tmp_path))
        thread.join(timeout=5)
        _assert_decision(box["result"], resolved=False, choice=None)
        assert not _pending_files(tmp_path)

    def test_interrupt_resolves_deny_and_retracts(self, tmp_path, monkeypatch):
        """/stop during a pending approval cleans up the mirror too."""
        from tools import approval as mod
        _short_timeout(monkeypatch, 30)
        monkeypatch.setattr(mod, "is_interrupted", lambda: True)

        thread, box = _start_wait()
        thread.join(timeout=5)
        _assert_decision(box["result"], resolved=True, choice="deny")
        assert not _pending_files(tmp_path)


class TestExternalDecision:
    def test_external_decision_unblocks_wait_and_maps_choice(
            self, tmp_path, monkeypatch):
        """The core #21563 fix: a supervisor's decision resolves the wait."""
        _short_timeout(monkeypatch, 30)

        thread, box = _start_wait()
        assert _wait_for(lambda: _pending_files(tmp_path))
        approval_id = _pending_files(tmp_path)[0].stem

        _write_response(tmp_path, approval_id, {"decision": "always"})

        thread.join(timeout=5)
        assert not thread.is_alive(), \
            "external decision did not unblock the waiting agent thread"
        _assert_decision(box["result"], resolved=True, choice="always")
        assert not _pending_files(tmp_path)
        assert not list(_responses_dir(tmp_path).glob("*.json"))

    def test_invalid_external_decision_fails_closed(
            self, tmp_path, monkeypatch):
        """Garbage decisions are consumed and ignored — never auto-approve."""
        _short_timeout(monkeypatch, 30)

        thread, box = _start_wait()
        assert _wait_for(lambda: _pending_files(tmp_path))
        approval_id = _pending_files(tmp_path)[0].stem

        _write_response(tmp_path, approval_id, {"decision": "yolo-everything"})
        assert _wait_for(
            lambda: not list(_responses_dir(tmp_path).glob("*.json")))
        assert thread.is_alive(), \
            "invalid decision must not resolve the approval"

        _write_response(tmp_path, approval_id, {"decision": "deny"})
        thread.join(timeout=5)
        _assert_decision(box["result"], resolved=True, choice="deny")

    def test_unparseable_response_file_ignored(self, tmp_path, monkeypatch):
        """A corrupt response file is discarded without resolving."""
        from tools import approval as mod
        _short_timeout(monkeypatch, 30)

        thread, box = _start_wait()
        assert _wait_for(lambda: _pending_files(tmp_path))
        approval_id = _pending_files(tmp_path)[0].stem

        _write_response(tmp_path, approval_id, "this is not json")
        assert _wait_for(
            lambda: not list(_responses_dir(tmp_path).glob("*.json")))
        assert thread.is_alive()

        mod.resolve_gateway_approval(SESSION_KEY, "once")
        thread.join(timeout=5)
        _assert_decision(box["result"], resolved=True, choice="once")

    def test_consume_external_decision_absent_returns_none(self):
        from tools import approval as mod
        assert mod._consume_external_decision("no-such-approval") is None


class TestStaleSweep:
    def test_publish_sweeps_expired_garbage_and_old_responses(self, tmp_path):
        from tools import approval as mod

        pending = _pending_dir(tmp_path)
        responses = _responses_dir(tmp_path)
        pending.mkdir(parents=True)
        responses.mkdir(parents=True)

        now = time.time()
        (pending / "expired1.json").write_text(
            json.dumps({"id": "expired1", "expires_at": now - 10}))
        (pending / "garbage.json").write_text("{{{not json")
        (pending / "alive.json").write_text(
            json.dumps({"id": "alive", "expires_at": now + 300}))
        old_response = responses / "orphan.json"
        old_response.write_text(json.dumps({"decision": "once"}))
        os.utime(old_response, (now - 7200, now - 7200))

        mod._publish_pending_approval(
            "fresh", "session-x", dict(APPROVAL_DATA), 60, "gateway")

        names = {p.name for p in _pending_files(tmp_path)}
        assert names == {"alive.json", "fresh.json"}
        assert not old_response.exists()


class TestEndToEndMcpBridge:
    """Full #21563 loop: real wait thread ⇄ real EventBridge, real files."""

    def test_bridge_lists_responds_and_unblocks_agent_thread(
            self, tmp_path, monkeypatch):
        from mcp_serve import EventBridge
        _short_timeout(monkeypatch, 30)

        thread, box = _start_wait()
        assert _wait_for(lambda: _pending_files(tmp_path))

        bridge = EventBridge()
        approvals = bridge.list_pending_approvals()
        assert len(approvals) == 1
        assert approvals[0]["command"] == APPROVAL_DATA["command"]
        approval_id = approvals[0]["id"]

        events = bridge.poll_events(after_cursor=0)["events"]
        assert any(e["type"] == "approval_requested" for e in events)

        result = bridge.respond_to_approval(approval_id, "once",
                                            confirm_timeout=5.0)
        assert result["resolved"] is True

        thread.join(timeout=5)
        assert not thread.is_alive(), \
            "permissions_respond must unblock the waiting agent thread"
        _assert_decision(box["result"], resolved=True, choice="once")

        assert bridge.list_pending_approvals() == []
        events = bridge.poll_events(after_cursor=0)["events"]
        assert any(e["type"] == "approval_resolved" for e in events)
