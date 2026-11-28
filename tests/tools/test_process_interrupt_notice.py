"""Owed death notices: a lifecycle kill must never silently disappear (agent_close bug).

A background process killed by Hermes lifecycle cleanup (``agent_close`` / ``kill_all`` /
``gateway_shutdown`` / ``gateway_restart``) — or found dead at boot — carried a completion
contract (``notify_on_complete`` / ``persist_on_release``). Its owner must be told
"task interrupted" with the kill cause and time through the same wake chain as a completion
notification, surviving a gateway restart (the 2026-11-12 incident: gateway restart killed
the Codex task at 15:34 and nobody knew until the user asked at 20:54).

Chain under test: kill -> receipt (``interrupt_notice_state="pending"``) ->
``arm_interrupt_notices`` (replay watcher with ``dead_session``) -> watcher delivery ->
``mark_interrupt_notice_delivered`` (no later boot replays).
"""

import json
import subprocess
import time
from unittest.mock import patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession
from tools.process_registry_results import (
    INTERRUPT_KILL_SOURCES,
    mark_interrupt_notice_delivered,
    needs_interrupt_notice,
    pending_interrupt_notices,
    save_completed_result,
)


@pytest.fixture()
def registry():
    return ProcessRegistry()


@pytest.fixture()
def receipts_home(tmp_path, monkeypatch):
    """Isolate the durable receipt ledger (never touch the live profile's)."""
    from tools import process_registry_results as receipts
    monkeypatch.setattr(receipts, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def _receipt(receipts_home, session_id):
    return json.loads(
        (receipts_home / "logs" / "process-results" / f"{session_id}.json").read_text(encoding="utf-8-sig"))


def _make_session(sid="proc_intr", *, notify=True, persist=False, **kw) -> ProcessSession:
    return ProcessSession(
        id=sid, command="zsh -ic 'codex exec long-task'", task_id="t1",
        started_at=time.time() - 60, notify_on_complete=notify, persist_on_release=persist,
        session_key="agent:main:feishu:forum:oc_x:omt_y",
        parent_session_id="20261110_161920_962e",
        watcher_platform="feishu", watcher_chat_id="oc_x", watcher_thread_id="omt_y",
        **kw)


# =========================================================================
# The owed-notice predicate
# =========================================================================

class TestNeedsInterruptNotice:
    def test_lifecycle_kill_of_contract_holder_owes_notice(self):
        for source in sorted(INTERRUPT_KILL_SOURCES):
            s = _make_session(completion_reason="killed", termination_source=source)
            assert needs_interrupt_notice(s), source

    def test_deliberate_stops_owe_no_notice(self):
        # Explicit kill (the caller sees the output) and abandoned-turn reaping
        # (deliberate stop: "a killed abandoned process can't revive stopped work").
        s = _make_session(completion_reason="killed", termination_source="process.kill")
        assert not needs_interrupt_notice(s)
        s = _make_session(completion_reason="killed", termination_source="gateway_turn_timeout")
        assert not needs_interrupt_notice(s)

    def test_silent_process_owes_no_notice(self):
        s = _make_session(notify=False, persist=False,
                          completion_reason="killed", termination_source="agent_close")
        assert not needs_interrupt_notice(s)

    def test_downtime_loss_of_contract_holder_owes_notice(self):
        s = _make_session(completion_reason="lost", termination_source="gateway_downtime")
        assert needs_interrupt_notice(s)
        s = _make_session(notify=False, completion_reason="lost", termination_source="gateway_downtime")
        assert not needs_interrupt_notice(s)

    def test_normal_exit_owes_no_notice(self):
        assert not needs_interrupt_notice(_make_session(completion_reason="exited"))


# =========================================================================
# Kill -> durable pending receipt (the 2026-11-12 incident shape)
# =========================================================================

class TestKillRecordsPendingNotice:
    pytestmark = pytest.mark.platforms("posix")  # spawns a real `sleep` child

    def _kill_real(self, registry, receipts_home, session):
        proc = subprocess.Popen(["sleep", "60"])
        session.process, session.pid = proc, proc.pid
        registry._running[session.id] = session
        with patch.object(registry, "_write_checkpoint"):
            result = registry.kill_process(session.id, source="agent_close", consume_output=True)
        assert result["status"] == "killed"
        return result

    def test_agent_close_kill_writes_pending_notice(self, registry, receipts_home):
        session = _make_session()
        self._kill_real(registry, receipts_home, session)
        record = _receipt(receipts_home, session.id)
        assert record["interrupt_notice_state"] == "pending"
        assert record["completion_reason"] == "killed"
        assert record["termination_source"] == "agent_close"
        assert record["exit_code"] == -15
        assert record["exited_at"] > 0
        assert record["notify_on_complete"] is True
        # Route fields survive for the post-restart replay.
        assert record["watcher_platform"] == "feishu"
        assert record["watcher_thread_id"] == "omt_y"

    def test_forced_consume_still_owes_notice(self, registry, receipts_home):
        """consume_output=True (the agent_close sweep) marks the completion consumed —
        that must NOT cancel the owed death notice (the mark means the killer took the
        output, not that the agent learned the task died)."""
        session = _make_session()
        self._kill_real(registry, receipts_home, session)
        assert registry.is_completion_consumed(session.id)
        assert _receipt(receipts_home, session.id)["interrupt_notice_state"] == "pending"

    def test_explicit_kill_writes_no_pending_notice(self, registry, receipts_home):
        session = _make_session()
        proc = subprocess.Popen(["sleep", "60"])
        session.process, session.pid = proc, proc.pid
        registry._running[session.id] = session
        with patch.object(registry, "_write_checkpoint"):
            registry.kill_process(session.id, source="process.kill", consume_output=True)
        assert _receipt(receipts_home, session.id)["interrupt_notice_state"] == ""

    def test_persist_job_killed_at_host_shutdown_owes_notice(self, registry, receipts_home):
        session = _make_session(notify=False, persist=True)
        proc = subprocess.Popen(["sleep", "60"])
        session.process, session.pid = proc, proc.pid
        registry._running[session.id] = session
        with patch.object(registry, "_write_checkpoint"):
            registry.kill_process(session.id, source="gateway_shutdown", consume_output=False)
        record = _receipt(receipts_home, session.id)
        assert record["interrupt_notice_state"] == "pending"
        assert record["persist_on_release"] is True


# =========================================================================
# Notice text: task interrupted + kill cause + time
# =========================================================================

class TestNoticeText:
    def test_lifecycle_kill_text_names_cause_and_time(self):
        from tools.process_registry_notifications import format_process_notification
        killed_at = time.mktime(time.strptime("2026-11-12 15:34:50", "%Y-%m-%d %H:%M:%S"))
        text = format_process_notification({
            "type": "completion", "session_id": "proc_abc", "command": "codex exec long-task",
            "exit_code": -15, "completion_reason": "killed", "termination_source": "agent_close",
            "exited_at": killed_at, "output": "partial work",
        })
        assert "task interrupted" in text
        assert "agent_close" in text
        assert "2026-11-12 15:34:50" in text
        assert "SIGTERM" in text

    def test_downtime_loss_text_says_outcome_unknown(self):
        from tools.process_registry_notifications import format_process_notification
        text = format_process_notification({
            "type": "completion", "session_id": "proc_abc", "command": "codex exec long-task",
            "exit_code": None, "completion_reason": "lost", "termination_source": "gateway_downtime",
            "output": "",
        })
        assert "task interrupted" in text
        assert "outcome unknown" in text
        assert "none captured" in text

    def test_plain_completion_text_unchanged(self):
        from tools.process_registry_notifications import format_process_notification
        text = format_process_notification({
            "type": "completion", "session_id": "proc_abc", "command": "echo done",
            "exit_code": 0, "output": "done",
        })
        assert "completed normally (exit code 0)" in text
        assert "task interrupted" not in text


# =========================================================================
# Replay: kill while the gateway dies -> boot arms a death-notice watcher
# =========================================================================

class TestReplayAfterRestart:
    def _pending_receipt_record(self, receipts_home, **kw):
        session = _make_session(**kw)
        session.exited = True
        session.exit_code = -15
        session.completion_reason = "killed"
        session.termination_source = "agent_close"
        session.exited_at = time.time()
        with patch("tools.process_registry.ProcessRegistry._write_checkpoint"), \
             patch("hermes_constants.assert_named_profile_home_live"):
            save_completed_result(session)
        return session

    def test_boot_replay_arms_watcher_with_dead_session(self, registry, receipts_home):
        session = self._pending_receipt_record(receipts_home)
        armed = registry.arm_interrupt_notices()
        assert armed == 1
        watcher = registry.pending_watchers[-1]
        assert watcher["session_id"] == session.id
        assert watcher["interrupted_death"] is True
        assert watcher["notify_on_complete"] is True
        dead = watcher["dead_session"]
        assert dead.exited is True
        assert dead.completion_reason == "killed"
        assert dead.termination_source == "agent_close"
        # Route fields ride the descriptor for the wake chain.
        assert watcher["platform"] == "feishu"
        assert watcher["thread_id"] == "omt_y"

    def test_replay_is_idempotent_and_honors_delivered_stamp(self, registry, receipts_home):
        session = self._pending_receipt_record(receipts_home)
        assert registry.arm_interrupt_notices() == 1
        assert registry.arm_interrupt_notices(rescan=True) == 0  # dedup by session id
        assert mark_interrupt_notice_delivered(session.id) is True
        assert _receipt(receipts_home, session.id)["interrupt_notice_state"] == "delivered"
        assert mark_interrupt_notice_delivered(session.id) is False  # idempotent stamp
        fresh = ProcessRegistry()
        assert fresh.arm_interrupt_notices() == 0  # delivered: never replayed again
        assert pending_interrupt_notices() == []

    def test_recovery_records_downtime_death_and_replays(self, registry, receipts_home, tmp_path):
        """Simulate restart: a checkpoint entry whose PID is gone when the gateway comes
        back (killed while down / died in the gap) — death must be perceivable."""
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_gap",
            "command": "codex exec long-task",
            "pid": 999999999,  # almost certainly not running
            "pid_scope": "host",
            "task_id": "t1",
            "owner_task_id": "t1",
            "session_key": "agent:main:feishu:forum:oc_x:omt_y",
            "parent_session_id": "20261110_161920_962e",
            "started_at": time.time() - 3600,
            "notify_on_complete": True,
            "watcher_platform": "feishu",
            "watcher_chat_id": "oc_x",
            "watcher_thread_id": "omt_y",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
        assert recovered == 0  # not alive: nothing to adopt
        record = _receipt(receipts_home, "proc_gap")
        assert record["interrupt_notice_state"] == "pending"
        assert record["completion_reason"] == "lost"
        assert record["termination_source"] == "gateway_downtime"
        watcher = registry.pending_watchers[-1]
        assert watcher["session_id"] == "proc_gap"
        assert watcher["interrupted_death"] is True

    def test_recovery_leaves_silent_dead_entries_alone(self, registry, receipts_home, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_silent",
            "command": "sleep 999",
            "pid": 999999999,
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            assert registry.recover_from_checkpoint() == 0
        assert not (receipts_home / "logs" / "process-results" / "proc_silent.json").exists()
        assert registry.pending_watchers == []

    def test_multiplex_recovery_scans_each_profile_without_a_checkpoint(
        self, registry, receipts_home, tmp_path, monkeypatch
    ):
        """One shared registry must replay each served profile's owed death notice."""
        from tools import process_registry as registry_module
        from tools import process_registry_results as receipts

        secondary_home = tmp_path / "secondary"
        active_home = [receipts_home]
        monkeypatch.setattr(registry_module, "get_hermes_home", lambda: active_home[0])
        monkeypatch.setattr(receipts, "get_hermes_home", lambda: active_home[0])

        # Two profile-local ledgers can contain the same session ID. Neither
        # has a processes.json: the notice receipt must suffice at startup.
        with patch("hermes_constants.assert_named_profile_home_live"):
            for home, profile in ((receipts_home, "main"), (secondary_home, "work")):
                active_home[0] = home
                session = _make_session()
                session.session_key = f"agent:{profile}:feishu:forum:oc_x:omt_y"
                session.exited = True
                session.exit_code = -15
                session.completion_reason = "killed"
                session.termination_source = "agent_close"
                session.exited_at = time.time()
                save_completed_result(session)

        active_home[0] = receipts_home
        assert registry.recover_from_checkpoint() == 0
        assert len(registry.pending_watchers) == 1

        active_home[0] = secondary_home
        assert registry.recover_from_checkpoint() == 0
        assert [w["dead_session"].session_key for w in registry.pending_watchers] == [
            "agent:main:feishu:forum:oc_x:omt_y",
            "agent:work:feishu:forum:oc_x:omt_y",
        ]
        assert registry.arm_interrupt_notices() == 0  # repeat scan is idempotent

# =========================================================================
# persist_on_release across a gateway restart: keep alive when possible
# =========================================================================

class TestRestartKeepsPersistedJobs:
    def test_gateway_restart_sweep_skips_persisted_jobs(self, registry):
        volatile = _make_session(sid="proc_volatile", persist=False)
        persisted = _make_session(sid="proc_persisted", persist=True)
        registry._running[volatile.id] = volatile
        registry._running[persisted.id] = persisted
        calls = []

        def fake_kill(session_id, **kwargs):
            calls.append((session_id, kwargs.get("source")))
            registry._running[session_id].exited = True
            return {"status": "killed"}

        registry.kill_process = fake_kill
        assert registry.kill_all(None, source="gateway_restart") == 1
        assert [c[0] for c in calls] == ["proc_volatile"]
        assert calls[0][1] == "gateway_restart"
        assert persisted.exited is False

    def test_host_shutdown_still_kills_persisted_jobs(self, registry):
        persisted = _make_session(sid="proc_persisted", persist=True)
        registry._running[persisted.id] = persisted
        calls = []
        registry.kill_process = lambda session_id, **kw: (
            calls.append(session_id) or {"status": "killed"})
        assert registry.kill_all(None, source="gateway_shutdown") == 1
        assert calls == ["proc_persisted"]

    def test_gateway_restart_is_a_lifecycle_source(self):
        assert "gateway_restart" in ProcessRegistry._LIFECYCLE_KILL_SOURCES
        assert "gateway_restart" in INTERRUPT_KILL_SOURCES
