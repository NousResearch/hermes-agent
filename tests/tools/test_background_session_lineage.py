"""Every background job records the conversation that spawned it.

``ProcessSession.parent_session_id`` is the durable session-db id of the spawning
chat. It is what lets another surface (Hermes Desktop on the same profile) match a
``terminal(background=true)`` job started from Telegram back to the conversation the
user is looking at, and it is what lets the gateway drop a completion whose session
was closed at a ``/new`` boundary.

It used to be stamped only on the way to configuring an async notification, so the
DEFAULT launch — plain ``terminal(background=true)``, no ``notify_on_complete``, no
``watch_patterns`` — left the field empty and the job was unattributable.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

import gateway.session_context as session_context
import tools.process_registry as process_registry_module
from tools.process_registry import ProcessSession
from tools.terminal_tool_background import spawn_background_process

DURABLE_ID = "20260901_120000_abcdef"

_SESSION_ENV = {
    "HERMES_SESSION_PLATFORM": "telegram",
    "HERMES_SESSION_CHAT_ID": "4242",
    "HERMES_SESSION_USER_ID": "u-1",
    "HERMES_SESSION_USER_NAME": "ada",
    "HERMES_SESSION_THREAD_ID": "",
    "HERMES_SESSION_MESSAGE_ID": "m-9",
    "HERMES_SESSION_ID": DURABLE_ID,
}


class _QuietPopen:
    """A child that starts and then says nothing — the shape this bug needs."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.pid = 424242
        self.stdout = None
        self.stdin = None
        self.returncode = None

    def poll(self):
        return None


class _Registry:
    """Stand-in for the process registry: records the session it was handed."""

    def __init__(self) -> None:
        self.pending_watchers: list[dict[str, Any]] = []
        self.spawned: ProcessSession | None = None

    def spawn_local(self, *, command, cwd, task_id, owner_task_id, session_key, **_kw) -> ProcessSession:
        self.spawned = ProcessSession(
            id="proc_spawned1234", command=command, cwd=cwd, task_id=task_id,
            owner_task_id=owner_task_id, session_key=session_key, pid=4242, started_at=time.time())
        return self.spawned


@pytest.fixture()
def gateway_session(monkeypatch, tmp_path):
    """A live messaging session: routing metadata bound, async delivery available."""
    registry = _Registry()
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(
        session_context, "get_session_env", lambda name, default="": _SESSION_ENV.get(name, default))
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)
    return registry


def _spawn(tmp_path, **overrides) -> dict:
    kwargs = dict(
        command="npm run dev", env=None, env_type="local", effective_task_id="t-telegram",
        task_id="t-telegram", session_key="telegram:4242", workdir=None, cwd=str(tmp_path),
        effective_pty=False, notify_on_complete=False, watch_patterns=None,
        approval_note=None, pty_disabled_reason=None)
    kwargs.update(overrides)
    return json.loads(spawn_background_process(**kwargs))


class TestBackgroundSessionLineage:
    def test_silent_default_launch_records_the_spawning_conversation(self, gateway_session, tmp_path):
        """The reported bug: no notification flags, so nothing stamped the lineage."""
        result = _spawn(tmp_path)

        assert result["exit_code"] == 0
        assert gateway_session.spawned.parent_session_id == DURABLE_ID

    def test_a_silent_job_still_asks_for_no_notification(self, gateway_session, tmp_path):
        result = _spawn(tmp_path)

        # Stamping identity must not quietly turn a silent job into a notifying one.
        assert "notify_on_complete" not in result
        assert gateway_session.spawned.notify_on_complete is False
        assert gateway_session.spawned.watch_patterns == []
        assert gateway_session.pending_watchers == []

    def test_notifying_jobs_keep_their_routing_and_watcher(self, gateway_session, tmp_path):
        result = _spawn(tmp_path, notify_on_complete=True)

        assert result["notify_on_complete"] is True
        session = gateway_session.spawned
        assert session.parent_session_id == DURABLE_ID
        assert (session.watcher_platform, session.watcher_chat_id) == ("telegram", "4242")
        assert session.watcher_message_id == "m-9"
        assert [w["session_id"] for w in gateway_session.pending_watchers] == ["proc_spawned1234"]

    def test_lineage_survives_a_channel_that_cannot_deliver_async(
        self, gateway_session, tmp_path, monkeypatch):
        """A stateless runner loses the notification, not its identity."""
        monkeypatch.setattr(session_context, "async_delivery_supported", lambda: False)

        result = _spawn(tmp_path, notify_on_complete=True)

        assert result["notify_on_complete"] is False
        assert "notify_unsupported" in result
        assert gateway_session.spawned.parent_session_id == DURABLE_ID

    def test_a_quiet_job_publishes_its_lineage_to_the_shared_checkpoint(
        self, monkeypatch, tmp_path):
        """The checkpoint row is written when the process is REGISTERED, which happens
        before the spawn caller stamps anything on it. A quiet job never emits output,
        so nothing rewrites that row for the rest of its life and the durable id another
        gateway matches on is missing from the only file both processes can see.
        """
        registry = process_registry_module.ProcessRegistry()
        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(process_registry_module, "process_registry", registry)
        monkeypatch.setattr(process_registry_module, "CHECKPOINT_PATH", checkpoint)
        monkeypatch.setattr(
            session_context, "get_session_env", lambda name, default="": _SESSION_ENV.get(name, default))
        monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)
        # A real registration through spawn_local, minus the child and its reader:
        # the session stays running and quiet, exactly like the reported case.
        monkeypatch.setattr(process_registry_module, "_find_shell", lambda: "/bin/sh")
        monkeypatch.setattr(process_registry_module.subprocess, "Popen", _QuietPopen)
        monkeypatch.setattr(
            process_registry_module.ProcessRegistry, "_reader_loop", lambda self, session: None)

        result = _spawn(tmp_path)

        published = json.loads(checkpoint.read_text(encoding="utf-8"))
        row = next(r for r in published if r["session_id"] == result["session_id"])
        assert row["parent_session_id"] == DURABLE_ID

    def test_a_quiet_notifying_job_publishes_its_routing_too(self, monkeypatch, tmp_path):
        """Same staleness hit the watcher routing and the notification flags, which are
        also stamped after registration and also live in the checkpoint."""
        registry = process_registry_module.ProcessRegistry()
        checkpoint = tmp_path / "processes.json"
        monkeypatch.setattr(process_registry_module, "process_registry", registry)
        monkeypatch.setattr(process_registry_module, "CHECKPOINT_PATH", checkpoint)
        monkeypatch.setattr(
            session_context, "get_session_env", lambda name, default="": _SESSION_ENV.get(name, default))
        monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)
        monkeypatch.setattr(process_registry_module, "_find_shell", lambda: "/bin/sh")
        monkeypatch.setattr(process_registry_module.subprocess, "Popen", _QuietPopen)
        monkeypatch.setattr(
            process_registry_module.ProcessRegistry, "_reader_loop", lambda self, session: None)

        result = _spawn(tmp_path, notify_on_complete=True)

        published = json.loads(checkpoint.read_text(encoding="utf-8"))
        row = next(r for r in published if r["session_id"] == result["session_id"])
        assert row["notify_on_complete"] is True
        assert row["watcher_platform"] == "telegram"
        assert row["watcher_chat_id"] == "4242"

    def test_a_session_with_no_conversation_id_stamps_nothing(
        self, gateway_session, tmp_path, monkeypatch):
        monkeypatch.setattr(session_context, "get_session_env", lambda name, default="": default)

        _spawn(tmp_path)

        # Empty is the honest answer; a placeholder would match other blank rows.
        assert gateway_session.spawned.parent_session_id == ""
        assert gateway_session.spawned.watcher_platform == ""
