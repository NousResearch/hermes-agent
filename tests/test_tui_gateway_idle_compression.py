import os
import threading
import types

import pytest

from hermes_constants import (
    get_hermes_home,
    get_hermes_home_override,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tui_gateway import server


def _history():
    return [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "last_active": 0.0,
        "transport": None,
        "attached_images": [],
        "cols": 80,
        "slash_worker": None,
        **extra,
    }


def _idle_cfg():
    return {
        "enabled": True,
        "threshold": 0.5,
        "idle_after_seconds": 10.0,
        "min_interval_seconds": 60.0,
        "emit_status": False,
    }


def _configure_ready_idle_pass(monkeypatch):
    monkeypatch.setattr(server, "_load_idle_compression_config", _idle_cfg)
    monkeypatch.setattr(server.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        "agent.model_metadata.estimate_request_tokens_rough",
        lambda messages, system_prompt="", tools=None: 600,
    )
    monkeypatch.setattr(
        server, "_session_compression_threshold_tokens", lambda _agent, _cfg: 500
    )


def _run_in_real_thread(target):
    result = []
    errors = []

    def run():
        try:
            result.append(target())
        except BaseException as exc:  # surface worker failures in the test thread
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    return result[0]


class _SyncLease:
    def __init__(self):
        self.release_calls = 0

    def release(self):
        self.release_calls += 1


class _ContendedLock:
    """A Lock-compatible Event probe for deterministic finalize handoff tests."""

    def __init__(self):
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def acquire(self, *args, **kwargs):
        if self._lock.locked():
            self.contended.set()
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


def _spy_session_key_approval_effects(monkeypatch):
    from tools import approval

    calls = []
    monkeypatch.setattr(
        approval,
        "is_session_yolo_enabled",
        lambda key: calls.append(("is_yolo", key)) or key == "old-key",
    )
    monkeypatch.setattr(
        approval,
        "enable_session_yolo",
        lambda key: calls.append(("enable_yolo", key)),
    )
    monkeypatch.setattr(
        approval,
        "disable_session_yolo",
        lambda key: calls.append(("disable_yolo", key)),
    )
    monkeypatch.setattr(
        approval,
        "register_gateway_notify",
        lambda key, _callback: calls.append(("register_notify", key)),
    )
    monkeypatch.setattr(
        approval,
        "unregister_gateway_notify",
        lambda key: calls.append(("unregister_notify", key)),
    )
    return calls


def _write_idle_config(home, *, enabled):
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "compression:\n"
        "  enabled: true\n"
        "  threshold: 0.5\n"
        "  idle:\n"
        f"    enabled: {'true' if enabled else 'false'}\n"
        "    idle_after_seconds: 10\n"
        "    min_interval_seconds: 60\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("default_enabled", "named_enabled", "expected_calls"),
    [(False, True, 1), (True, False, 0)],
)
def test_idle_worker_real_thread_uses_named_profile_both_directions(
    monkeypatch, tmp_path, default_enabled, named_enabled, expected_calls
):
    default_home = tmp_path / "default"
    named_home = tmp_path / "profiles" / "research"
    _write_idle_config(default_home, enabled=default_enabled)
    _write_idle_config(named_home, enabled=named_enabled)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(server, "_hermes_home", default_home)
    monkeypatch.setattr(
        "agent.model_metadata.estimate_request_tokens_rough",
        lambda messages, system_prompt="", tools=None: 600,
    )
    calls = []

    def compress(*_args, **_kwargs):
        calls.append((get_hermes_home(), get_hermes_home_override()))
        return 0, {}

    monkeypatch.setattr(server, "_compress_session_history", compress)
    session = _session(
        sid="profile-sid",
        agent=types.SimpleNamespace(
            session_id="session-key",
            context_compressor=types.SimpleNamespace(context_length=1000),
        ),
        history=_history(),
        profile_home=str(named_home),
    )
    server._sessions["profile-sid"] = session
    server._cfg_cache = None
    server._cfg_mtime = None
    server._cfg_path = None
    parent_token = set_hermes_home_override(default_home)
    env_before = os.environ.get("HERMES_HOME")
    thread_scope = {}

    def worker_call():
        thread_scope["before"] = get_hermes_home()
        result = server._run_idle_compression_once("profile-sid", session)
        thread_scope["after"] = get_hermes_home()
        return result

    try:
        assert _run_in_real_thread(worker_call) is False
        assert len(calls) == expected_calls
        if calls:
            assert calls == [(named_home, str(named_home))]
        assert thread_scope == {"before": default_home, "after": default_home}
        assert get_hermes_home_override() == str(default_home)
        assert os.environ.get("HERMES_HOME") == env_before
    finally:
        reset_hermes_home_override(parent_token)
        server._sessions.pop("profile-sid", None)
        server._cfg_cache = None
        server._cfg_mtime = None
        server._cfg_path = None


def test_schedule_clears_ambient_profile_for_default_session(monkeypatch, tmp_path):
    default_home = tmp_path / "default"
    named_home = tmp_path / "profiles" / "disabled"
    _write_idle_config(default_home, enabled=True)
    _write_idle_config(named_home, enabled=False)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(server, "_hermes_home", default_home)
    server._cfg_cache = None
    server._cfg_mtime = None
    server._cfg_path = None

    class FakeThread:
        created = []

        def __init__(self, target, **_kwargs):
            self.target = target
            self.__class__.created.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    monkeypatch.setattr(server.threading, "Thread", FakeThread)
    session = _session(sid="default-sid", profile_home=None)
    server._sessions["default-sid"] = session
    parent_token = set_hermes_home_override(named_home)
    env_before = os.environ.get("HERMES_HOME")
    try:
        server._schedule_idle_compression("default-sid", session)
        assert len(FakeThread.created) == 1
        assert get_hermes_home_override() == str(named_home)
        assert os.environ.get("HERMES_HOME") == env_before
    finally:
        reset_hermes_home_override(parent_token)
        server._sessions.pop("default-sid", None)
        server._IDLE_COMPRESSION_THREADS.pop("default-sid", None)
        server._cfg_cache = None
        server._cfg_mtime = None
        server._cfg_path = None


def test_schedule_registry_backed_deferred_record_without_sid(monkeypatch):
    class FakeThread:
        created = []

        def __init__(self, target, **_kwargs):
            self.target = target
            self.__class__.created.append(self)

        def start(self):
            return None

        def is_alive(self):
            return True

    monkeypatch.setattr(server.threading, "Thread", FakeThread)
    monkeypatch.setattr(server, "_load_idle_compression_config", _idle_cfg)
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "all")
    session = server._deferred_session_record(
        "session-key",
        cols=80,
        cwd="/tmp",
        history=[],
        lease=None,
    )
    assert "sid" not in session
    assert "_sid" not in session
    server._sessions["deferred-sid"] = session

    try:
        server._schedule_idle_compression_scoped("deferred-sid", session)
        assert len(FakeThread.created) == 1
        assert "deferred-sid" in server._IDLE_COMPRESSION_THREADS
    finally:
        server._sessions.pop("deferred-sid", None)
        server._IDLE_COMPRESSION_THREADS.pop("deferred-sid", None)


@pytest.mark.parametrize("mode", ["queue", "steer", "interrupt"])
def test_prompt_submit_queues_during_idle_compression_without_control_calls(
    monkeypatch, mode
):
    calls = []
    agent = types.SimpleNamespace(
        steer=lambda text: calls.append(("steer", text)) or True,
        interrupt=lambda: calls.append(("interrupt", None)),
    )
    transport = object()
    session = _session(
        agent=agent,
        idle_compression_running=True,
        transport=transport,
    )
    server._sessions["busy-sid"] = session
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: mode)
    try:
        response = server.handle_request(
            {
                "id": "1",
                "method": "prompt.submit",
                "params": {"session_id": "busy-sid", "text": "do not lose me"},
            }
        )
        assert response["result"]["status"] == "queued"
        assert session["queued_prompt"] == {
            "text": "do not lose me",
            "transport": transport,
        }
        assert session["running"] is False
        assert "idle_compression_cancel_requested" not in session
        assert calls == []
    finally:
        server._sessions.pop("busy-sid", None)


def test_two_prompts_merge_then_compression_commits_syncs_and_drains_once(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    original = _history()
    sequence = []

    class BlockingAgent:
        session_id = "old-key"
        context_compressor = types.SimpleNamespace(context_length=1000)

        def _compress_context(self, history, system_message, **_kwargs):
            assert session["history_lock"].acquire(blocking=False)
            session["history_lock"].release()
            started.set()
            assert release.wait(timeout=5)
            self.session_id = "continuation-key"
            return history[:2], {}

    session = _session(
        sid="commit-sid",
        agent=BlockingAgent(),
        session_key="old-key",
        history=list(original),
    )
    server._sessions["commit-sid"] = session
    _configure_ready_idle_pass(monkeypatch)

    def sync(sid, current, **_kwargs):
        assert sid == "commit-sid"
        assert current["history"] == original[:2]
        assert current["history_version"] == 1
        assert current["history_lock"].acquire(blocking=False)
        current["history_lock"].release()
        current["session_key"] = current["agent"].session_id
        sequence.append("sync")

    def drain(_rid, sid, current):
        assert sid == "commit-sid"
        assert current["idle_compression_running"] is False
        assert current["session_key"] == "continuation-key"
        assert current["history"] == original[:2]
        assert current["history_lock"].acquire(blocking=False)
        current["history_lock"].release()
        queued = current.pop("queued_prompt", None)
        sequence.append(("drain", queued))
        return queued is not None

    monkeypatch.setattr(server, "_sync_session_key_after_compress", sync)
    monkeypatch.setattr(server, "_drain_queued_prompt", drain)
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            server._run_idle_compression_once("commit-sid", session)
        )
    )
    worker.start()
    assert started.wait(timeout=5)
    assert server._handle_busy_submit("1", "commit-sid", session, "first", "ws-1")[
        "result"
    ]["status"] == "queued"
    assert server._handle_busy_submit("2", "commit-sid", session, "second", "ws-2")[
        "result"
    ]["status"] == "queued"
    assert session["queued_prompt"] == {
        "text": "first\n\nsecond",
        "transport": "ws-2",
    }
    assert "idle_compression_cancel_requested" not in session
    release.set()
    worker.join(timeout=5)
    try:
        assert not worker.is_alive()
        assert result == [True]
        assert sequence == [
            "sync",
            (
                "drain",
                {"text": "first\n\nsecond", "transport": "ws-2"},
            ),
        ]
        assert session["history"] == original[:2]
        assert session["history_version"] == 1
        assert session["session_key"] == session["agent"].session_id
    finally:
        release.set()
        server._sessions.pop("commit-sid", None)


@pytest.mark.parametrize("outcome", ["success", "noop", "exception"])
def test_idle_cleanup_releases_semaphore_and_drains_live_queue(monkeypatch, outcome):
    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(server, "_IDLE_COMPRESSION_GLOBAL_SEMAPHORE", semaphore)
    _configure_ready_idle_pass(monkeypatch)
    session = _session(
        sid="cleanup-sid",
        agent=types.SimpleNamespace(session_id="session-key"),
        history=_history(),
        queued_prompt={"text": "next", "transport": None},
    )
    server._sessions["cleanup-sid"] = session
    drained = []

    def compress(*_args, **_kwargs):
        if outcome == "exception":
            raise RuntimeError("compress failed")
        return (2 if outcome == "success" else 0), {}

    def drain(_rid, _sid, current):
        assert current["idle_compression_running"] is False
        drained.append(current["queued_prompt"]["text"])
        return True

    monkeypatch.setattr(server, "_compress_session_history", compress)
    monkeypatch.setattr(server, "_drain_queued_prompt", drain)
    try:
        assert server._run_idle_compression_once("cleanup-sid", session) is (
            outcome == "success"
        )
        assert session["idle_compression_running"] is False
        assert drained == ["next"]
        assert semaphore.acquire(blocking=False)
        semaphore.release()
    finally:
        server._sessions.pop("cleanup-sid", None)


@pytest.mark.parametrize(
    ("fence", "expect_drain"),
    [
        ("history_version", True),
        ("normal_turn", False),
        ("finalized", False),
        ("replacement", False),
    ],
)
def test_idle_result_fences_stale_or_non_live_session(
    monkeypatch, fence, expect_drain
):
    started = threading.Event()
    release = threading.Event()
    original = _history()

    class BlockingAgent:
        session_id = "session-key"
        context_compressor = types.SimpleNamespace(context_length=1000)

        def _compress_context(self, history, system_message, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return history[:2], {}

    session = _session(
        sid="fence-sid",
        agent=BlockingAgent(),
        history=list(original),
        history_version=7,
        queued_prompt={"text": "queued", "transport": None},
    )
    server._sessions["fence-sid"] = session
    _configure_ready_idle_pass(monkeypatch)
    synced = []
    drained = []
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_a, **_k: synced.append(True)
    )
    monkeypatch.setattr(
        server, "_drain_queued_prompt", lambda *_a, **_k: drained.append(True) or True
    )
    worker = threading.Thread(
        target=server._run_idle_compression_once,
        args=("fence-sid", session),
    )
    worker.start()
    assert started.wait(timeout=5)
    with session["history_lock"]:
        if fence == "history_version":
            session["history_version"] += 1
        elif fence == "normal_turn":
            session["running"] = True
        elif fence == "finalized":
            session["_finalized"] = True
    if fence == "replacement":
        server._sessions["fence-sid"] = _session(sid="fence-sid")
    release.set()
    worker.join(timeout=5)
    try:
        assert not worker.is_alive()
        assert session["history"] == original
        assert session["history_version"] == (8 if fence == "history_version" else 7)
        assert synced == []
        assert drained == ([True] if expect_drain else [])
        assert session["idle_compression_running"] is False
    finally:
        release.set()
        server._sessions.pop("fence-sid", None)


def test_replaced_registry_record_without_sid_cannot_publish_or_drain(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    original = _history()

    class BlockingAgent:
        session_id = "session-key"
        context_compressor = types.SimpleNamespace(context_length=1000)

        def _compress_context(self, history, system_message, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return history[:2], {}

    session = _session(
        agent=BlockingAgent(),
        history=list(original),
        queued_prompt={"text": "never drain", "transport": None},
    )
    assert "sid" not in session
    assert "_sid" not in session
    server._sessions["replacement-sid"] = session
    _configure_ready_idle_pass(monkeypatch)
    synced = []
    drained = []
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_a, **_k: synced.append(True)
    )
    monkeypatch.setattr(
        server, "_drain_queued_prompt", lambda *_a, **_k: drained.append(True) or True
    )
    worker = threading.Thread(
        target=server._run_idle_compression_once,
        args=("replacement-sid", session),
    )
    worker.start()
    assert started.wait(timeout=5)
    server._sessions["replacement-sid"] = _session()
    live_after_replacement = server._session_has_live_identity(
        "replacement-sid", session
    )
    release.set()
    worker.join(timeout=5)

    try:
        assert not worker.is_alive()
        assert live_after_replacement is False
        assert session["history"] == original
        assert synced == []
        assert drained == []
    finally:
        release.set()
        server._sessions.pop("replacement-sid", None)


def test_finalize_during_compression_prevents_commit_sync_and_drain(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    original = _history()

    class BlockingAgent:
        session_id = "session-key"
        context_compressor = types.SimpleNamespace(context_length=1000)

        def _compress_context(self, history, system_message, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return history[:2], {}

    session = _session(
        sid="finalize-sid",
        agent=BlockingAgent(),
        history=list(original),
        queued_prompt={"text": "never run", "transport": None},
    )
    server._sessions["finalize-sid"] = session
    _configure_ready_idle_pass(monkeypatch)
    synced = []
    drained = []
    monkeypatch.setattr(server, "_release_active_session_slot", lambda *_a: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_a, **_k: synced.append(True)
    )
    monkeypatch.setattr(
        server, "_drain_queued_prompt", lambda *_a, **_k: drained.append(True) or True
    )
    worker = threading.Thread(
        target=server._run_idle_compression_once,
        args=("finalize-sid", session),
    )
    worker.start()
    assert started.wait(timeout=5)
    server._finalize_session(session)
    release.set()
    worker.join(timeout=5)
    try:
        assert not worker.is_alive()
        assert session["_finalized"] is True
        assert session["history"] == original
        assert session["history_version"] == 0
        assert synced == []
        assert drained == []
    finally:
        release.set()
        server._sessions.pop("finalize-sid", None)


def test_session_key_sync_finalize_during_transfer_aborts_without_side_effects(
    monkeypatch,
):
    sid = "sync-finalize-sid"
    entered = threading.Event()
    release = threading.Event()
    finalized_marked = threading.Event()
    sync_lock = _ContendedLock()
    lease = _SyncLease()
    restart_calls = []
    approval_calls = _spy_session_key_approval_effects(monkeypatch)

    class FinalizeObservedSession(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == "_finalized" and value is True:
                finalized_marked.set()

    session = FinalizeObservedSession(
        _session(
            sid=sid,
            agent=types.SimpleNamespace(
                session_id="new-key", model="test-model", platform="tui"
            ),
            session_key="old-key",
            active_session_lease=lease,
            _session_key_sync_lock=sync_lock,
        )
    )
    server._sessions[sid] = session

    def transfer(_sid, current, *, new_session_id):
        assert current is session
        assert new_session_id == "new-key"
        entered.set()
        assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(server, "_transfer_active_session_slot", transfer)
    monkeypatch.setattr(
        server,
        "_restart_slash_worker",
        lambda *_args, **_kwargs: restart_calls.append(True),
    )
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "tools.async_delegation.interrupt_for_session", lambda *_a, **_k: None
    )

    sync_errors = []
    finalize_errors = []

    def run_sync():
        try:
            server._sync_session_key_after_compress(sid, session)
        except BaseException as exc:
            sync_errors.append(exc)

    def run_finalize():
        try:
            server._finalize_session(session)
        except BaseException as exc:
            finalize_errors.append(exc)

    sync_thread = threading.Thread(target=run_sync)
    finalize_thread = threading.Thread(target=run_finalize)
    contended = False
    try:
        sync_thread.start()
        assert entered.wait(timeout=5)
        finalize_thread.start()
        assert finalized_marked.wait(timeout=5)
        assert session["_finalized"] is True
        contended = sync_lock.contended.wait(timeout=5)
    finally:
        release.set()
        sync_thread.join(timeout=5)
        finalize_thread.join(timeout=5)
        server._sessions.pop(sid, None)

    assert contended is True
    assert not sync_thread.is_alive()
    assert not finalize_thread.is_alive()
    assert sync_errors == []
    assert finalize_errors == []
    assert session["session_key"] == "old-key"
    assert restart_calls == []
    assert approval_calls == []
    assert lease.release_calls == 1
    assert "active_session_lease" not in session


def test_session_key_sync_replacement_during_transfer_aborts_without_side_effects(
    monkeypatch,
):
    sid = "sync-replacement-sid"
    entered = threading.Event()
    release = threading.Event()
    lease = _SyncLease()
    restart_calls = []
    approval_calls = _spy_session_key_approval_effects(monkeypatch)
    session = _session(
        sid=sid,
        agent=types.SimpleNamespace(session_id="new-key"),
        session_key="old-key",
        active_session_lease=lease,
    )
    replacement = _session(sid=sid)
    server._sessions[sid] = session

    def transfer(_sid, current, *, new_session_id):
        assert current is session
        assert new_session_id == "new-key"
        entered.set()
        assert release.wait(timeout=5)
        return True

    monkeypatch.setattr(server, "_transfer_active_session_slot", transfer)
    monkeypatch.setattr(
        server,
        "_restart_slash_worker",
        lambda *_args, **_kwargs: restart_calls.append(True),
    )
    errors = []

    def run_sync():
        try:
            server._sync_session_key_after_compress(sid, session)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_sync)
    try:
        worker.start()
        assert entered.wait(timeout=5)
        with server._sessions_lock:
            server._sessions[sid] = replacement
        release.set()
        worker.join(timeout=5)
    finally:
        release.set()
        worker.join(timeout=5)
        server._sessions.pop(sid, None)

    assert not worker.is_alive()
    assert errors == []
    assert session["session_key"] == "old-key"
    assert restart_calls == []
    assert approval_calls == []
    assert lease.release_calls == 1
    assert "active_session_lease" not in session


def test_session_key_sync_live_session_preserves_transfer_approval_and_worker(
    monkeypatch,
):
    sid = "sync-live-sid"
    lease = _SyncLease()
    transfer_calls = []
    restart_calls = []
    approval_calls = _spy_session_key_approval_effects(monkeypatch)
    session = _session(
        sid=sid,
        agent=types.SimpleNamespace(session_id="new-key"),
        session_key="old-key",
        pending_title="continuation title",
        active_session_lease=lease,
    )
    server._sessions[sid] = session

    def transfer(_sid, current, *, new_session_id):
        transfer_calls.append((_sid, current, new_session_id))
        return True

    monkeypatch.setattr(server, "_transfer_active_session_slot", transfer)
    monkeypatch.setattr(
        server,
        "_restart_slash_worker",
        lambda _sid, current: restart_calls.append(
            (_sid, current["session_key"])
        ),
    )
    try:
        server._sync_session_key_after_compress(sid, session)
    finally:
        server._sessions.pop(sid, None)

    assert transfer_calls == [(sid, session, "new-key")]
    assert session["session_key"] == "new-key"
    assert session["pending_title"] is None
    assert ("enable_yolo", "new-key") in approval_calls
    assert ("disable_yolo", "old-key") in approval_calls
    assert ("register_notify", "new-key") in approval_calls
    assert ("unregister_notify", "old-key") in approval_calls
    assert restart_calls == [(sid, "new-key")]
    assert lease.release_calls == 0


def test_finalize_popped_session_uses_teardown_sid_to_stop_idle_worker(monkeypatch):
    sid = "popped-sid"
    session = _session()
    session["agent"] = None
    session["session_key"] = ""
    server._sessions[sid] = session
    popped = server._pop_session_by_id(sid)
    assert popped is session
    assert session.get("_sid") == sid
    assert "sid" not in session
    assert "tui_session_id" not in session

    stop_event = threading.Event()
    server._IDLE_COMPRESSION_THREADS[sid] = (object(), stop_event)
    monkeypatch.setattr(server, "_release_active_session_slot", lambda *_a: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_a, **_k: None)

    try:
        server._finalize_session(session)
        assert stop_event.is_set()
        assert sid not in server._IDLE_COMPRESSION_THREADS
    finally:
        server._sessions.pop(sid, None)
        server._IDLE_COMPRESSION_THREADS.pop(sid, None)


def test_context_override_resets_and_semaphore_releases_after_exception(
    monkeypatch, tmp_path
):
    named_home = tmp_path / "profiles" / "broken"
    _write_idle_config(named_home, enabled=True)
    monkeypatch.setattr(
        "agent.model_metadata.estimate_request_tokens_rough",
        lambda messages, system_prompt="", tools=None: 600,
    )
    semaphore = threading.BoundedSemaphore(1)
    monkeypatch.setattr(server, "_IDLE_COMPRESSION_GLOBAL_SEMAPHORE", semaphore)
    seen = []

    def fail(*_args, **_kwargs):
        seen.append(get_hermes_home())
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "_compress_session_history", fail)
    session = _session(
        sid="exception-sid",
        agent=types.SimpleNamespace(
            session_id="session-key",
            context_compressor=types.SimpleNamespace(context_length=1000),
        ),
        history=_history(),
        profile_home=str(named_home),
    )
    server._sessions["exception-sid"] = session
    parent_home = get_hermes_home()
    thread_scope = {}

    def worker_call():
        thread_scope["before"] = get_hermes_home()
        result = server._run_idle_compression_once("exception-sid", session)
        thread_scope["after"] = get_hermes_home()
        return result

    try:
        assert _run_in_real_thread(worker_call) is False
        assert seen == [named_home]
        assert thread_scope == {"before": parent_home, "after": parent_home}
        assert session["idle_compression_running"] is False
        assert semaphore.acquire(blocking=False)
        semaphore.release()
    finally:
        server._sessions.pop("exception-sid", None)


def test_session_steer_never_targets_idle_compactor(monkeypatch):
    calls = []
    session = _session(
        agent=types.SimpleNamespace(steer=lambda text: calls.append(text) or True),
        idle_compression_running=True,
    )
    server._sessions["steer-sid"] = session
    try:
        response = server.handle_request(
            {
                "id": "1",
                "method": "session.steer",
                "params": {"session_id": "steer-sid", "text": "not the compactor"},
            }
        )
        assert "error" in response
        assert calls == []
    finally:
        server._sessions.pop("steer-sid", None)


def test_slash_mutation_is_blocked_during_idle_compression(monkeypatch):
    calls = []
    session = _session(
        agent=types.SimpleNamespace(),
        idle_compression_running=True,
    )
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda *_a, **_k: calls.append(True) or {},
    )
    result = server._mirror_slash_side_effects("slash-sid", session, "/model x")
    assert "session busy" in result
    assert calls == []


def test_idle_compression_skips_agentless_session(monkeypatch):
    _configure_ready_idle_pass(monkeypatch)
    session = _session(agent=None, history=_history())
    session["agent"] = None
    assert server._run_idle_compression_once("agentless-sid", session) is False
    assert session.get("idle_compression_running") is not True
