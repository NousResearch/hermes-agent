"""Tests for notify_on_complete background process feature.

Covers:
  - ProcessSession.notify_on_complete field
  - ProcessRegistry.completion_queue population on _move_to_finished()
  - Checkpoint persistence of notify_on_complete
  - Terminal tool schema includes notify_on_complete
  - Terminal tool handler passes notify_on_complete through
"""

import json
import os
import time
import pytest
from unittest.mock import MagicMock, patch

from tools.process_registry import (
    ProcessRegistry,
    ProcessSession,
)


@pytest.fixture()
def registry():
    """Create a fresh ProcessRegistry."""
    return ProcessRegistry()


def _make_session(
    sid="proc_test_notify",
    command="echo hello",
    task_id="t1",
    exited=False,
    exit_code=None,
    output="",
    notify_on_complete=False,
) -> ProcessSession:
    s = ProcessSession(
        id=sid,
        command=command,
        task_id=task_id,
        started_at=time.time(),
        exited=exited,
        exit_code=exit_code,
        output_buffer=output,
        notify_on_complete=notify_on_complete,
    )
    return s


# =========================================================================
# ProcessSession field
# =========================================================================

class TestProcessSessionField:
    def test_default_false(self):
        s = ProcessSession(id="proc_1", command="echo hi")
        assert s.notify_on_complete is False

    def test_set_true(self):
        s = ProcessSession(id="proc_1", command="echo hi", notify_on_complete=True)
        assert s.notify_on_complete is True


# =========================================================================
# Completion queue
# =========================================================================

class TestCompletionQueue:
    def test_queue_exists(self, registry):
        assert hasattr(registry, "completion_queue")
        assert registry.completion_queue.empty()


    def test_move_to_finished_idempotent_no_duplicate(self, registry):
        """Calling _move_to_finished twice must NOT enqueue two notifications.

        Regression test: kill_process() and the reader thread can both call
        _move_to_finished() for the same session, producing duplicate
        [SYSTEM: Background process ...] messages.
        """
        s = _make_session(notify_on_complete=True, output="done", exit_code=-15)
        s.exited = True
        s.exit_code = -15
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)  # first call — should enqueue
            s.exit_code = 143  # reader thread updates exit code
            registry._move_to_finished(s)  # second call — should be no-op

        assert registry.completion_queue.qsize() == 1
        completion = registry.completion_queue.get_nowait()
        assert completion["exit_code"] == -15  # from the first (kill) call


    def test_output_truncated_to_2000(self, registry):
        """Long output is truncated to last 2000 chars."""
        long_output = "x" * 5000
        s = _make_session(
            notify_on_complete=True,
            output=long_output,
        )
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        completion = registry.completion_queue.get_nowait()
        assert len(completion["output"]) == 2000

    def test_multiple_completions_queued(self, registry):
        """Multiple notify processes all push to the same queue."""
        for i in range(3):
            s = _make_session(
                sid=f"proc_{i}",
                notify_on_complete=True,
                output=f"output_{i}",
            )
            s.exited = True
            s.exit_code = 0
            registry._running[s.id] = s
            with patch.object(registry, "_write_checkpoint"):
                registry._move_to_finished(s)

        completions = []
        while not registry.completion_queue.empty():
            completions.append(registry.completion_queue.get_nowait())
        assert len(completions) == 3
        ids = {c["session_id"] for c in completions}
        assert ids == {"proc_0", "proc_1", "proc_2"}


# =========================================================================
# Checkpoint persistence
# =========================================================================

class TestCheckpointNotify:
    def test_checkpoint_includes_notify(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session(notify_on_complete=True)
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert len(data) == 1
            assert data[0]["notify_on_complete"] is True


    def test_recover_defaults_false(self, registry, tmp_path):
        """Old checkpoint entries without the field default to False."""
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            s = registry.get("proc_live")
            assert s.notify_on_complete is False


# =========================================================================
# Terminal tool schema
# =========================================================================

class TestTerminalSchema:
    def test_schema_advertises_unified_notify(self):
        """`notify` is the single advertised notification arg: bool (notify on
        exit) or list of strings (notify on pattern). The legacy
        notify_on_complete/watch_patterns args stay handler-accepted but
        unadvertised."""
        from tools.terminal_tool import TERMINAL_SCHEMA
        props = TERMINAL_SCHEMA["parameters"]["properties"]
        assert "notify" in props
        types = {alt["type"] for alt in props["notify"]["anyOf"]}
        assert types == {"boolean", "array"}
        assert "notify_on_complete" not in props
        assert "watch_patterns" not in props

    def test_handler_passes_notify(self):
        """_handle_terminal passes notify_on_complete to terminal_tool."""
        from tools.terminal_tool import _handle_terminal
        with patch("tools.terminal_tool.terminal_tool", return_value='{"ok":true}') as mock_tt:
            _handle_terminal(
                {"command": "echo hi", "background": True, "notify_on_complete": True},
                task_id="t1",
            )
            _, kwargs = mock_tt.call_args
            assert kwargs["notify_on_complete"] is True


# =========================================================================
# Code execution blocked params
# =========================================================================

class TestCodeExecutionBlocked:
    def test_notify_on_complete_blocked_in_sandbox(self):
        from tools.code_execution_rpc import _TERMINAL_BLOCKED_PARAMS
        assert "notify_on_complete" in _TERMINAL_BLOCKED_PARAMS


# =========================================================================
# Completion consumed suppression
# =========================================================================

class TestCompletionConsumed:
    """Test that wait/log consume completion notifications while poll stays read-only."""

    def test_wait_marks_completion_consumed(self, registry):
        """wait() returning exited status marks session as consumed."""
        s = _make_session(sid="proc_wait", notify_on_complete=True, output="done")
        s.exited = True
        s.exit_code = 0
        registry._running[s.id] = s
        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        # Notification is in the queue
        assert not registry.completion_queue.empty()
        assert not registry.is_completion_consumed("proc_wait")

        # Agent calls wait() — gets the result directly
        result = registry.wait("proc_wait", timeout=1)
        assert result["status"] == "exited"

        # Now the completion is marked as consumed
        assert registry.is_completion_consumed("proc_wait")


    def test_poll_observed_does_not_suppress_gateway_watcher(self, registry):
        """The gateway/tui watcher gate (is_completion_consumed) must stay False
        after a read-only poll, so the autonomous delivery turn still fires
        even though the CLI drain was deduped (#10156)."""
        s = _make_session(sid="proc_gw", notify_on_complete=True, output="done")
        s.exited = True
        s.exit_code = 0
        registry._finished[s.id] = s

        registry.poll("proc_gw")
        # CLI-side dedup signal present...
        assert "proc_gw" in registry._poll_observed
        # ...but the gateway watcher gate is untouched, so it still delivers.
        assert not registry.is_completion_consumed("proc_gw")

    def test_running_poll_does_not_mark_poll_observed(self, registry):
        """poll() on a still-running process must not record _poll_observed."""
        s = _make_session(sid="proc_run2", notify_on_complete=True, output="partial")
        registry._running[s.id] = s

        registry.poll("proc_run2")
        assert "proc_run2" not in registry._poll_observed

    def test_wait_and_log_still_skip_cli_drain(self, registry):
        """wait()/read_log() consume the output, so the CLI drain skips their
        completions via _completion_consumed (the original #8228 contract)."""
        for sid, action in (("proc_w", "wait"), ("proc_l", "log")):
            s = _make_session(sid=sid, notify_on_complete=True, output="done")
            s.exited = True
            s.exit_code = 0
            registry._running[s.id] = s
            with patch.object(registry, "_write_checkpoint"):
                registry._move_to_finished(s)
            if action == "wait":
                registry.wait(sid, timeout=1)
            else:
                registry.read_log(sid)
            assert registry.is_completion_consumed(sid)
        assert registry.drain_notifications() == []


# ---------------------------------------------------------------------------
# Silent-background-process hint
#
# background=True without notify_on_complete=True OR watch_patterns runs
# the process silently — the agent has no way to learn it finished short
# of calling process(action="poll") explicitly. The tool result must
# include a "hint" field that nudges the agent toward
# notify_on_complete=True for bounded tasks. May 2026 PR #31231 incident:
# bg CI poller exited green, agent never noticed, user had to surface it.
# ---------------------------------------------------------------------------


def _silent_bg_base_config(tmp_path):
    return {
        "env_type": "local",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
    }


def _silent_bg_harness(monkeypatch, tmp_path):
    """Common test fixture: patch enough of terminal_tool to spawn a fake
    background process and capture the JSON result the agent sees."""
    import tools.terminal_tool as terminal_tool_module
    from tools import process_registry as process_registry_module
    from types import SimpleNamespace

    config = _silent_bg_base_config(tmp_path)
    dummy_env = SimpleNamespace(env={})

    def fake_spawn_local(**kwargs):
        return SimpleNamespace(
            id="proc_silent_test",
            pid=4242,
            notify_on_complete=False,
            watcher_platform="",
            watcher_chat_id="",
            watcher_user_id="",
            watcher_user_name="",
            watcher_thread_id="",
            watcher_message_id="",
            watcher_interval=0,
        )

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)
    return terminal_tool_module


def test_background_without_notify_emits_silent_process_hint(monkeypatch, tmp_path):
    """The footgun case (May 2026 PR #31231): bg=True alone runs silently
    and the agent has no signal it finished. Tool must nudge."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    try:
        result = json.loads(
            tt.terminal_tool(
                command="while true; do gh pr checks 999; sleep 30; done",
                background=True,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert result["session_id"] == "proc_silent_test"
    hint = result.get("hint", "")
    assert hint, "Silent background process must include a hint field"
    assert "notify_on_complete" in hint, (
        "Hint must name the corrective flag so the agent can self-correct"
    )
    assert "silent" in hint.lower() or "no way to learn" in hint.lower(), (
        "Hint must explain the failure mode, not just suggest the fix"
    )


def test_background_with_notify_does_not_emit_hint(monkeypatch, tmp_path):
    """The correct shape — bg+notify together — must not nag."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    try:
        result = json.loads(
            tt.terminal_tool(
                command="pytest tests/",
                background=True,
                notify_on_complete=True,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert "hint" not in result, (
        f"Correct usage must not emit a hint, got: {result.get('hint')!r}"
    )
    assert result.get("notify_on_complete") is True


def test_foreground_command_does_not_emit_hint(monkeypatch, tmp_path):
    """Hint only applies to background processes — foreground returns its
    result synchronously and the agent always sees the outcome."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)

    # Foreground path doesn't go through spawn_local. Patch the local-env
    # exec method to short-circuit to a clean exit so the test doesn't
    # actually shell out.
    from types import SimpleNamespace
    dummy_env = SimpleNamespace(
        env={},
        execute=lambda *a, **kw: {"output": "done", "exit_code": 0, "error": None},
    )
    monkeypatch.setitem(tt._active_environments, "default", dummy_env)

    try:
        result = json.loads(
            tt.terminal_tool(
                command="echo hello",
                background=False,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert "hint" not in result, (
        f"Foreground commands must not emit the background-silence hint, got: {result.get('hint')!r}"
    )


# ---------------------------------------------------------------------------
# Homebrewed-CI-watcher hint
#
# Background processes whose command looks like a hand-rolled CI poller
# (`gh pr view` / `gh pr checks` combined with jq/awk on stdout) get an
# additional hint pointing at the canonical green-ci-policy snippet. The
# homebrew shape has burned us repeatedly (May 2026 PRs #31329, #31448,
# #31695, #31709, #31745, #32264, #33131) with stdout buffering, jq null
# keys, conclusion-vs-status confusion, and TTY-only banner grepping —
# none of which the canonical snippets suffer from. Fire on every detection;
# false positives are cheap (~one read).
# ---------------------------------------------------------------------------


def test_non_ci_background_command_does_not_emit_homebrew_hint(monkeypatch, tmp_path):
    """A long-running task that happens to use awk for unrelated reasons
    must not be mistaken for a CI poller — the gating signal is the
    combination of `gh pr ...` AND a stdout parser."""
    tt = _silent_bg_harness(monkeypatch, tmp_path)
    try:
        result = json.loads(
            tt.terminal_tool(
                command="cat /var/log/syslog | awk '/error/ {print}' > /tmp/errs.log",
                background=True,
                notify_on_complete=True,
            )
        )
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert "hint" not in result, (
        f"Non-CI command using awk must not be flagged as homebrew CI poller, got: {result.get('hint')!r}"
    )


def test_background_spawn_captures_ui_origin_without_notify(monkeypatch, tmp_path):
    """#61719 residual: live `agent.terminal.output` chunks are emitted for
    EVERY background process, but the spawn-time UI owner was only captured
    when notify_on_complete/watch_patterns were set — a plain background=true
    spawn (a delegated child's, in particular) had no positive owner and its
    live output was dropped by the desktop router.

    The owner must be passed INTO spawn_local() (sweeper review): the local
    reader thread starts inside the spawn and can emit output before it
    returns, so a post-spawn attribute assignment races the first chunks."""
    import gateway.session_context as session_context

    tt = _silent_bg_harness(monkeypatch, tmp_path)
    from types import SimpleNamespace
    from tools import process_registry as process_registry_module

    spawned = {}

    def capturing_spawn_local(**kwargs):
        spawned["kwargs"] = kwargs
        return SimpleNamespace(
            id="proc_ui_origin_test",
            pid=4243,
            notify_on_complete=False,
            watcher_platform="",
            watcher_chat_id="",
            watcher_user_id="",
            watcher_user_name="",
            watcher_thread_id="",
            watcher_message_id="",
            watcher_interval=0,
        )

    monkeypatch.setattr(
        process_registry_module.process_registry, "spawn_local", capturing_spawn_local
    )
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda key, default="": "sid_ui_1" if key == "HERMES_UI_SESSION_ID" else default,
    )
    try:
        tt.terminal_tool(command="sleep 60", background=True)
    finally:
        tt._active_environments.pop("default", None)
        tt._last_activity.pop("default", None)

    assert spawned["kwargs"].get("origin_ui_session_id") == "sid_ui_1", (
        "plain background spawn (no notify/watch) must hand the spawn-time UI "
        "owner to spawn_local() so the session owns it before the reader starts"
    )


def test_immediate_output_carries_ui_owner_through_real_registry(tmp_path):
    """Sweeper ask: real registry lifecycle, not a stubbed spawn. The very
    first chunk a fast process emits must already see the UI owner on the
    session — spawn_local() sets it at ProcessSession construction, before
    the reader thread exists."""
    from tools.process_registry import process_registry

    recorded = []
    prev_sink = process_registry.on_output
    process_registry.on_output = lambda session, chunk: recorded.append(
        (getattr(session, "origin_ui_session_id", ""), chunk)
    )
    try:
        session = process_registry.spawn_local(
            command="echo immediate-owner-check",
            cwd=str(tmp_path),
            origin_ui_session_id="sid_live_1",
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not any(
            "immediate-owner-check" in chunk for _, chunk in recorded
        ):
            time.sleep(0.05)
    finally:
        process_registry.on_output = prev_sink
        try:
            process_registry.kill_process(session.id)
        except Exception:
            pass

    matching = [sid for sid, chunk in recorded if "immediate-owner-check" in chunk]
    assert matching, f"no live output observed; recorded={recorded!r}"
    assert all(sid == "sid_live_1" for sid in matching), (
        f"immediate output emitted without its UI owner: {matching}"
    )
