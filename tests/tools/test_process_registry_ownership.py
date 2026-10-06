"""Background-process origin survives notifications and checkpoint recovery."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr("tools.process_registry._SYSTEMD_SCOPE_AVAILABLE", False)
    return ProcessRegistry()


def _make_session(sid, *, exited=False, exit_code=None):
    return ProcessSession(id=sid, command="echo hello", task_id="t1", exited=exited, exit_code=exit_code)


# =========================================================================
# Background-process ownership (origin_ui_session_id)
# =========================================================================

class TestOriginUiSessionOwnership:
    """The commissioning UI tab must survive into notifications and restarts.

    ``session_key`` is the durable conversation key and several live tabs can
    share one, so it cannot identify the window a command was started from.
    """

    def test_completion_notification_carries_the_origin_tab(self, registry):
        s = _make_session(sid="proc_owner_completion", exited=True, exit_code=0)
        s.session_key = "shared-key"
        s.origin_ui_session_id = "tab_origin"
        s.notify_on_complete = True
        registry._running[s.id] = s

        with patch.object(registry, "_write_checkpoint"):
            registry._move_to_finished(s)

        results = registry.drain_notifications(
            owns_event=lambda event: event.get("origin_ui_session_id") == "tab_origin"
        )
        assert len(results) == 1
        event, formatted = results[0]
        assert event["type"] == "completion"
        assert event["origin_ui_session_id"] == "tab_origin"
        assert event["session_key"] == "shared-key"
        assert "proc_owner_completion" in formatted

    def test_watch_match_notification_carries_the_origin_tab(self, registry):
        s = _make_session(sid="proc_owner_watch")
        s.session_key = "shared-key"
        s.origin_ui_session_id = "tab_origin"
        s.watch_patterns = ["ready"]
        registry._running[s.id] = s

        registry._check_watch_patterns(s, "server ready\n")

        event = registry.completion_queue.get_nowait()
        assert event["type"] == "watch_match"
        assert event["origin_ui_session_id"] == "tab_origin"

    def test_spawn_records_the_origin_tab(self, registry):
        with patch("subprocess.Popen") as mock_popen, \
             patch("threading.Thread"), \
             patch.object(registry, "_write_checkpoint"):
            mock_popen.return_value = MagicMock(pid=4321, poll=lambda: None)
            session = registry.spawn_local(
                "sleep 1",
                cwd="/tmp",
                session_key="shared-key",
                origin_ui_session_id="tab_origin",
            )

        assert session.origin_ui_session_id == "tab_origin"

    def test_yielded_foreground_process_keeps_the_origin_tab(self, registry, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.terminal_tool_background import yield_to_background_handler

        monkeypatch.setattr("tools.process_registry.process_registry", registry)
        tokens = set_session_vars(ui_session_id="tab_origin", session_key="shared-key")
        try:
            handler = yield_to_background_handler(
                command="sleep 1", env_type="local", cwd="/tmp", effective_task_id="task",
                task_id="task", session_key="shared-key",
            )
        finally:
            clear_session_vars(tokens)

        with patch.object(registry, "_track_started") as track, patch.object(registry, "_safe_host_start_time"):
            result = handler(MagicMock(pid=4321), "partial output")

        # The yielding callback can run after its commissioning context has ended.
        adopted = track.call_args.args[0]
        assert result["yielded_session_id"] == adopted.id
        assert adopted.origin_ui_session_id == "tab_origin"
        assert adopted.output_buffer == "partial output"

    def test_origin_tab_survives_checkpoint_recovery(self, registry, tmp_path):
        """A gateway restart must not forget which tab owns a live process."""
        checkpoint = tmp_path / "procs.json"
        s = _make_session(sid="proc_owner_ckpt")
        s.pid = os.getpid()  # a PID that is definitely alive
        s.pid_scope = "host"
        s.session_key = "shared-key"
        s.origin_ui_session_id = "tab_origin"
        registry._running[s.id] = s

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            registry._write_checkpoint()
            written = json.loads(checkpoint.read_text())
            assert written[0]["origin_ui_session_id"] == "tab_origin"

            fresh = ProcessRegistry()
            with patch.object(fresh, "_host_pid_is_ours", lambda *_a: True), \
                 patch.object(fresh, "_is_host_pid_alive", lambda *_a: True), \
                 patch.object(fresh, "_write_checkpoint"):
                assert fresh.recover_from_checkpoint() == 1

        assert fresh._running["proc_owner_ckpt"].origin_ui_session_id == "tab_origin"
