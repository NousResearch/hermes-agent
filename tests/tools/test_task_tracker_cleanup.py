"""clear_task_trackers: per-task file-tool state must die with its task.

The four module-level registries in tools.file_tools are keyed by task_id
and had no removal path for finished tasks. Subagent task ids are unique
per delegation (``subagent-<n>-<uuid>``), so a long-running gateway process
accreted one dead entry set per delegation for its whole lifetime. These
tests cover the new clear_task_trackers() helper and its AIAgent.close()
wiring.
"""

import types

import tools.file_tools as ft
from tools.file_tools import clear_task_trackers


def _seed(task_id: str) -> None:
    with ft._read_tracker_lock:
        ft._read_tracker[task_id] = {"last_key": None, "read_history": set()}
    with ft._patch_failure_lock:
        ft._patch_failure_tracker[task_id] = {"/tmp/a.py": 2}
    with ft._file_ops_lock:
        ft._last_known_cwd[task_id] = "/tmp"
        ft._file_ops_cache[task_id] = object()


def _purge(*task_ids: str) -> None:
    for tid in task_ids:
        clear_task_trackers(tid)


class TestClearTaskTrackers:
    def test_pops_all_four_registries(self):
        tid = "subagent-1-deadbeef"
        _seed(tid)
        try:
            clear_task_trackers(tid)
            with ft._read_tracker_lock:
                assert tid not in ft._read_tracker
            with ft._patch_failure_lock:
                assert tid not in ft._patch_failure_tracker
            with ft._file_ops_lock:
                assert tid not in ft._last_known_cwd
                assert tid not in ft._file_ops_cache
        finally:
            _purge(tid)

    def test_other_tasks_untouched(self):
        dead, alive = "subagent-2-11111111", "subagent-3-22222222"
        _seed(dead)
        _seed(alive)
        try:
            clear_task_trackers(dead)
            with ft._read_tracker_lock:
                assert alive in ft._read_tracker
            with ft._file_ops_lock:
                assert ft._last_known_cwd.get(alive) == "/tmp"
        finally:
            _purge(dead, alive)

    def test_shared_container_alias_survives_child_clear(self):
        """Subagents collapse to the parent's "default" container key
        (terminal_tool._resolve_container_task_id); clearing a child must
        pop only the child's exact key, never the shared alias."""
        child = "subagent-4-33333333"
        _seed(child)
        with ft._file_ops_lock:
            ft._last_known_cwd["default"] = "/parent/cwd"
        try:
            clear_task_trackers(child)
            with ft._file_ops_lock:
                assert ft._last_known_cwd.get("default") == "/parent/cwd"
        finally:
            _purge(child)
            with ft._file_ops_lock:
                ft._last_known_cwd.pop("default", None)

    def test_empty_and_missing_ids_are_noops(self):
        clear_task_trackers("")
        clear_task_trackers("never-seeded-task-id")


class TestAgentCloseWiring:
    def test_close_clears_session_and_conversation_task_ids(self):
        """AIAgent.close() must clear trackers for both id keyings: tool
        calls key on the conversation task_id while close() historically
        only knew session_id."""
        from run_agent import AIAgent

        session_tid = "close-test-session"
        convo_tid = "subagent-5-44444444"
        _seed(session_tid)
        _seed(convo_tid)

        fake = types.SimpleNamespace(
            session_id=session_tid,
            _current_task_id=convo_tid,
            client=None,
            _session_messages=[],
            _end_session_on_close=False,
        )
        try:
            AIAgent.close(fake)
            with ft._read_tracker_lock:
                assert session_tid not in ft._read_tracker
                assert convo_tid not in ft._read_tracker
            with ft._file_ops_lock:
                assert session_tid not in ft._file_ops_cache
                assert convo_tid not in ft._file_ops_cache
        finally:
            _purge(session_tid, convo_tid)
