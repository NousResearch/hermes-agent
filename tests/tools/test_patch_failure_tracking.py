"""Tests for per-file consecutive patch-failure tracking.

When the agent repeatedly fails to patch the same file with similar but
non-matching old_strings, it's usually stuck in a loop with a stale view
of the file.  After 3 consecutive failures on the same path, the patch
tool injects an escalating ``_hint`` that tells the model to break out
of the loop (re-read, use longer context, or fall back to write_file).

See issue #507 (Roo Code deep-dive, item 2f).
"""

import json

import pytest


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    """Isolate HERMES_HOME and clear module-level caches afterward so the
    real shell-out side effects from _handle_patch don't leak into
    subsequent tests (see test_line_ending_preservation.py for details)."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
    try:
        from tools.file_tools import clear_file_ops_cache
        from tools.file_tools_read_tracking import _read_tracker_lock, _read_tracker
        clear_file_ops_cache()
        with _read_tracker_lock:
            _read_tracker.clear()
    except Exception:
        pass
    try:
        from tools.terminal_tool import _active_environments, _env_lock
        with _env_lock:
            _active_environments.clear()
    except Exception:
        pass


@pytest.fixture
def fresh_tracker():
    """Reset the module-level tracker before each test so the count starts
    at zero regardless of prior test order."""
    from tools.file_tools_read_tracking import _patch_failure_tracker, _patch_failure_lock

    with _patch_failure_lock:
        _patch_failure_tracker.clear()
    yield
    with _patch_failure_lock:
        _patch_failure_tracker.clear()


class TestPatchFailureEscalation:
    def test_first_two_failures_use_normal_hint(self, hermes_home, tmp_path, fresh_tracker):
        from tools.file_tools import _handle_patch

        target = tmp_path / "f.py"
        target.write_text("def foo():\n    return 1\n")

        for _i in range(2):
            result = _handle_patch(
                {
                    "mode": "replace",
                    "path": str(target),
                    "old_string": f"NONEXISTENT_{_i}_XYZQQQ",
                    "new_string": "x",
                },
                task_id="esc_t1",
            )
            d = json.loads(result)
            hint = d.get("_hint", "") or ""
            assert "failure #" not in hint, (
                f"Escalating hint fired too early on attempt {_i + 1}: {hint!r}"
            )


    def test_different_tasks_have_independent_counters(
        self, hermes_home, tmp_path, fresh_tracker
    ):
        from tools.file_tools import _handle_patch

        target = tmp_path / "shared.py"
        target.write_text("z = 0\n")

        # Three failures under task A.
        for _i in range(3):
            _handle_patch(
                {
                    "mode": "replace",
                    "path": str(target),
                    "old_string": f"GHOST_A_{_i}_QWE",
                    "new_string": "x",
                },
                task_id="task_A",
            )

        # First failure under task B — should NOT see escalation.
        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "GHOST_B_QWE",
                "new_string": "x",
            },
            task_id="task_B",
        )
        d = json.loads(result)
        hint = d.get("_hint", "") or ""
        assert "failure #" not in hint, (
            f"task_B's hint cross-contaminated from task_A: {hint!r}"
        )
