"""Tests for per-file consecutive patch-failure tracking.

When the agent repeatedly fails to patch the same file with similar but
non-matching old_strings, it's usually stuck in a loop with a stale view
of the file.  After 2 consecutive failures on the same path, the patch
tool returns a ``PATCH-LOOP BLOCKER`` that tells the model to re-read the
current target file and patch against fresh context.

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
        from tools.file_tools import clear_file_ops_cache, _read_tracker_lock, _read_tracker
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
    from tools.file_tools import _patch_failure_tracker, _patch_failure_lock

    with _patch_failure_lock:
        _patch_failure_tracker.clear()
    yield
    with _patch_failure_lock:
        _patch_failure_tracker.clear()


class TestPatchFailureEscalation:
    def test_first_failure_uses_normal_hint(self, hermes_home, tmp_path, fresh_tracker):
        from tools.file_tools import _handle_patch

        target = tmp_path / "f.py"
        target.write_text("def foo():\n    return 1\n")

        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "NONEXISTENT_XYZQQQ",
                "new_string": "x",
            },
            task_id="esc_t1",
        )
        d = json.loads(result)
        hint = d.get("_hint", "") or ""
        assert "PATCH-LOOP BLOCKER" not in hint, (
            f"Patch-loop blocker fired too early on attempt 1: {hint!r}"
        )

    def test_second_consecutive_replace_failure_returns_patch_loop_blocker(
        self, hermes_home, tmp_path, fresh_tracker
    ):
        from tools.file_tools import _handle_patch

        target = tmp_path / "f.py"
        target.write_text("def foo():\n    return 1\n")

        last_hint = ""
        for _i in range(2):
            result = _handle_patch(
                {
                    "mode": "replace",
                    "path": str(target),
                    "old_string": f"DOES_NOT_EXIST_{_i}_FOOFOOFOO",
                    "new_string": "x",
                },
                task_id="esc_t2",
            )
            d = json.loads(result)
            last_hint = d.get("_hint", "") or ""

        assert "PATCH-LOOP BLOCKER" in last_hint, repr(last_hint)
        assert "second consecutive" in last_hint
        assert "re-read the current target file" in last_hint
        assert "patch against fresh context" in last_hint
        assert "checkout" not in last_hint.lower()
        assert "reset" not in last_hint.lower()

    def test_success_clears_failure_counter(self, hermes_home, tmp_path, fresh_tracker):
        from tools.file_tools import _handle_patch

        target = tmp_path / "f.py"
        target.write_text("def foo():\n    return 1\n")

        # Two failures: counter is blocked.
        for _i in range(2):
            _handle_patch(
                {
                    "mode": "replace",
                    "path": str(target),
                    "old_string": f"GHOST_{_i}_ABCABC",
                    "new_string": "x",
                },
                task_id="esc_t3",
            )

        # Successful patch: clears the counter.
        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "return 1",
                "new_string": "return 99",
            },
            task_id="esc_t3",
        )
        d = json.loads(result)
        assert not d.get("error"), d

        # Next failure should be back to "attempt 1" — generic hint only.
        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "STILL_GHOST_XYZ",
                "new_string": "x",
            },
            task_id="esc_t3",
        )
        d = json.loads(result)
        hint = d.get("_hint", "") or ""
        assert "PATCH-LOOP BLOCKER" not in hint, (
            f"Counter should have been reset after success: {hint!r}"
        )

    def test_different_paths_have_independent_counters(
        self, hermes_home, tmp_path, fresh_tracker
    ):
        from tools.file_tools import _handle_patch

        a = tmp_path / "a.py"
        a.write_text("x = 1\n")
        b = tmp_path / "b.py"
        b.write_text("y = 2\n")

        # Two failures on a.py.
        for _i in range(2):
            _handle_patch(
                {
                    "mode": "replace",
                    "path": str(a),
                    "old_string": f"NONE_A_{_i}_ZZZ",
                    "new_string": "x",
                },
                task_id="esc_t4",
            )

        # One failure on b.py — should NOT inherit a.py's count.
        result = _handle_patch(
            {
                "mode": "replace",
                "path": str(b),
                "old_string": "NONE_B_ZZZ",
                "new_string": "x",
            },
            task_id="esc_t4",
        )
        d = json.loads(result)
        hint = d.get("_hint", "") or ""
        assert "PATCH-LOOP BLOCKER" not in hint, (
            f"b.py's hint inherited a.py's count: {hint!r}"
        )

    def test_different_tasks_have_independent_counters(
        self, hermes_home, tmp_path, fresh_tracker
    ):
        from tools.file_tools import _handle_patch

        target = tmp_path / "shared.py"
        target.write_text("z = 0\n")

        # Two failures under task A.
        for _i in range(2):
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
        assert "PATCH-LOOP BLOCKER" not in hint, (
            f"task_B's hint cross-contaminated from task_A: {hint!r}"
        )

    def test_second_consecutive_v4a_context_failure_returns_patch_loop_blocker(
        self, hermes_home, tmp_path, fresh_tracker
    ):
        from tools.file_tools import _handle_patch

        target = tmp_path / "v4a_target.py"
        target.write_text("def foo():\n    return 1\n")

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {target}\n"
            " THIS LINE DOES NOT EXIST\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )

        last_hint = ""
        last_error = ""
        for _i in range(2):
            result = _handle_patch(
                {"mode": "patch", "patch": patch},
                task_id="esc_v4a",
            )
            d = json.loads(result)
            assert d.get("error"), d
            last_error = d.get("error", "") or ""
            last_hint = d.get("_hint", "") or ""

        assert last_error, "The blocker must preserve the patch failure payload"
        assert "PATCH-LOOP BLOCKER" in last_hint, repr(last_hint)
        assert "second consecutive" in last_hint
        assert "V4A" in last_hint
        assert "re-read the current target file" in last_hint
        assert "patch against fresh context" in last_hint
        assert "checkout" not in last_hint.lower()
        assert "reset" not in last_hint.lower()
