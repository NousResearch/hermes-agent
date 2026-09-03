"""Background verification runs must reach the evidence ledger.

``record_terminal_result()`` has exactly one production call site
(``tools/terminal_tool.py``), and it sits on the FOREGROUND completion path.
A ``background=True`` spawn returns early with a synthetic ``exit_code: 0``
while the command is still running, so it never reaches that site.

The consequence is not cosmetic. Any suite too slow for the foreground timeout
*must* run in the background, therefore can never record evidence, so
``verification_status()`` keeps returning ``stale`` (``last_edit_at`` stays
newer than the last recorded event) and the verify-on-stop nudge replays an
older foreground run forever — quoting test counts that no longer exist.

These tests exercise the real registry against real short-lived processes. The
project's canonical verify command (``pnpm run test``) is resolved through a
PATH shim so the assertions are hermetic and never depend on a real pnpm.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from agent.verification_evidence import mark_workspace_edited, verification_status
from tools.process_registry import ProcessRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return ProcessRegistry()


@pytest.fixture
def project(tmp_path):
    """A node project whose canonical verify command is `pnpm run test`."""
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "lint": "eslint ."}})
    )
    (tmp_path / "pnpm-lock.yaml").write_text("")
    return tmp_path


@pytest.fixture
def shim_env(tmp_path):
    """Put a fake `pnpm` on PATH so no real toolchain is required.

    The shim exits with the code named by HERMES_TEST_SHIM_RC (default 0), which
    lets one shim serve both the passing and failing cases. It optionally sleeps
    first (HERMES_TEST_SHIM_SLEEP) so a test can kill a still-running command
    without appending shell syntax that the classifier would reject.
    """
    bin_dir = tmp_path / "shimbin"
    bin_dir.mkdir()
    shim = bin_dir / "pnpm"
    shim.write_text(
        "#!/bin/sh\n"
        'echo "shim pnpm $*"\n'
        'if [ -n "${HERMES_TEST_SHIM_SLEEP:-}" ]; then sleep "$HERMES_TEST_SHIM_SLEEP"; fi\n'
        'exit "${HERMES_TEST_SHIM_RC:-0}"\n'
    )
    shim.chmod(0o755)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}


def _await_exit(registry, session_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        session = registry._finished.get(session_id)
        if session is not None and session.exited:
            # The evidence write happens on the same completion path; give the
            # reader thread a moment to finish it before asserting.
            time.sleep(0.05)
            return session
        time.sleep(0.05)
    raise AssertionError(f"background process {session_id} did not finish in {timeout}s")


def test_background_verification_run_records_evidence(registry, project, shim_env):
    """A passing background test command must land in the ledger.

    This is the regression: before the fix the ledger had no row for it,
    because only the foreground path recorded.
    """
    session = registry.spawn_local(
        "pnpm run test",
        cwd=str(project),
        task_id="t-bg-evidence",
        env_vars=shim_env,
    )
    session.parent_session_id = "sess-bg"
    _await_exit(registry, session.id)

    status = verification_status(session_id="sess-bg", cwd=project)
    assert status["status"] == "passed", (
        f"background verification run was not recorded: {status}"
    )
    evidence = status["evidence"]
    assert evidence is not None
    assert evidence["canonical_command"] == "pnpm run test"
    assert evidence["exit_code"] == 0


def test_background_run_clears_stale_after_an_edit(registry, project, shim_env):
    """The real user-visible symptom: stale must become passed again.

    An edit marks the workspace stale. A background verification that passes
    afterwards has to repoint the ledger, or the nudge keeps replaying the
    older event forever.
    """
    mark_workspace_edited(
        session_id="sess-bg",
        cwd=project,
        paths=[str(project / "src" / "app.ts")],
    )
    assert verification_status(session_id="sess-bg", cwd=project)["status"] in {
        "unverified",
        "stale",
    }

    session = registry.spawn_local(
        "pnpm run test",
        cwd=str(project),
        task_id="t-bg-stale",
        env_vars=shim_env,
    )
    session.parent_session_id = "sess-bg"
    _await_exit(registry, session.id)

    assert verification_status(session_id="sess-bg", cwd=project)["status"] == "passed"


def test_failing_background_run_is_recorded_as_failed(registry, project, shim_env):
    """A red background suite must never be recorded as passing."""
    session = registry.spawn_local(
        "pnpm run test",
        cwd=str(project),
        task_id="t-bg-fail",
        env_vars={**shim_env, "HERMES_TEST_SHIM_RC": "1"},
    )
    session.parent_session_id = "sess-bg-fail"
    _await_exit(registry, session.id)

    status = verification_status(session_id="sess-bg-fail", cwd=project)
    assert status["status"] != "passed"
    assert status["evidence"] is not None
    assert status["evidence"]["status"] == "failed"


def test_non_verification_background_command_records_nothing(
    registry, project, shim_env
):
    """Only verification commands are evidence; a plain command is not."""
    session = registry.spawn_local(
        "echo hello",
        cwd=str(project),
        task_id="t-bg-noise",
        env_vars=shim_env,
    )
    session.parent_session_id = "sess-bg-noise"
    _await_exit(registry, session.id)

    status = verification_status(session_id="sess-bg-noise", cwd=project)
    assert status["evidence"] is None
    assert status["status"] == "unverified"


def test_evidence_is_recorded_exactly_once(registry, project, shim_env):
    """A re-entrant completion must not insert duplicate evidence rows.

    ``_move_to_finished`` is reachable more than once for the same session
    (kill racing the reader thread), which is why the notification path guards
    on ``was_running``. The evidence write sits under the same guard and needs
    its own proof, otherwise a duplicate-insert regression would be invisible.
    """
    session = registry.spawn_local(
        "pnpm run test",
        cwd=str(project),
        task_id="t-bg-once",
        env_vars=shim_env,
    )
    session.parent_session_id = "sess-bg-once"
    finished = _await_exit(registry, session.id)

    # Re-drive the completion path exactly as a racing kill would.
    registry._move_to_finished(finished)
    registry._move_to_finished(finished)

    db = Path(os.environ["HERMES_HOME"]) / "verification_evidence.db"
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) FROM verification_events WHERE session_id = ?",
            ("sess-bg-once",),
        ).fetchone()[0]
    assert rows == 1, f"expected exactly one evidence row, got {rows}"


def test_killed_background_run_is_not_recorded_at_all(
    registry, project, shim_env
):
    """A verification process the user killed is not proof of anything.

    Asserting merely "not passed" would be too weak: a killed run exits
    non-zero, so it would satisfy that even if the completion-reason guard
    were removed and the kill were recorded as a genuine test *failure*.
    Recording a spurious failure is its own bug — it would tell the user their
    suite is red when they simply stopped it. So require that nothing is
    recorded at all.

    The command must also be the bare canonical verify command. A compound
    like ``pnpm run test; sleep 30`` is rejected by the classifier on its own,
    which would make this test pass whether or not the guard exists.
    """
    session = registry.spawn_local(
        "pnpm run test",
        cwd=str(project),
        task_id="t-bg-killed",
        # Hold the process open so there is a live run to kill, without
        # turning the command into an unclassifiable compound.
        env_vars={**shim_env, "HERMES_TEST_SHIM_SLEEP": "30"},
    )
    session.parent_session_id = "sess-bg-killed"
    time.sleep(0.3)
    registry.kill_process(session.id)
    _await_exit(registry, session.id)

    status = verification_status(session_id="sess-bg-killed", cwd=project)
    assert status["evidence"] is None, (
        f"a killed run must not be recorded as evidence: {status}"
    )
    assert status["status"] == "unverified"
