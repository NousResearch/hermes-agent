"""Tests for the STRICT-READONLY file-write workspace gate (V2 IDENTITY + S14 BINDING REPAIR).

Regression matrix coverage:

S5   strict worker writes own workspace  (write_file, patch — both)
S6   strict worker cannot write repo
S7   strict worker cannot write ~/.hermes/reports
S8   strict worker cannot write profile/config/.env/skills
S9   strict worker cannot write another workspace
S10  traversal blocked
S11  symlink escape blocked
S12  missing HERMES_KANBAN_TASK and/or HERMES_KANBAN_WORKSPACE => BLOCK
S13  invalid workspace env (relative / sentinel / non-existent) => BLOCK
S14  task/workspace binding mismatch => BLOCK
     (the dispatcher-pinned workspace must equal the canonical workspace
     of the persisted task identified by HERMES_KANBAN_TASK in the
     authoritative Kanban DB; see ``test_S14_*`` for the true matrix)
S19  unrelated task mutation remains blocked (pinned-workspace containment
     prevents cross-task leaks; the protection does NOT come from
     comparing the tool ``task_id`` argument against ``HERMES_KANBAN_TASK``)

Identity model under test:

* ``HERMES_KANBAN_TASK``         e.g. ``t_9ff5c2bf`` (dispatcher-pinned)
* tool ``task_id`` argument      e.g. ``20260819_203942_1782cf`` (Hermes
                                  session_id propagated through
                                  ``cli.py: run_conversation(task_id=...)``)
* ``HERMES_KANBAN_WORKSPACE``    dispatcher-pinned canonical path
* ``HERMES_KANBAN_DB`` + ``HERMES_KANBAN_BOARD``   authoritative DB/board
                                                       pins the S14
                                                       binding check
                                                       uses to look up
                                                       the task row.

The V2 previous gate (rejected on Canary #2) compared A
``HERMES_KANBAN_TASK`` against the tool's session_id-like ``task_id``
argument and denied every write because the two never match in
production; the V2 S14 repair replaced that with the authoritative
DB-anchored binding shown here. The trusted Kanban artifact promotion
path (``kanban_db.store_attachment_bytes``) is exercised in
``test_kanban_worker_strict_dispatch.py`` as S18 and is intentionally
NOT routed through this gate.

Tests use ONLY stdlib + hermes internals; no live network calls.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import pytest

import hermes_state
from tools.file_tools import patch_tool, write_file_tool


PRIMARY = "/home/jr-ubuntu/.hermes/hermes-agent"
import sys as _sys
if PRIMARY not in _sys.path:
    _sys.path.insert(0, PRIMARY)


# ---------------------------------------------------------------------------
# Identity model — values used through the file
# ---------------------------------------------------------------------------
#
# These constants encode the real production identity split proven by
# Canary #2. Keep the two strings as DIFFERENT values so any future
# test that re-introduces a ``task_id == env_task`` comparison fails on
# inspection.
_KANBAN_TASK_ID = "t_strict_test"                         # dispatcher-pinned
_SESSION_ID = "20260819_203942_1782cf"                    # tool task_id


# ---------------------------------------------------------------------------
# Kanban DB isolation helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_db_isolated(tmp_path, monkeypatch):
    """Isolate the Kanban DB into a fresh tmp sqlite file.

    Mirrors the canonical pattern used by
    ``tests/hermes_cli/test_kanban_initiator_strict_propagation.py``
    (``isolated_kanban_db``) and the worker dispatch test. The DB is
    opened with ``kb.connect(db_path)`` so the production schema /
    migrations run exactly once.
    """
    db_file = tmp_path / "test_kanban.db"
    from hermes_cli import kanban_db as kb

    _original_connect = kb.connect
    conn = _original_connect(db_file)
    conn.close()

    @contextmanager
    def _patched_connect_closing():
        c = _original_connect(db_file)
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(
        kb, "connect",
        lambda board=None: _original_connect(db_file),
    )
    monkeypatch.setattr(kb, "connect_closing", _patched_connect_closing)

    return {
        "db_path": db_file,
        "kb": kb,
    }


# ---------------------------------------------------------------------------
# Strict-env fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strict_env(tmp_path, monkeypatch, kanban_db_isolated):
    """Set up a STRICT-READONLY Kanban worker process environment.

    Persists authoritative task T1 in an isolated Kanban DB so the new
    S14 binding check has real state to anchor on. Wires the production
    identity split: dispatcher-pinned Kanban ``t_<id>`` ≠ the Hermes
    session_id that ``cli.py`` propagates as the tool ``task_id``.
    """
    kb = kanban_db_isolated["kb"]
    db_path = kanban_db_isolated["db_path"]
    home = tmp_path / "home"
    home.mkdir()

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "inside.txt").write_text("before\n", encoding="utf-8")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("repo readme\n", encoding="utf-8")
    reports = home / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "old.json").write_text("{}", encoding="utf-8")
    profile_dir = home / "profiles" / "coder"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text("model: x\n", encoding="utf-8")
    other_workspace = tmp_path / "other_ws"
    other_workspace.mkdir()
    (other_workspace / "other.txt").write_text("OTHER\n", encoding="utf-8")

    # Persist authoritative task T1 with an explicit absolute
    # ``workspace_path = <fixture workspace>`` so the S14 binding
    # helper's canonical computation matches the dispatcher-pinned
    # ``HERMES_KANBAN_WORKSPACE``. Without this the helper would derive
    # ``workspaces_root/T1`` (a different path), the binding check
    # would fire spuriously, and the happy-path tests would all
    # fail. This mirrors the real ``_default_spawn`` flow which calls
    # ``resolve_workspace`` + ``set_workspace_path`` before exporting
    # ``HERMES_KANBAN_WORKSPACE`` to the worker.
    with kb.connect_closing() as conn:
        kb.create_task(
            conn,
            title="strict-test-T1",
            created_by="user",
            workspace_kind="scratch",
            workspace_path=str(workspace),
            initial_status="running",
            strict_readonly=True,
        )
    # Look up the task id the schema actually assigned (sqlite ROWID
    # alias; deterministic for a single insert but we don't need to
    # hardcode it because the env we set below is what the gate sees).
    with kb.connect_closing() as conn:
        rows = conn.execute(
            "SELECT id FROM tasks WHERE title = ? ORDER BY id DESC LIMIT 1",
            ("strict-test-T1",),
        ).fetchall()
    actual_id = rows[0]["id"]

    monkeypatch.setenv("HERMES_KANBAN_STRICT_READONLY", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", actual_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_HOME", str(home))

    return {
        "home": home,
        "workspace": workspace,
        "repo": repo,
        "reports": reports,
        "profile": profile_dir,
        "other_workspace": other_workspace,
        "task_id": actual_id,
        "kanban_task_id": actual_id,
        "kb": kb,
        "db_path": db_path,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_error(result_json: str) -> bool:
    try:
        payload = json.loads(result_json)
    except Exception:
        return True
    if "error" in payload:
        return True
    return False


def _error_text(result_json: str) -> str:
    try:
        payload = json.loads(result_json)
    except Exception:
        return result_json
    err = payload.get("error") if isinstance(payload, dict) else None
    if err is None:
        return result_json
    if isinstance(err, dict):
        return json.dumps(err, ensure_ascii=False)
    return str(err)


def _call_write(target: Path, *, task_id: str, content: str = "x\n") -> str:
    return write_file_tool(path=str(target), content=content, task_id=task_id)


def _call_patch(target: Path, *, task_id: str) -> str:
    return patch_tool(
        mode="replace",
        path=str(target),
        old_string="before",
        new_string="after",
        task_id=task_id,
    )


# ---------------------------------------------------------------------------
# A. session-id tool task_id + own canonical workspace => ALLOW
# ---------------------------------------------------------------------------


def test_A_write_file_session_id_inside_workspace_allowed(strict_env):
    """A. write_file: session_id-like tool task_id + own workspace => ALLOW."""
    target = strict_env["workspace"] / "artifact_one.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert not _is_error(result), (
        f"unexpected error: {result}\n"
        f"regression: gate miscompared session_id against HERMES_KANBAN_TASK (Canary #2)"
    )
    assert target.read_text(encoding="utf-8") == "x\n"


def test_A_patch_session_id_inside_workspace_allowed(strict_env):
    """A. patch: session_id-like tool task_id + own workspace => ALLOW."""
    target = strict_env["workspace"] / "inside.txt"
    result = _call_patch(target, task_id=_SESSION_ID)
    assert not _is_error(result), (
        f"unexpected error: {result}\n"
        f"regression: gate miscompared session_id against HERMES_KANBAN_TASK (Canary #2)"
    )
    assert target.read_text(encoding="utf-8") == "after\n"


def test_A_nested_subdir_session_id_allowed(strict_env):
    """A. (nested): write to a subdir created inside the workspace."""
    sub = strict_env["workspace"] / "subdir"
    target = sub / "deep.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert not _is_error(result), f"unexpected error: {result}"
    assert target.read_text(encoding="utf-8") == "x\n"


# ---------------------------------------------------------------------------
# B / D. missing HERMES_KANBAN_TASK or HERMES_KANBAN_WORKSPACE => DENY
# ---------------------------------------------------------------------------


def test_B_missing_kanban_task_denied(strict_env, monkeypatch):
    """B: strict mode with empty HERMES_KANBAN_TASK is denied."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    target = strict_env["workspace"] / "should_fail.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), (
        f"missing dispatcher identity must be denied; got {result!r}"
    )
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "missing dispatcher identity" in text.lower()
    assert not target.exists()


def test_D_missing_workspace_env_denied(strict_env, monkeypatch):
    """D: with HERMES_KANBAN_STRICT_READONLY=1 but
    HERMES_KANBAN_WORKSPACE absent, write is denied."""
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    target = strict_env["workspace"] / "should_fail.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), (
        f"missing workspace must be denied; got {result!r}"
    )
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "missing dispatcher identity" in text.lower()
    assert not target.exists()


# ---------------------------------------------------------------------------
# C / E. invalid workspace env => DENY, error points to the workspace env
# ---------------------------------------------------------------------------


def test_C_invalid_workspace_relative_denied(strict_env, monkeypatch):
    """C: a relative HERMES_KANBAN_WORKSPACE is denied (path policy)."""
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "relative/path")
    target = strict_env["workspace"] / "should_fail.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), f"relative workspace must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "absolute" in text.lower()


def test_C_invalid_workspace_sentinel_denied(strict_env, monkeypatch):
    """E (sentinel): ``.`` / ``cwd`` / ``auto`` workspace is denied."""
    for sentinel in (".", "cwd", "auto"):
        monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", sentinel)
        target = strict_env["workspace"] / "x.txt"
        result = _call_write(target, task_id=_SESSION_ID)
        assert _is_error(result), f"sentinel {sentinel!r} not denied: {result!r}"
        text = _error_text(result)
        assert "STRICT_READONLY" in text
        assert "sentinel" in text.lower() or "missing" in text.lower()


def test_C_invalid_workspace_nonexistent_denied(strict_env, monkeypatch):
    """E (nonexistent): a path that does not exist on disk is denied."""
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/nope/does/not/exist/at/all")
    target = strict_env["workspace"] / "should_fail.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), f"nonexistent workspace must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "existing directory" in text.lower() or "absolute" in text.lower()


# ---------------------------------------------------------------------------
# S6. cannot write repo root
# ---------------------------------------------------------------------------


def test_S6_repo_root_write_denied(strict_env):
    """S6 (write_file): absolute repo path is denied (path containment)."""
    target = strict_env["repo"] / "INJECTED.md"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), f"repo path must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower(), (
        f"expected path-containment error, got {text!r}"
    )
    assert not target.exists()


def test_S6_repo_root_patch_denied(strict_env):
    """S6 (patch): patch inside the repo is denied (path containment)."""
    target = strict_env["repo"] / "README.md"
    result = _call_patch(target, task_id=_SESSION_ID)
    assert _is_error(result), f"repo path must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()
    assert target.read_text(encoding="utf-8") == "repo readme\n"


# ---------------------------------------------------------------------------
# S7. cannot write ~/.hermes/reports
# ---------------------------------------------------------------------------


def test_S7_reports_write_denied(strict_env):
    target = strict_env["reports"] / "leaked.json"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), f"reports path must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()
    assert not target.exists()


# ---------------------------------------------------------------------------
# S8. cannot write profile / config / .env / skills
# ---------------------------------------------------------------------------


def test_S8_profile_config_write_denied(strict_env):
    target = strict_env["profile"] / "config.yaml"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), f"profile path must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()
    assert target.read_text(encoding="utf-8") == "model: x\n"


def test_S8_hermes_config_write_denied(strict_env):
    cfg = strict_env["home"] / "config.yaml"
    cfg.write_text("ok: 1\n", encoding="utf-8")
    result = _call_write(cfg, task_id=_SESSION_ID)
    assert _is_error(result), f"hermes config path must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()
    assert cfg.read_text(encoding="utf-8") == "ok: 1\n"


def test_S8_dotenv_write_denied(strict_env):
    env = strict_env["home"] / ".env"
    env.write_text("OK=1\n", encoding="utf-8")
    result = _call_write(env, task_id=_SESSION_ID)
    assert _is_error(result), f"dotenv path must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()
    assert env.read_text(encoding="utf-8") == "OK=1\n"


def test_S8_skills_dir_write_denied(strict_env):
    skills = strict_env["home"] / "skills" / "evil"
    skills.mkdir(parents=True, exist_ok=True)
    target = skills / "SKILL.md"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), f"skills path must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()
    assert not target.exists()


# ---------------------------------------------------------------------------
# S9 / F. another task workspace => DENY (path containment)
# ---------------------------------------------------------------------------


def test_S9_other_workspace_write_denied(strict_env):
    """S9: writes into a different kanban workspace path are denied by
    path containment."""
    target = strict_env["other_workspace"] / "hijack.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), f"other-task workspace must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()
    assert not target.exists()


# ---------------------------------------------------------------------------
# G. traversal blocked (path containment)
# ---------------------------------------------------------------------------


def test_G_traversal_dotdot_blocked(strict_env):
    target = strict_env["workspace"] / ".." / "escaped.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), f"traversal must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()


def test_G_workspace_root_self_blocked(strict_env):
    result = _call_write(strict_env["workspace"], task_id=_SESSION_ID)
    assert _is_error(result), f"workspace root must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "target equals the workspace root" in text.lower()


# ---------------------------------------------------------------------------
# H. symlink escape blocked (path containment)
# ---------------------------------------------------------------------------


def test_H_symlink_escape_blocked(strict_env):
    outside = strict_env["home"] / "outside.txt"
    outside.write_text("OUTSIDE\n", encoding="utf-8")
    link = strict_env["workspace"] / "escape_link"
    os.symlink(str(outside), str(link))
    result = _call_write(link, task_id=_SESSION_ID)
    assert _is_error(result), f"symlink escape must be denied; got {result!r}"
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "escapes the workspace boundary" in text.lower()
    assert outside.read_text(encoding="utf-8") == "OUTSIDE\n"


# ===========================================================================
# S14 — TASK/WORKSPACE BINDING MISMATCH => BLOCK
# ===========================================================================
#
# The V2 S14 contract is the production bug this file section exists to
# prevent regressing: a writer whose env pins T1 + W2_T2 (W2 belonging to
# a *different* Kanban task) must be denied, even if W2_T2 itself exists
# on disk and the target lives inside it. Containment on W2 is not
# sufficient — the gate must look up T1 in the authoritative Kanban DB
# and compare the persisted canonical workspace to the pinned one.
#
# Pre-S14-repair the gate trusted ``HERMES_KANBAN_WORKSPACE`` literally,
# so a T1+W2_T2 binding silently allowed the write (S14 fail-open). The
# tests below demonstrate both the FAIL-CLOSED behaviour post-repair and
# the path-containment behaviour when the binding is consistent.


def test_S14_true_matrix_T1_workspace_W2_denied(
    tmp_path, monkeypatch, kanban_db_isolated
):
    """S14 (true matrix, scratch): dispatcher pins T1 + W2_T2, model
    targets W2/file → DENY because T1↔W2_T2 binding is wrong.

    Setup:

      * T1 persisted with ``scratch`` kind and NO workspace_path
        → canonical workspace is ``workspaces_root/T1``.
      * T2 persisted with explicit absolute workspace_path = ``W2``
        → canonical workspace is ``W2``.
      * Strict worker env: ``HERMES_KANBAN_TASK=T1``,
        ``HERMES_KANBAN_WORKSPACE=W2`` (T2's workspace).
      * Tool task_id = session-like ``20260819_203942_1782cf``.
      * Target = ``W2/artifact.txt`` (lexically inside W2_T2).

    Expected: DENY because canonical(T1) = workspaces_root/T1 ≠ W2.
    """
    kb = kanban_db_isolated["kb"]
    db_path = kanban_db_isolated["db_path"]

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # T1 (scratch, no workspace_path)
    with kb.connect_closing() as conn:
        t1 = kb.create_task(
            conn,
            title="t1-strict-target",
            created_by="user",
            workspace_kind="scratch",
            initial_status="running",
            strict_readonly=True,
        )

    # T2 with explicit absolute workspace_path = W2
    workspace2 = tmp_path / "ws_T2"
    workspace2.mkdir()
    with kb.connect_closing() as conn:
        t2 = kb.create_task(
            conn,
            title="t2-strict-other",
            created_by="user",
            workspace_kind="scratch",
            workspace_path=str(workspace2),
            initial_status="running",
            strict_readonly=True,
        )

    # Wire strict-mode env as the dispatcher would, but with the
    # INCONSISTENT binding we're testing: T1 pinned, W2_T2 pinned.
    monkeypatch.setenv("HERMES_KANBAN_STRICT_READONLY", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", t1)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace2))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    target = workspace2 / "artifact.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), (
        f"S14 TRUE MATRIX: T1+W2_T2 binding must be denied; got {result!r}"
    )
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    # The denial message must surface the binding mismatch — that is the
    # S14 contract, distinct from path containment.
    assert "binding mismatch" in text.lower(), (
        f"expected S14 binding-mismatch error, got {text!r}"
    )
    assert not target.exists()


def test_S14_dir_kind_T1_workspace_W2_denied(
    tmp_path, monkeypatch, kanban_db_isolated
):
    """S14 (true matrix, ``dir:``): even when W1.T1 and W2.T2 are both
    absolute explicit paths, the dispatcher-pinned mismatch must be
    detected and denied.

    Setup:

      * T1 persisted with ``dir:`` kind, workspace_path = ``W1``.
      * T2 persisted with ``dir:`` kind, workspace_path = ``W2``.
      * Strict worker env: ``HERMES_KANBAN_TASK=T1``,
        ``HERMES_KANBAN_WORKSPACE=W2``.
      * Target = ``W2/file.txt`` (inside W2_T2).

    Expected: DENY because canonical(T1) = W1 ≠ W2.
    """
    kb = kanban_db_isolated["kb"]
    db_path = kanban_db_isolated["db_path"]

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    workspace1 = tmp_path / "ws_T1"
    workspace1.mkdir()
    workspace2 = tmp_path / "ws_T2"
    workspace2.mkdir()

    with kb.connect_closing() as conn:
        t1 = kb.create_task(
            conn,
            title="t1-dir",
            created_by="user",
            workspace_kind="dir",
            workspace_path=str(workspace1),
            initial_status="running",
            strict_readonly=True,
        )
        t2 = kb.create_task(
            conn,
            title="t2-dir",
            created_by="user",
            workspace_kind="dir",
            workspace_path=str(workspace2),
            initial_status="running",
            strict_readonly=True,
        )

    monkeypatch.setenv("HERMES_KANBAN_STRICT_READONLY", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", t1)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace2))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    target = workspace2 / "file.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), (
        f"S14 (dir): T1+W2_T2 binding must be denied; got {result!r}"
    )
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "binding mismatch" in text.lower(), (
        f"expected S14 binding-mismatch error, got {text!r}"
    )
    assert not target.exists()


def test_S14_worktree_kind_T1_workspace_W2_denied(
    tmp_path, monkeypatch, kanban_db_isolated
):
    """S14 (true matrix, ``worktree:`` with explicit workspace_path).

    Setup mirrors the ``dir:`` case but uses ``worktree`` kind. Each
    task gets its own absolute ``workspace_path`` (the dispatcher side
    of ``_resolve_worktree_workspace`` would normally have created the
    linked git worktree; the strict gate's pure resolver instead
    trusts the persisted path verbatim, so the binding check still
    works without any I/O).

      * T1.worktree, workspace_path = ``W1``.
      * T2.worktree, workspace_path = ``W2``.
      * Env: ``HERMES_KANBAN_TASK=T1``, ``HERMES_KANBAN_WORKSPACE=W2``.
      * Target = ``W2/file``.

    Expected: DENY because canonical(T1) = W1 ≠ W2.
    """
    kb = kanban_db_isolated["kb"]
    db_path = kanban_db_isolated["db_path"]

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    workspace1 = tmp_path / "wt_T1"
    workspace1.mkdir()
    workspace2 = tmp_path / "wt_T2"
    workspace2.mkdir()

    with kb.connect_closing() as conn:
        t1 = kb.create_task(
            conn,
            title="t1-worktree",
            created_by="user",
            workspace_kind="worktree",
            workspace_path=str(workspace1),
            initial_status="running",
            strict_readonly=True,
        )
        t2 = kb.create_task(
            conn,
            title="t2-worktree",
            created_by="user",
            workspace_kind="worktree",
            workspace_path=str(workspace2),
            initial_status="running",
            strict_readonly=True,
        )

    monkeypatch.setenv("HERMES_KANBAN_STRICT_READONLY", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", t1)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace2))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    target = workspace2 / "file"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), (
        f"S14 (worktree): T1+W2_T2 binding must be denied; got {result!r}"
    )
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "binding mismatch" in text.lower(), (
        f"expected S14 binding-mismatch error, got {text!r}"
    )
    assert not target.exists()


def test_S14_binding_missing_db_pin_denied(strict_env, monkeypatch):
    """S14 (defence in depth): if the dispatcher forgot to pin
    ``HERMES_KANBAN_DB`` (or the env inherited a partial state), the
    binding check cannot be performed — the gate MUST deny rather than
    fall through to env-only containment."""
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    target = strict_env["workspace"] / "should_fail.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), (
        f"missing HERMES_KANBAN_DB must fail closed; got {result!r}"
    )
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "binding" in text.lower() or "board/db identity" in text.lower(), (
        f"expected binding-check denial, got {text!r}"
    )
    assert not target.exists()


def test_S14_binding_missing_board_pin_denied(strict_env, monkeypatch):
    """S14 (defence in depth): missing ``HERMES_KANBAN_BOARD`` is also
    fail-closed."""
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    target = strict_env["workspace"] / "should_fail.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), (
        f"missing HERMES_KANBAN_BOARD must fail closed; got {result!r}"
    )
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "binding" in text.lower() or "board/db identity" in text.lower()
    assert not target.exists()


def test_S14_binding_unknown_task_denied(strict_env, monkeypatch):
    """S14 (defence in depth): an ``HERMES_KANBAN_TASK`` that is not
    present in the authoritative Kanban DB is denied — the gate does
    NOT trust the env when authoritative state disagrees."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_does_not_exist_anywhere")
    target = strict_env["workspace"] / "should_fail.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert _is_error(result), (
        f"unknown task must fail closed; got {result!r}"
    )
    text = _error_text(result)
    assert "STRICT_READONLY" in text
    assert "not found" in text.lower() or "binding" in text.lower(), (
        f"expected task-not-found denial, got {text!r}"
    )
    assert not target.exists()


# ---------------------------------------------------------------------------
# VALID BINDING — write/patch MUST still succeed under T1↔W1
# ---------------------------------------------------------------------------


def test_S14_valid_binding_write_allowed(strict_env):
    """Valid binding: T1+W1 (workspace), session_id tool task_id. ALLOW."""
    target = strict_env["workspace"] / "artifact_two.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    text = _error_text(result) if _is_error(result) else ""
    assert not _is_error(result), (
        f"valid T1+W1 binding must allow write; got {result!r}\n"
        f"error excerpt: {text!r}"
    )
    assert target.read_text(encoding="utf-8") == "x\n"


def test_S14_valid_binding_patch_allowed(strict_env):
    """Valid binding: T1+W1 (workspace), session_id tool task_id. ALLOW."""
    target = strict_env["workspace"] / "inside.txt"
    result = _call_patch(target, task_id=_SESSION_ID)
    text = _error_text(result) if _is_error(result) else ""
    assert not _is_error(result), (
        f"valid T1+W1 binding must allow patch; got {result!r}\n"
        f"error excerpt: {text!r}"
    )
    assert target.read_text(encoding="utf-8") == "after\n"


# ---------------------------------------------------------------------------
# CANARY #2 REGRESSION (must remain green post S14)
# ---------------------------------------------------------------------------


def test_canary2_identity_alignment_regression(strict_env):
    """Canary #2 regression: legitimate in-workspace write must succeed
    even though the tool ``task_id`` (Hermes session_id) does not
    equal ``HERMES_KANBAN_TASK`` (Kanban task id).

    Pre-fix this was denied by ``task_id != env_task``. Post-fix it
    is allowed by the S14 DB-anchored binding check (the binding
    matches), and post-S14-repair the binding check is anchored on
    the authoritative Kanban DB lookup rather than a string
    comparison.
    """
    session_like_tool_id = "20260819_203942_1782cf"
    workspace = strict_env["workspace"]
    target = workspace / "artifact_one.txt"
    result = _call_write(target, task_id=session_like_tool_id, content="hello\n")
    text = _error_text(result) if _is_error(result) else ""
    assert not _is_error(result), (
        "Canary #2 regression: legitimate in-workspace write was denied. "
        "Pre-fix this surfaced as a 'task/workspace mismatch' string "
        "compare; post-fix the S14 DB-anchored binding check must allow it.\n"
        f"  HERMES_KANBAN_TASK='{strict_env['kanban_task_id']}'\n"
        f"  tool task_id='{session_like_tool_id}' (Hermes session_id)\n"
        f"  target within workspace '{workspace}'\n"
        f"  result: {result!r}"
    )
    assert target.read_text(encoding="utf-8") == "hello\n"


# ---------------------------------------------------------------------------
# Static source-anchored regression for the session-id-vs-kanban-id bug
# ---------------------------------------------------------------------------


def test_pre_fix_behavior_not_reintroduced():
    """Source-level guarantee: the string-equality comparison that
    caused Canary #2 must never return. A regression that re-introduces
    it fails this test in review (and the production source itself
    will fail code review)."""
    import tools.file_tools as _ft
    src = Path(_ft.__file__).read_text(encoding="utf-8")
    forbidden = re.search(
        r"str\(task_id\)\s*!=\s*env_task|env_task\s*!=\s*str\(task_id\)"
        r"|task_id\s*==\s*env_task|env_task\s*==\s*task_id",
        src,
    )
    assert forbidden is None, (
        "Regression: tools/file_tools.py re-introduced the session-id-vs-"
        "kanban-id string equality comparison that Canary #2 caught."
    )


# ---------------------------------------------------------------------------
# J. non-strict behavior unchanged
# ---------------------------------------------------------------------------


def test_J_outside_strict_mode_unaffected(strict_env, monkeypatch, tmp_path):
    """Outside strict mode the gate is a no-op."""
    monkeypatch.delenv("HERMES_KANBAN_STRICT_READONLY", raising=False)
    safe_dir = tmp_path / "safe_outside"
    safe_dir.mkdir()
    target = safe_dir / "ok.txt"
    result = _call_write(target, task_id=_SESSION_ID)
    assert not _is_error(result), (
        f"unexpected error outside strict mode: {result}"
    )
    assert target.read_text(encoding="utf-8") == "x\n"
