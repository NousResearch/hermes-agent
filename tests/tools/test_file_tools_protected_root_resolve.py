"""Tests for protected-root write guard: resolve-then-check + fail-closed.

These tests cover the bug where guard fired against the raw relative path +
wrong cwd *before* _resolve_path_for_task() ran, letting writes through to
the actual protected path.  After the fix, the guard uses the resolved
absolute path.

Scenarios:
  (a) cwd outside protected root, task-resolved path inside → write_file blocked
  (b) same condition, patch replace mode → blocked
  (c) V4A patch: one header inside protected root → entire patch refused
  (d) guard raises exception → write/patch fail-closed (refused, not allowed)
  (e) linked-worktree target (.worktrees/) still allowed (regression guard)
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_file_ops_mock(tmp_path, filename="target.py"):
    """Return a mock file_ops that pretends to write target_file and tracks calls."""
    target = tmp_path / filename
    ops = MagicMock()

    def _fake_write(path, content):
        target.write_text(content)
        m = MagicMock()
        m.to_dict.return_value = {"status": "ok"}
        return m

    ops.write_file.side_effect = _fake_write
    return ops, target


# ---------------------------------------------------------------------------
# (a) write_file_tool — relative path, task workspace = protected root
# ---------------------------------------------------------------------------
def test_write_file_resolved_protected_root_is_blocked(monkeypatch, tmp_path):
    """Guard fires on the RESOLVED path even when cwd is outside the root."""
    protected = tmp_path / "protected_repo"
    protected.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(protected))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(outside)  # process cwd is OUTSIDE the protected root

    # _resolve_path_for_task returns a path INSIDE the protected root
    resolved_target = str(protected / "agent.py")

    from tools.file_tools import write_file_tool

    with patch("tools.file_tools._resolve_path_for_task", return_value=Path(resolved_target)), \
         patch("tools.file_tools._get_file_ops") as mock_ops:
        result = json.loads(write_file_tool("agent.py", "malicious content"))

    assert "error" in result, f"expected error, got: {result}"
    assert "protected checkout" in result["error"]
    mock_ops.assert_not_called()
    assert not Path(resolved_target).exists()


# ---------------------------------------------------------------------------
# (b) patch_tool replace mode — relative path, task workspace = protected root
# ---------------------------------------------------------------------------
def test_patch_replace_resolved_protected_root_is_blocked(monkeypatch, tmp_path):
    """patch_tool replace mode also checks the resolved path, not raw path + cwd."""
    protected = tmp_path / "protected_repo"
    protected.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(protected))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(outside)

    resolved_target = str(protected / "run_agent.py")

    from tools.file_tools import patch_tool

    with patch("tools.file_tools._resolve_path_for_task", return_value=Path(resolved_target)), \
         patch("tools.file_tools._get_file_ops") as mock_ops:
        result = json.loads(patch_tool(
            mode="replace",
            path="run_agent.py",
            old_string="old",
            new_string="new",
        ))

    assert "error" in result, f"expected error, got: {result}"
    assert "protected checkout" in result["error"]
    mock_ops.assert_not_called()


# ---------------------------------------------------------------------------
# (c) V4A patch — one header inside protected root → entire patch refused
# ---------------------------------------------------------------------------
def test_patch_v4a_one_protected_target_refuses_whole_patch(monkeypatch, tmp_path):
    """If any V4A path is inside the protected root, the entire patch is refused."""
    protected = tmp_path / "protected_repo"
    protected.mkdir()
    safe = tmp_path / "workspace"
    safe.mkdir()
    safe_file = safe / "README.md"
    safe_file.write_text("# Original\n")

    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(protected))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    # One safe path, one protected path.
    # _resolve_path_for_task maps safe_path → safe_file,
    # protected_path → inside protected root.
    safe_path = str(safe_file)
    protected_path = str(protected / "core.py")

    v4a_patch = (
        "*** Begin Patch\n"
        f"*** Update File: {safe_path}\n"
        "@@ -1 +1 @@\n"
        "-# Original\n"
        "+# Modified\n"
        f"*** Update File: {protected_path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "*** End Patch\n"
    )

    from tools.file_tools import patch_tool

    # _resolve_path_for_task returns paths as-is (they're already absolute)
    with patch("tools.file_tools._resolve_path_for_task", side_effect=lambda p, _tid: Path(p)), \
         patch("tools.file_tools._get_file_ops") as mock_ops:
        result = json.loads(patch_tool(mode="patch", patch=v4a_patch))

    assert "error" in result, f"expected error, got: {result}"
    assert "protected checkout" in result["error"]
    mock_ops.assert_not_called()
    # Safe file must be untouched (atomic refusal)
    assert safe_file.read_text() == "# Original\n"


# ---------------------------------------------------------------------------
# (d) guard exception → fail-closed: write/patch refused, not silently allowed
# ---------------------------------------------------------------------------
def test_write_file_guard_exception_is_fail_closed(monkeypatch, tmp_path):
    """If check_protected_root_write_guard raises, the write is refused."""
    protected = tmp_path / "protected_repo"
    protected.mkdir()

    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(protected))

    resolved_target = str(protected / "boom.py")

    from tools.file_tools import write_file_tool

    def _guard_raises(cmd, cwd):
        raise RuntimeError("simulated guard failure")

    with patch("tools.file_tools._resolve_path_for_task", return_value=Path(resolved_target)), \
         patch("tools.file_tools.check_protected_root_write_guard", side_effect=_guard_raises), \
         patch("tools.file_tools._get_file_ops") as mock_ops:
        result = json.loads(write_file_tool("boom.py", "data"))

    assert "error" in result, f"expected error on guard exception, got: {result}"
    # The error payload should carry the guard-error type
    mock_ops.assert_not_called()


def test_patch_guard_exception_is_fail_closed(monkeypatch, tmp_path):
    """If check_protected_root_write_guard raises during patch, the patch is refused."""
    protected = tmp_path / "protected_repo"
    protected.mkdir()

    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(protected))

    resolved_target = str(protected / "core.py")

    from tools.file_tools import patch_tool

    def _guard_raises(cmd, cwd):
        raise RuntimeError("simulated guard failure in patch")

    with patch("tools.file_tools._resolve_path_for_task", return_value=Path(resolved_target)), \
         patch("tools.file_tools.check_protected_root_write_guard", side_effect=_guard_raises), \
         patch("tools.file_tools._get_file_ops") as mock_ops:
        result = json.loads(patch_tool(
            mode="replace",
            path="core.py",
            old_string="old",
            new_string="new",
        ))

    assert "error" in result, f"expected error on guard exception, got: {result}"
    mock_ops.assert_not_called()


# ---------------------------------------------------------------------------
# (e) linked worktree (.worktrees/) is still allowed — regression guard
# ---------------------------------------------------------------------------
def test_write_file_linked_worktree_still_allowed(monkeypatch, tmp_path):
    """write_file to a path inside .worktrees/ is NOT blocked (exempt by design)."""
    protected = tmp_path / "protected_repo"
    worktree = protected / ".worktrees" / "task-001"
    worktree.mkdir(parents=True)

    monkeypatch.setenv("HERMES_PROTECTED_GIT_ROOTS", str(protected))

    resolved_target = str(worktree / "tools" / "agent.py")
    Path(resolved_target).parent.mkdir(parents=True, exist_ok=True)

    from tools.file_tools import write_file_tool

    ops = MagicMock()
    written = MagicMock()
    written.to_dict.return_value = {"status": "ok"}
    ops.write_file.return_value = written

    with patch("tools.file_tools._resolve_path_for_task", return_value=Path(resolved_target)), \
         patch("tools.file_tools._get_file_ops", return_value=ops):
        result = json.loads(write_file_tool("agent.py", "# safe content"))

    assert result.get("status") == "ok" or not result.get("error"), \
        f"write to linked worktree should succeed, got: {result}"
    ops.write_file.assert_called_once()
