"""Invariant tests for delegation.shared_notes (Cursor Projects-inspired shared context).

Two contracts:
1. Default off: the child system prompt is byte-identical to a build with the module absent —
   no notes block, no append hint (prompt-cache / default-behavior safety).
2. Enabled: prior notes are injected as untrusted context (newest tail survives the cap) and
   the child is told the exact profile-scoped append path, which never lives inside the repo.
"""

import os

import pytest


@pytest.fixture
def notes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    workspace = tmp_path / "repo"
    workspace.mkdir()
    return home, workspace


def _prompt(workspace, cfg):
    from unittest.mock import patch

    from tools.delegate_tool_progress import _build_child_system_prompt

    with patch("tools.delegate_tool._load_config", return_value=cfg):
        return _build_child_system_prompt("do the task", None, workspace_path=str(workspace))


def test_shared_notes_off_by_default_prompt_unchanged(notes_env):
    """Off (default) leaves the child prompt without any notes machinery, even when a notes
    file exists for the workspace — the flag, not the file, is the gate."""
    home, workspace = notes_env
    from tools.delegate_tool_shared_notes import shared_notes_path

    path = shared_notes_path(str(workspace))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("- run tests via scripts/run_tests.sh\n", encoding="utf-8")

    prompt = _prompt(workspace, {})
    assert "SHARED DELEGATION NOTES" not in prompt
    assert "shared notes file" not in prompt
    assert str(path) not in prompt


def test_shared_notes_enabled_injects_tail_and_append_path(notes_env):
    """Enabled: existing notes appear as untrusted context with the newest tail surviving the
    cap, and the append path is profile-scoped (under HERMES_HOME, never the workspace)."""
    home, workspace = notes_env
    from tools.delegate_tool_shared_notes import (
        SHARED_NOTES_MAX_CHARS,
        shared_notes_path,
    )

    path = shared_notes_path(str(workspace))
    path.parent.mkdir(parents=True, exist_ok=True)
    filler = "- old note that should be truncated away\n" * (SHARED_NOTES_MAX_CHARS // 20)
    path.write_text(filler + "- NEWEST: build needs node 20\n", encoding="utf-8")

    prompt = _prompt(workspace, {"shared_notes": True})
    assert "SHARED DELEGATION NOTES" in prompt
    assert "NEWEST: build needs node 20" in prompt  # tail-first cap keeps latest learnings
    assert str(path) in prompt  # child knows where to append
    # State stays out of the repo and under the profile home.
    assert os.path.commonpath([str(path), str(home)]) == str(home)
    assert not str(path).startswith(str(workspace))
