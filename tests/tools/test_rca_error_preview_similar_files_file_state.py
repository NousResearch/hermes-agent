"""Regression tests for the 2026-10-31 RCA fixes:
- RC-1: _extract_error_preview preserves terminal exit_code + leaf exception line across 200-char truncation.
- RC-2: _suggest_similar_files stem-prefix and stem-set overlap matches abbreviated script stems.
- RC-3: FileStateRegistry.check_stale uses sha256 content-identity fallback when st_mtime moves with 0 byte delta.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from agent.tool_dispatch_helpers import _extract_error_preview
from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_state import FileStateRegistry


def test_extract_error_preview_preserves_leaf_exception_and_exit_code():
    payload = json.dumps({
        "output": (
            'Traceback (most recent call last):\n'
            '  File "/Users/emirsaffar/workspace/research/remarkable-os-browser-hermes-panel-2026-10-31/scripts/rm_hermes_control_panel.py", line 225, in <module>\n'
            '    main()\n'
            '    ~~~~^^\n'
            '  File "/Users/emirsaffar/workspace/research/remarkable-os-browser-hermes-panel-2026-10-31/scripts/rm_hermes_control_panel.py", line 79, in query_sessions\n'
            '    conn.execute("SELECT purpose FROM sessions")\n'
            'sqlite3.OperationalError: no such column: purpose\n'
        ),
        "exit_code": 1,
        "error": None,
    })
    preview = _extract_error_preview(payload, max_len=240)
    assert len(preview) <= 240
    assert "sqlite3.OperationalError: no such column: purpose" in preview
    assert "[exit_code=1]" in preview
    assert "~~~~^^" not in preview


def test_suggest_similar_files_matches_abbreviated_script_stems(tmp_path: Path):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for name in [
        "cap_src02_rmscene_ink_ocr.py",
        "cap_src10_appload_dashboards.py",
        "rm_hermes_control_panel.py",
    ]:
        (scripts_dir / name).write_text("# test\n")

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30, persistent=False)
    ops = ShellFileOperations(env, cwd=str(tmp_path))
    r10 = ops.read_file(str(scripts_dir / "cap_src10.py"))
    assert str(scripts_dir / "cap_src10_appload_dashboards.py") in (r10.similar_files or [])

    r02 = ops.read_file(str(scripts_dir / "cap_src02.py"))
    assert str(scripts_dir / "cap_src02_rmscene_ink_ocr.py") in (r02.similar_files or [])

    r_neg = ops.read_file(str(scripts_dir / "zzzz_nonexistent_9999.xyz"))
    assert not (r_neg.similar_files or [])


def test_file_state_sha256_fallback_allows_noop_mtime_bump_and_blocks_real_edit(tmp_path: Path):
    target = tmp_path / "draft.txt"
    content = "Hello Luca\nBody line"
    target.write_text(content)
    resolved = str(target.resolve())

    reg = FileStateRegistry()
    reg.note_write("agent-1", resolved)
    assert reg.check_stale("agent-1", resolved) is None

    # Bump mtime by +5s with identical bytes (0 byte delta)
    st = os.stat(resolved)
    os.utime(resolved, (st.st_atime, st.st_mtime + 5.0))
    assert reg.check_stale("agent-1", resolved) is None

    # Modify 1 byte and bump mtime by +10s -> must block
    target.write_text(content + "!")
    os.utime(resolved, (st.st_atime, st.st_mtime + 10.0))
    stale_msg = reg.check_stale("agent-1", resolved)
    assert stale_msg is not None
    assert "modified since you last read it" in stale_msg
