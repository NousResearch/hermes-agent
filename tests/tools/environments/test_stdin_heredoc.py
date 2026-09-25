"""Stdin heredoc embedding for SDK backends (tools/environments/base.py).

Regression tests for #122011 (brace-group diagnosis from #94849 by JoshSnider):
appending ``<< DELIM`` to a compound command feeds only its LAST command, so
``cat`` read empty stdin and ``mv`` swapped an empty temp over the target.
The embedder instead attaches the heredoc to an inner ``cat`` and re-emits
the body minus the heredoc's own trailing newline, byte-exact.
"""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from tools.environments.base import BaseEnvironment


def test_heredoc_binds_inner_cat_not_command_tail():
    out = BaseEnvironment._embed_stdin_heredoc('cat > "$tmp"; trap - EXIT', "x")
    assert "cat << 'HERMES_STDIN_" in out
    assert out.count("HERMES_EOF_") == 2  # sentinel line + suffix pattern


def _working_bash():
    bash = shutil.which("bash")
    if bash is None:
        return None
    try:
        proc = subprocess.run([bash, "-c", "true"], capture_output=True, timeout=30)
    except Exception:
        return None
    return bash if proc.returncode == 0 else None


def test_byte_exact_through_real_bash(tmp_path):
    bash = _working_bash()
    if bash is None:
        pytest.skip("no working bash on PATH")
    bodies = [
        "no trailing newline",
        "line one\nline two\n",
        "",
        "a\n\n",
        "C:\\path \\ back $var `tick`",
        "trailing-X\nX",
        "cat # trailing comment",
    ]
    for body in bodies:
        target = tmp_path / "out.bin"
        if target.exists():
            target.unlink()
        full = BaseEnvironment._embed_stdin_heredoc(
            f'cat > "{target.as_posix()}"; trap - EXIT', body)
        proc = subprocess.run([bash, "-c", full], capture_output=True, timeout=60)
        assert proc.returncode == 0, (body, proc.stderr.decode("utf-8", "replace"))
        assert target.read_bytes() == body.encode("utf-8"), body
