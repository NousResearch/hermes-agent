"""Verify that TUI-context subprocess calls specify stdin=.

This is the pytest wrapper for scripts/check_subprocess_stdin.py.
It runs as part of the test suite so CI catches regressions when new
subprocess calls are added without stdin=subprocess.DEVNULL.
"""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_subprocess_stdin.py"


def test_all_tui_subprocess_calls_have_stdin():
    """Every subprocess.run/Popen in TUI-context code must set stdin=."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess stdin= check failed:\n{result.stdout}\n{result.stderr}"
    )
