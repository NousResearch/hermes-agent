"""The hardline floor's command-position groups stay linear on blank-line runs.

Each probe runs in a subprocess with a timeout: a regressed pattern holds the GIL in C (``re``) for
tens of seconds, so an in-process timing assertion would hang pytest.
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize("build", [
    # A busybox run after a blank-line run: every newline re-split the `busybox ` repeats.
    '"echo hi" + "\\n" * 400 + "busybox " * 50 + "x"',
    # Directory runs between blank lines: every newline re-split the path components.
    '("\\n" * 20 + "a/" * 20) * 300',
    '"echo hi" + "\\n" * 400 + "a/" * 200 + "x"',
])
def test_command_position_groups_finish_on_blank_line_runs(build):
    code = (
        "import time\n"
        "from tools.approval_detection import detect_dangerous_command, detect_hardline_command\n"
        "detect_hardline_command('ls')\n"
        f"command = {build}\n"
        "started = time.perf_counter()\n"
        "assert detect_hardline_command(command) == (False, None)\n"
        "detect_dangerous_command(command)\n"
        "assert time.perf_counter() - started < 3.0\n"
    )
    # Pin the checkout under test: a bare `python -c` resolves `tools` through the venv's editable
    # install (the primary clone), not through the worktree pytest is running in.
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60, cwd=REPO_ROOT, env=env)
