"""Long non-launchctl inputs must not starve other Gateway threads.

Subprocess timeout bounds regressions without hanging pytest on the GIL.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_hardline_projection_long_heredoc_is_bounded():
    code = '''
from tools.approval_detection import detect_hardline_command
body = "\\n".join(f'  "key{i}": "value{i}",' for i in range(460))
command = "cat <<EOF > /tmp/x.json\\n{\\n" + body + "\\n}\\nEOF"
assert len(command) == 10_851
assert detect_hardline_command(command) == (False, None)
'''
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    subprocess.run([sys.executable, '-c', code], check=True, timeout=5, cwd=REPO_ROOT, env=env)


def test_launchctl_guard_long_negative_input_is_bounded():
    code = '''
from tools.approval_detection import DANGEROUS_PATTERNS_COMPILED
rules = [(rx, desc) for rx, desc in DANGEROUS_PATTERNS_COMPILED
         if desc == "stop/restart hermes launchd service (kills running agents)"]
assert len(rules) == 1
rx = rules[0][0]
assert rx.search("x" * 100_000) is None
'''
    # Pin the checkout under test: a bare `python -c` resolves `tools` through the venv's editable
    # install (the primary clone), not through the worktree pytest is running in.
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    subprocess.run([sys.executable, '-c', code], check=True, timeout=5, cwd=REPO_ROOT, env=env)
