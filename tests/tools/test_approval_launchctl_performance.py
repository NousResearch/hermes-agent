"""Long non-launchctl inputs must not starve other Gateway threads.

Subprocess timeout bounds regressions without hanging pytest on the GIL.
"""
import subprocess
import sys


def test_launchctl_guard_long_negative_input_is_bounded():
    code = '''
from tools.approval_detection import DANGEROUS_PATTERNS_COMPILED
rules = [(rx, desc) for rx, desc in DANGEROUS_PATTERNS_COMPILED
         if desc == "stop/restart hermes launchd service (kills running agents)"]
assert len(rules) == 1
rx = rules[0][0]
assert rx.search("x" * 100_000) is None
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=5)
