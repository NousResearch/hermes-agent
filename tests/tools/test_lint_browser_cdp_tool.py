"""Lint regression tests for tools/browser_cdp_tool.py.

Ensures specific rules have zero violations in this file.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_browser_cdp_tool_has_zero_f401_violations():
    """tools/browser_cdp_tool.py must have zero F401 (unused-import) violations."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select=F401",
         "--output-format=concise", "tools/browser_cdp_tool.py"],
        capture_output=True, text=True, check=False,
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 0, (
        "tools/browser_cdp_tool.py has F401 violation(s):\n"
        + result.stdout
    )


def test_env_probe_has_zero_f401_violations():
    """tools/env_probe.py must have zero F401 (unused-import) violations."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select=F401",
         "--output-format=concise", "tools/env_probe.py"],
        capture_output=True, text=True, check=False,
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 0, (
        "tools/env_probe.py has F401 violation(s):\n"
        + result.stdout
    )
