"""Regression tests: tools/cronjob_tools.py must be clean of select lint rules."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_cronjob_tools_has_zero_f401_violations() -> None:
    """tools/cronjob_tools.py must have zero F401 (unused-import) violations."""
    target = REPO_ROOT / "tools" / "cronjob_tools.py"
    assert target.exists(), f"Target file not found: {target}"

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select=F401",
         str(target), "--output-format=concise"],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, (
        "tools/cronjob_tools.py has F401 violation(s):\n"
        f"{result.stdout}{result.stderr}"
    )
