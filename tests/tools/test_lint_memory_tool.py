"""Regression tests: tools/memory_tool.py must be clean of select lint rules."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_memory_tool_has_zero_f401_violations() -> None:
    """tools/memory_tool.py must have zero F401 (unused-import) violations."""
    target = REPO_ROOT / "tools" / "memory_tool.py"
    assert target.exists(), f"Target file not found: {target}"

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select=F401",
         str(target), "--output-format=concise"],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, (
        "tools/memory_tool.py has F401 violation(s):\n"
        f"{result.stdout}\n{result.stderr}"
    )
