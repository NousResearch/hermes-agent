"""Lint regression tests for tools/ directory.

These tests guard against reintroduction of previously-fixed lint violations.
Each test asserts zero violations of a specific rule in a specific file.

TDD: Write the test before fixing the violation. Watch it fail. Then fix.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable


def test_tools_patch_parser_has_zero_f821_violations() -> None:
    """tools/patch_parser.py must have zero F821 (undefined name) violations."""
    target = REPO_ROOT / "tools" / "patch_parser.py"
    assert target.exists(), f"Target file not found: {target}"

    result = subprocess.run(
        [PYTHON, "-m", "ruff", "check", "--select=F821",
         "--output-format=concise", str(target)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f"tools/patch_parser.py has F821 violation(s):\n"
        f"{result.stdout}{result.stderr}"
    )


def test_tools_browser_cdp_tool_has_zero_f401_violations() -> None:
    """tools/browser_cdp_tool.py must have zero F401 (unused import) violations."""
    target = REPO_ROOT / "tools" / "browser_cdp_tool.py"
    assert target.exists(), f"Target file not found: {target}"

    result = subprocess.run(
        [PYTHON, "-m", "ruff", "check", "--select=F401",
         "--output-format=concise", str(target)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f"tools/browser_cdp_tool.py has F401 violation(s):\n"
        f"{result.stdout}{result.stderr}"
    )


def test_tools_browser_tool_has_zero_f401_violations() -> None:
    """tools/browser_tool.py must have zero F401 (unused import) violations."""
    target = REPO_ROOT / "tools" / "browser_tool.py"
    assert target.exists(), f"Target file not found: {target}"

    result = subprocess.run(
        [PYTHON, "-m", "ruff", "check", "--select=F401",
         "--output-format=concise", str(target)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f"tools/browser_tool.py has F401 violation(s):\n"
        f"{result.stdout}{result.stderr}"
    )

def test_tools_memory_tool_has_zero_f401_violations() -> None:
    """tools/memory_tool.py must have zero F401 (unused import) violations."""
    target = REPO_ROOT / "tools" / "memory_tool.py"
    assert target.exists(), f"Target file not found: {target}"

    result = subprocess.run(
        [PYTHON, "-m", "ruff", "check", "--select=F401",
         "--output-format=concise", str(target)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f"tools/memory_tool.py has F401 violation(s):\n"
        f"{result.stdout}{result.stderr}"
    )
