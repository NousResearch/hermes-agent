"""Verify that TUI-context subprocess calls specify stdin=.

This is the pytest wrapper for scripts/check_subprocess_stdin.py.
It runs as part of the test suite so CI catches regressions when new
subprocess calls are added without stdin=subprocess.DEVNULL.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_subprocess_stdin.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("_stdin_guard", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_oauth_setup_token_keeps_inherited_stdin():
    """The interactive 'claude setup-token' login must NOT be muzzled.

    Forcing stdin=subprocess.DEVNULL here would feed the OAuth prompt EOF and
    break interactive token setup. A blanket DEVNULL sweep over TUI-context
    subprocess calls must leave this one inheriting stdin. Regression guard for
    the over-application caught while salvaging the stdin-EOF fix.
    """
    src = (REPO_ROOT / "agent" / "anthropic_adapter.py").read_text()
    assert 'subprocess.run([claude_path, "setup-token"])' in src, (
        "interactive setup-token call changed shape; re-verify it still "
        "inherits stdin (no stdin=subprocess.DEVNULL)"
    )
    assert 'subprocess.run([claude_path, "setup-token"], stdin' not in src, (
        "setup-token must inherit stdin so the user can complete the OAuth "
        "login prompt; do not add stdin=subprocess.DEVNULL"
    )


def test_inline_noqa_marker_exempts_a_call():
    """The guard honors an inline 'noqa: subprocess-stdin' exemption marker."""
    guard = _load_guard()
    flagged = guard.find_subprocess_calls(
        "import subprocess\nsubprocess.run(['ls'])\n", "x.py"
    )
    assert len(flagged) == 1, "unmarked missing-stdin call should be flagged"

    exempt = guard.find_subprocess_calls(
        "import subprocess\nsubprocess.run(['ls'])  # noqa: subprocess-stdin\n",
        "x.py",
    )
    assert exempt == [], "inline marker should exempt the call"
