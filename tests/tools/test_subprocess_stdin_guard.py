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

    The call's owner moved from agent/anthropic_adapter.py into
    agent/anthropic_credentials.py in the adapter godfile split; the guard
    scans both seams so a future move fails loudly instead of going dark.
    """
    candidates = [
        REPO_ROOT / "agent" / "anthropic_credentials.py",
        REPO_ROOT / "agent" / "anthropic_adapter.py",
    ]
    sources = [p.read_text() for p in candidates if p.exists()]
    owners = [
        src for src in sources
        if 'subprocess.run([claude_path, "setup-token"])' in src
    ]
    assert owners, (
        "interactive setup-token call changed shape or moved; re-verify it "
        "still inherits stdin (no stdin=subprocess.DEVNULL) and update this "
        "guard's candidate list"
    )
    for src in sources:
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



def test_splatted_kwargs_helper_counts_only_when_it_sets_stdin():
    """``**_KW`` / ``**_kw()`` splats are safe only when the same-file definition sets stdin=."""
    guard = _load_guard()
    safe_const = (
        "import subprocess\n"
        "_KW = dict(capture_output=True, stdin=subprocess.DEVNULL)\n"
        "subprocess.run(['ls'], **_KW)\n"
    )
    safe_fn = (
        "import subprocess\n"
        "def _kw(timeout):\n"
        "    return dict(timeout=timeout, stdin=subprocess.DEVNULL)\n"
        "subprocess.run(['ls'], **_kw(3))\n"
    )
    unsafe_const = (
        "import subprocess\n"
        "_KW = dict(capture_output=True, text=True)\n"
        "subprocess.run(['ls'], **_KW)\n"
    )
    undefined = "import subprocess\nsubprocess.run(['ls'], **_KW)\n"
    # A later unrelated call passing stdin= must NOT vouch for the splat (reviewer-found false negative
    # in the 30-line text-window version).
    unrelated_later = (
        "import subprocess\n"
        "kwargs = {'capture_output': True}\n"
        "subprocess.run(['child'], **kwargs)\n"
        "subprocess.run(['other'], stdin=subprocess.DEVNULL)\n"
    )
    assert guard.find_subprocess_calls(safe_const, "x.py") == []
    assert guard.find_subprocess_calls(safe_fn, "x.py") == []
    assert len(guard.find_subprocess_calls(unsafe_const, "x.py")) == 1
    assert len(guard.find_subprocess_calls(undefined, "x.py")) == 1
    assert [v["line"] for v in guard.find_subprocess_calls(unrelated_later, "x.py")] == [3]
