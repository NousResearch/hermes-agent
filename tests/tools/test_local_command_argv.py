"""A command's text never sits in the argv of the shell that runs it."""

import os
import subprocess
import sys

import pytest

from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="pkill/pgrep -f are POSIX tools")


def test_pattern_kill_does_not_kill_its_own_shell(tmp_path):
    """`pkill -f <pattern>` used to match the wrapper `bash -c "<command>"` and SIGTERM it
    mid-command (rc -15, no output); `pgrep -f` reported the wrapper as a phantom match."""
    marker = f"sleep {os.getpid() % 1000 + 7001}"
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    # Popen returns after exec, so both victims already carry the marker in their argv.
    victims = [subprocess.Popen(marker.split()) for _ in range(2)]
    try:
        counted = env.execute(f"pgrep -f '{marker}' | wc -l")
        assert counted["output"].strip() == "2"  # the two victims, no phantom wrapper

        killed = env.execute(f"pkill -f '{marker}'; echo after-kill")
        assert killed["returncode"] == 0
        assert "after-kill" in killed["output"]
        for v in victims:
            assert v.wait(timeout=5) is not None
        assert "HERMES_RUN_COMMAND" not in env.execute("env")["output"]
    finally:
        for v in victims:
            v.kill()
        env.cleanup()
