"""Verify that terminal command timeouts preserve partial output."""
import tools.environments.base as base_module
from tools.environments.local import LocalEnvironment


class TestTimeoutPreservesPartialOutput:
    """When a command times out, any output captured before the deadline
    should be included in the result — not discarded."""

    def test_timeout_includes_partial_output(self):
        """A command that prints then sleeps past the deadline should
        return both the printed text and the timeout notice."""
        env = LocalEnvironment()
        result = env.execute("echo 'hello from test' && sleep 30", timeout=2)

        assert result["returncode"] == 124
        assert "hello from test" in result["output"]
        assert "timed out" in result["output"].lower()

    def test_timeout_with_no_output(self):
        """A command that produces nothing before timeout should still
        return a clean timeout message."""
        env = LocalEnvironment()
        result = env.execute("sleep 30", timeout=1)

        assert result["returncode"] == 124
        assert "timed out" in result["output"].lower()
        assert not result["output"].startswith("\n")

    def test_output_activity_extends_foreground_timeout(self):
        """Periodic output should reset the inactivity timer for foreground runs."""
        env = LocalEnvironment()
        result = env.execute(
            (
                "python3 -c \"import sys,time; "
                "print('tick-1', flush=True); "
                "time.sleep(0.6); "
                "print('tick-2', flush=True); "
                "time.sleep(0.6); "
                "print('done', flush=True)\""
            ),
            timeout=1,
        )

        assert result["returncode"] == 0
        assert "tick-1" in result["output"]
        assert "tick-2" in result["output"]
        assert "done" in result["output"]

    def test_chatty_command_still_hits_hard_wall_timeout(self, monkeypatch):
        """Continuous output should not bypass the hard wall-clock cap."""
        monkeypatch.setattr(base_module, "_ACTIVITY_TIMEOUT_WALL_MULTIPLIER", 2)
        monkeypatch.setattr(base_module, "_ACTIVITY_TIMEOUT_WALL_MIN_GRACE_SECONDS", 0)

        env = LocalEnvironment()
        result = env.execute(
            (
                "python3 -c \"import sys,time; "
                "i = 0\n"
                "while True:\n"
                " print(f'tick-{i}', flush=True)\n"
                " i += 1\n"
                " time.sleep(0.2)\""
            ),
            timeout=1,
        )

        assert result["returncode"] == 124
        assert "hard timeout" in result["output"].lower()
        assert "tick-" in result["output"]
