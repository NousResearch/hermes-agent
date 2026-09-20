"""Verify that terminal command timeouts preserve partial output."""
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
