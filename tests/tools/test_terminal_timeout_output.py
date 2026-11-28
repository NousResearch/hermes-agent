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


def test_finalize_foreground_result_surfaces_structured_timeout_from_hermes_timed_out():
    import json
    from tools.terminal_tool_result import finalize_foreground_result

    env = LocalEnvironment()
    common = dict(
        command="sleep 30",
        env=env,
        env_type="local",
        effective_task_id="t",
        task_id="t",
        session_id="s",
        session_key="k",
        workdir=None,
        command_cwd=None,
        approval_note=None,
    )
    timed_out_payload = json.loads(
        finalize_foreground_result(
            result={
                "output": "partial line\n[Command timed out after 4s]",
                "returncode": 124,
                "hermes_timed_out": True,
            },
            **common,
        )
    )
    assert timed_out_payload["exit_code"] == 124
    assert "partial line" in timed_out_payload["output"]
    assert timed_out_payload["status"] == "timed_out"
    assert timed_out_payload["timed_out"] is True
    assert timed_out_payload["process_terminated"] is True
    assert isinstance(timed_out_payload["error"], str) and "timed out" in timed_out_payload["error"].lower()

    ordinary_124 = json.loads(
        finalize_foreground_result(
            result={"output": "custom exit 124", "returncode": 124},
            **common,
        )
    )
    assert ordinary_124["exit_code"] == 124
    assert ordinary_124["error"] is None
    assert "timed_out" not in ordinary_124
    assert "process_terminated" not in ordinary_124
    assert "status" not in ordinary_124
