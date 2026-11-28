"""A running foreground terminal command publishes its newest output line as ``tool.progress``."""

import json

from agent.tool_progress_relay import bound_tool_progress

SECRET = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
SLOW = (f"echo 'token {SECRET}'; sleep 0.8; for i in 1 2 3; do printf 'building step %s of 3\\r' $i; sleep 0.8; done;"
        " echo; echo finished")


def test_foreground_terminal_streams_redacted_newest_line(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    from tools.registry import registry
    import tools.terminal_tool  # registers the tool

    events = []
    with bound_tool_progress(lambda *a, **kw: events.append((a, kw)), "terminal", "call-3"):
        result = json.loads(registry.dispatch("terminal", {"command": SLOW, "timeout": 30}))
    assert "finished" in result["output"]

    lines = [a[2] for a, _ in events]
    assert all(a[0] == "tool.progress" and a[1] == "terminal" for a, _ in events)
    assert {kw["tool_call_id"] for _, kw in events} == {"call-3"}
    # \r redraws count as lines, and the secret printed first never reaches the UI unredacted.
    assert "building step 3 of 3" in lines and lines[0].startswith("token ")
    assert not any(SECRET in line for line in lines)


def test_internal_execute_reads_stay_silent(tmp_path):
    """Only the foreground terminal tool (bounded_capture) previews; file-op/RPC reads do not."""
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    events = []
    try:
        with bound_tool_progress(lambda *a, **kw: events.append(a), "read_file", "call-4"):
            assert "two" in env.execute("echo one; sleep 1.5; echo two")["output"]
    finally:
        env.cleanup()
    assert events == []
