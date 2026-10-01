"""Hermes's own happenings reach workflows as events, through the Relay trace."""

from workflow import events
from workflow.store import save_documents


def _listening(pattern: str) -> dict:
    return {
        "id": "react", "name": "react",
        "scenario": {
            "steps": [{"id": "go", "kind": "trigger", "config": {"on": {"type": "event", "spec": pattern}}}],
            "edges": [],
        },
    }


def _tool_call_in_a_session(session_id: str, tool: str) -> None:
    from agent import relay_runtime

    import hermes_cli.observability  # noqa: F401  (recorder + workflow bridge)

    key = relay_runtime.current_profile_key()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(profile_key=key, session_id=session_id, platform="cli")
    host, session = lease.host, lease.session
    span = host.run_in_session(session, host.relay.tools.call, tool, {"path": "x"}, handle=session.handle)
    host.run_in_session(session, host.relay.tools.call_end, span, host.relay.ToolExecutionResult({"ok": True}))
    relay_runtime.SESSION_COORDINATOR.finalize_conversation(profile_key=key, session_id=session_id)
    host.relay.subscribers.flush()


def test_a_tool_call_is_published_only_when_a_workflow_listens(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    save_documents([_listening("hermes.tool.write_*")], "react")

    _tool_call_in_a_session("chat-1", "read_file")
    _tool_call_in_a_session("chat-1", "write_file")

    published = [(e["name"], e["payload"]["sessionId"]) for e in events.drain()]
    assert ("hermes.tool.write_file", "chat-1") in published
    assert all(name != "hermes.tool.read_file" for name, _ in published)
    assert all(not name.startswith("hermes.session.") for name, _ in published)


def test_events_from_a_workflow_run_name_their_workflow(tmp_path, monkeypatch):
    """What lets the runner refuse to re-trigger a workflow from its own steps."""
    from workflow.store import save_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    save_documents([_listening("hermes.tool.*")], "react")
    save_run({"runId": "run-1-abc123", "workflowId": "react", "status": "running"})

    _tool_call_in_a_session("wf-run-1-abc123-work", "terminal")

    (event,) = [e for e in events.drain() if e["name"] == "hermes.tool.terminal"]
    assert event["payload"]["workflowId"] == "react"
    assert event["payload"]["workflowRunId"] == "run-1-abc123"
