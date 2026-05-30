import json

from gateway.subagent_events import SubagentEventStore
from tools.ao_bridge import AOSession
from tools.ao_delegate_tool import (
    _output_indicates_codex_complete,
    _summary_from_completed_output,
    ao_delegate_batch,
    ao_delegate_task,
    build_ao_worker_prompt,
)


class FakeBridge:
    def __init__(self):
        self.spawned = AOSession(
            id="oryn-workspace-1",
            project_id="OrynWorkspace",
            status="working",
            activity="active",
            branch="feat/test",
            workspace_path="/tmp/worktree",
            tmux_name="abc-oryn-workspace-1",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="medium",
            open_command="tmux attach -t abc-oryn-workspace-1",
        )
        self.done = AOSession(
            id="oryn-workspace-1",
            project_id="OrynWorkspace",
            status="done",
            activity="exited",
            branch="feat/test",
            workspace_path="/tmp/worktree",
            tmux_name="abc-oryn-workspace-1",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="medium",
            open_command="tmux attach -t abc-oryn-workspace-1",
        )
        self.status_calls = 0

    def spawn(self, **kwargs):
        self.spawn_kwargs = kwargs
        return self.spawned

    def status(self, session_id):
        self.status_calls += 1
        return self.done

    def capture_output(self, session, lines=40):
        return "worker finished\npytest passed"


class FakeBatchBridge:
    def __init__(self, fail_indexes=None):
        self.fail_indexes = set(fail_indexes or [])
        self.spawn_kwargs = []

    def spawn(self, **kwargs):
        index = len(self.spawn_kwargs) + 1
        self.spawn_kwargs.append(kwargs)
        if index in self.fail_indexes:
            raise RuntimeError("spawn failed")
        return AOSession(
            id=f"oryn-workspace-{index}",
            project_id=kwargs.get("project_id"),
            status="working",
            activity="active",
            branch=f"feat/task-{index}",
            workspace_path=f"/tmp/worktree-{index}",
            tmux_name=f"tmux-{index}",
            agent="codex",
            model="gpt-5.5",
            reasoning_effort="medium",
            open_command=f"tmux attach -t tmux-{index}",
        )


class FakeBridgeWithCompletedTUI(FakeBridge):
    def __init__(self):
        super().__init__()
        self.done.status = "spawning"
        self.done.activity = None

    def capture_output(self, session, lines=40):
        return """
› You are an AI coding agent managed by the Agent Orchestrator (ao).

────────────────────────────────────────────────────────────────────────────────

• FOUND_PANEL
  pwd: /tmp/worktree

────────────────────────────────────────────────────────────────────────────────

› Improve documentation in @filename

  gpt-5.5 medium · /tmp/worktree
"""


class FakeBridgeWithIntermediateTUI(FakeBridge):
    def __init__(self):
        super().__init__()
        self.done.status = "spawning"
        self.done.activity = None
        self.outputs = [
            """
────────────────────────────────────────────────────────────────────────────────

• The relevant logic is small. I’m grabbing exact line numbers now.

────────────────────────────────────────────────────────────────────────────────

› Find and fix a bug in @filename
""",
            """
────────────────────────────────────────────────────────────────────────────────

• AO_PANEL_DONE
  Open and Stop are present.

────────────────────────────────────────────────────────────────────────────────

› Find and fix a bug in @filename
""",
            """
────────────────────────────────────────────────────────────────────────────────

• AO_PANEL_DONE
  Open and Stop are present.

────────────────────────────────────────────────────────────────────────────────

› Find and fix a bug in @filename
""",
        ]

    def capture_output(self, session, lines=40):
        if self.outputs:
            return self.outputs.pop(0)
        return """
────────────────────────────────────────────────────────────────────────────────

• AO_PANEL_DONE
  Open and Stop are present.

────────────────────────────────────────────────────────────────────────────────

› Find and fix a bug in @filename
"""


class ParentAgent:
    def __init__(self):
        self.events = []

    def tool_progress_callback(self, event_type, tool_name=None, preview=None, **kwargs):
        self.events.append((event_type, tool_name, preview, kwargs))


def test_ao_delegate_task_emits_subagent_events_and_returns_session():
    parent = ParentAgent()
    bridge = FakeBridge()
    result = ao_delegate_task(
        prompt="Run the test task",
        goal="Test AO worker",
        project_id="OrynWorkspace",
        branch="codex/test-branch",
        max_wait_seconds=5,
        parent_agent=parent,
        bridge=bridge,
    )

    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["runtime"] == "ao"
    assert payload["session"]["ao_session_id"] == "oryn-workspace-1"
    assert payload["session"]["workspace_path"] == "/tmp/worktree"
    assert bridge.spawn_kwargs["branch"] == "codex/test-branch"
    assert "## Hermes AO Delegation Contract" in bridge.spawn_kwargs["prompt"]
    assert "Run the test task" in bridge.spawn_kwargs["prompt"]
    assert parent.events
    assert parent.events[0][3]["branch"] == "feat/test"
    assert parent.events[0][3]["_ao_prompt_metadata"]["prompt"] == "Run the test task"
    assert parent.events[0][3]["_ao_prompt_metadata"]["agent"] == "codex"
    assert parent.events[0][3]["_ao_prompt_metadata"]["model"] == "gpt-5.5"
    assert parent.events[0][3]["_ao_prompt_metadata"]["reasoning_effort"] == "medium"

    event_names = [event[0] for event in parent.events]
    assert "subagent.start" in event_names
    assert "subagent.progress" in event_names
    assert "subagent.complete" in event_names

    complete = parent.events[-1]
    assert complete[3]["runtime"] == "ao"
    assert complete[3]["ao_project_id"] == "OrynWorkspace"
    assert complete[3]["branch"] == "feat/test"
    assert complete[3]["status"] == "completed"


def test_ao_delegate_batch_spawns_multiple_workers_and_emits_start_events():
    parent = ParentAgent()
    bridge = FakeBatchBridge()
    result = ao_delegate_batch(
        tasks=[
            {"goal": "Inspect UI", "prompt": "Read the board UI", "project_id": "OrynWorkspace"},
            {"goal": "Inspect API", "prompt": "Read the board API", "project_id": "OrynPlatform"},
        ],
        parent_agent=parent,
        bridge=bridge,
    )

    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["session_count"] == 2
    assert payload["failure_count"] == 0
    assert [call["project_id"] for call in bridge.spawn_kwargs] == ["OrynWorkspace", "OrynPlatform"]
    assert "## Hermes AO Delegation Contract" in bridge.spawn_kwargs[0]["prompt"]
    assert "Read the board UI" in bridge.spawn_kwargs[0]["prompt"]
    assert bridge.spawn_kwargs[0]["prompt"] != "Read the board UI"
    assert [event[0] for event in parent.events] == ["subagent.start", "subagent.start"]
    assert parent.events[0][1] == "ao_delegate_batch"
    assert parent.events[0][3]["task_index"] == 1
    assert parent.events[1][3]["task_count"] == 2
    assert parent.events[0][3]["_ao_prompt_metadata"]["prompt"] == "Read the board UI"
    assert parent.events[0][3]["_ao_prompt_metadata"]["reasoning_effort"] == "medium"


def test_build_ao_worker_prompt_makes_brief_authoritative_and_is_idempotent():
    wrapped = build_ao_worker_prompt("Return only: unclear", goal="Weak summary test")

    assert wrapped.startswith("## Hermes AO Delegation Contract")
    assert "authoritative assignment" in wrapped
    assert "Return only: unclear" in wrapped
    assert build_ao_worker_prompt(wrapped, goal="Weak summary test") == wrapped


def test_ao_delegate_batch_reports_partial_failures():
    parent = ParentAgent()
    result = ao_delegate_batch(
        tasks=[
            {"goal": "One", "prompt": "Task one"},
            {"goal": "Two", "prompt": "Task two"},
        ],
        parent_agent=parent,
        bridge=FakeBatchBridge(fail_indexes={2}),
    )

    payload = json.loads(result)

    assert payload["ok"] is True
    assert payload["session_count"] == 1
    assert payload["failure_count"] == 1
    assert payload["failures"][0]["task_index"] == 2
    assert len(parent.events) == 1


def test_ao_delegate_batch_persists_start_events_without_parent_callback(tmp_path):
    store = SubagentEventStore(tmp_path / "state.db")
    result = ao_delegate_batch(
        tasks=[
            {"goal": "Verify Agent Board action history UI", "prompt": "Inspect actions", "project_id": "OrynWorkspace"},
            {"goal": "Verify weak summary warning behavior", "prompt": "Return unclear", "project_id": "OrynWorkspace"},
        ],
        bridge=FakeBatchBridge(),
        event_store=store,
    )

    payload = json.loads(result)
    events = store.list_events(limit=10)

    assert payload["ok"] is True
    assert [event["event"] for event in events] == ["subagent.start", "subagent.start"]
    assert [event["tool_name"] for event in events] == ["ao_delegate_batch", "ao_delegate_batch"]
    assert events[0]["goal"] == "Verify Agent Board action history UI"
    assert events[0]["task_index"] == 1
    assert events[1]["task_count"] == 2
    assert store.get_ao_prompt("oryn-workspace-1")["prompt"] == "Inspect actions"
    assert store.get_ao_prompt("oryn-workspace-1")["reasoning_effort"] == "medium"
    assert store.get_ao_prompt("oryn-workspace-2")["goal"] == "Verify weak summary warning behavior"
    store.close()


def test_ao_delegate_task_completes_when_codex_tui_returns_to_prompt(monkeypatch):
    monkeypatch.setattr("tools.ao_delegate_tool.time.sleep", lambda _: None)
    parent = ParentAgent()
    result = ao_delegate_task(
        prompt="Run the test task",
        goal="Test AO worker",
        project_id="OrynWorkspace",
        max_wait_seconds=5,
        parent_agent=parent,
        bridge=FakeBridgeWithCompletedTUI(),
    )

    payload = json.loads(result)

    assert payload["timed_out"] is False
    assert payload["status"] == "completed"
    assert "FOUND_PANEL" in payload["summary"]
    assert "Improve documentation" not in payload["summary"]
    assert parent.events[-1][0] == "subagent.complete"
    assert parent.events[-1][3]["status"] == "completed"


def test_ao_delegate_task_waits_for_stable_completed_tui(monkeypatch):
    monkeypatch.setattr("tools.ao_delegate_tool.time.sleep", lambda _: None)
    parent = ParentAgent()
    result = ao_delegate_task(
        prompt="Run the test task",
        goal="Test AO worker",
        project_id="OrynWorkspace",
        max_wait_seconds=5,
        parent_agent=parent,
        bridge=FakeBridgeWithIntermediateTUI(),
    )

    payload = json.loads(result)

    assert payload["timed_out"] is False
    assert payload["status"] == "completed"
    assert "AO_PANEL_DONE" in payload["summary"]
    assert "grabbing exact line numbers" not in payload["summary"]


def test_codex_completion_detection_requires_final_prompt_after_separator():
    incomplete = """
› You are an AI coding agent managed by the Agent Orchestrator (ao).

• Working (10s • esc to interrupt)
"""
    active_after_separator = """
────────────────────────────────────────────────────────────────────────────────

• The active project is set. I’m reading the requested file.

• Working (12s • esc to interrupt)

› Find and fix a bug in @filename
"""
    complete = """
────────────────────────────────────────────────────────────────────────────────

• Final answer

────────────────────────────────────────────────────────────────────────────────

› Next prompt
"""

    assert _output_indicates_codex_complete(incomplete) is False
    assert _output_indicates_codex_complete(active_after_separator) is False
    assert _output_indicates_codex_complete(complete) is True
    assert _summary_from_completed_output(complete) == "Final answer"
