"""steer_subagent — redirecting a live delegated child without stopping it.

Registry-level coverage for the delegation-side mirror of
interrupt_subagent(): text reaches the live child's AIAgent.steer(), and
every failure shape (unknown id, dead record, empty text, a steer that
raises) degrades to False instead of an exception.  Also covers the
missed-steer retention race (a child that finishes before the drain) and
the subagent.steer gateway RPC that fronts the helper.
"""

from tools.delegate_tool import (
    _register_subagent,
    _unregister_subagent,
    steer_subagent,
)


class _StubAgent:
    def __init__(self, accept: bool = True, boom: bool = False):
        self.accept = accept
        self.boom = boom
        self.steered: list[str] = []

    def steer(self, text: str) -> bool:
        if self.boom:
            raise RuntimeError("steer exploded")
        self.steered.append(text)
        return self.accept


def _with_registered(sid: str, agent) -> None:
    _register_subagent(
        {
            "subagent_id": sid,
            "parent_id": "root",
            "depth": 1,
            "goal": "test goal",
            "status": "running",
            "agent": agent,
        }
    )


def test_steer_reaches_the_live_child():
    agent = _StubAgent()
    _with_registered("sid-steer-1", agent)
    try:
        assert steer_subagent("sid-steer-1", "focus on pricing instead") is True
        assert agent.steered == ["focus on pricing instead"]
    finally:
        _unregister_subagent("sid-steer-1")


def test_unknown_subagent_is_false_not_an_error():
    assert steer_subagent("sid-not-registered", "hello") is False


def test_empty_text_is_refused_without_a_lookup():
    agent = _StubAgent()
    _with_registered("sid-steer-2", agent)
    try:
        assert steer_subagent("sid-steer-2", "   ") is False
        assert agent.steered == []
    finally:
        _unregister_subagent("sid-steer-2")


def test_record_without_live_agent_is_false():
    _register_subagent({"subagent_id": "sid-steer-3", "status": "running", "agent": None})
    try:
        assert steer_subagent("sid-steer-3", "hello") is False
    finally:
        _unregister_subagent("sid-steer-3")


def test_agent_rejection_propagates_as_false():
    agent = _StubAgent(accept=False)
    _with_registered("sid-steer-4", agent)
    try:
        assert steer_subagent("sid-steer-4", "hello") is False
    finally:
        _unregister_subagent("sid-steer-4")


def test_exception_in_steer_degrades_to_false():
    agent = _StubAgent(boom=True)
    _with_registered("sid-steer-5", agent)
    try:
        assert steer_subagent("sid-steer-5", "hello") is False
    finally:
        _unregister_subagent("sid-steer-5")


class TestMissedSteerRetention:
    """The final-answer race: a steer with no boundary left is NAMED, not lost."""

    def test_pending_steer_lands_in_completion_entry(self):
        import json
        from unittest.mock import MagicMock, patch

        from tools.delegate_tool import delegate_task

        parent = MagicMock()
        parent._delegate_depth = 0
        parent.model = "test-model"
        parent.interactive_mode = False

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "test-model"
            mock_child.session_prompt_tokens = 0
            mock_child.session_completion_tokens = 0
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [],
                # The finalizer's undelivered-steer hand-back
                # (turn_finalizer.py "pending_steer").
                "pending_steer": "focus on pricing instead",
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="race test", parent_agent=parent))
            entry = result["results"][0]

        assert entry["missed_steer"] == "focus on pricing instead"
        assert "steer did not land" in entry["summary"]
        assert "focus on pricing instead" in entry["summary"]
        # The race must not corrupt the outcome of the work itself.
        assert entry["status"] == "completed"

    def test_no_pending_steer_leaves_entry_untouched(self):
        import json
        from unittest.mock import MagicMock, patch

        from tools.delegate_tool import delegate_task

        parent = MagicMock()
        parent._delegate_depth = 0
        parent.model = "test-model"
        parent.interactive_mode = False

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.model = "test-model"
            mock_child.session_prompt_tokens = 0
            mock_child.session_completion_tokens = 0
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [],
            }
            MockAgent.return_value = mock_child

            result = json.loads(delegate_task(goal="clean run", parent_agent=parent))
            entry = result["results"][0]

        assert "missed_steer" not in entry
        assert "steer did not land" not in entry["summary"]


class TestSubagentSteerRPC:
    """subagent.steer gateway RPC — the programmatic caller beside subagent.interrupt."""

    def _call(self, params: dict) -> dict:
        import tui_gateway.server as srv

        return srv._methods["subagent.steer"](1, params)

    def test_missing_subagent_id_is_4000(self):
        envelope = self._call({"text": "hello"})
        assert envelope["error"]["code"] == 4000

    def test_empty_text_is_4002(self):
        envelope = self._call({"subagent_id": "sid-rpc-1", "text": "   "})
        assert envelope["error"]["code"] == 4002

    def test_live_child_queues_and_receives_text(self):
        agent = _StubAgent()
        _with_registered("sid-rpc-2", agent)
        try:
            envelope = self._call({"subagent_id": "sid-rpc-2", "text": "check the edge cases"})
            assert envelope["result"] == {
                "status": "queued",
                "subagent_id": "sid-rpc-2",
                "text": "check the edge cases",
            }
            assert agent.steered == ["check the edge cases"]
        finally:
            _unregister_subagent("sid-rpc-2")

    def test_unknown_child_is_rejected_not_an_error(self):
        envelope = self._call({"subagent_id": "sid-rpc-gone", "text": "hello"})
        assert envelope["result"]["status"] == "rejected"
