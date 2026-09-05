"""steer_subagent — redirecting a live delegated child without stopping it.

Registry-level coverage for the delegation-side mirror of
interrupt_subagent(): text reaches the live child's AIAgent.steer(), and
every failure shape (unknown id, dead record, empty text, a steer that
raises) degrades to False instead of an exception.  Also covers the
missed-steer retention race (a child that finishes before the drain) and
the subagent.steer gateway RPC that fronts the helper.
"""

import threading
from unittest.mock import MagicMock

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


def _with_registered(
    sid: str,
    agent,
    *,
    owner_session_id: str | None = None,
    owner_transport=None,
    owner_session_record=None,
    owner_agent_session_id: str | None = None,
) -> None:
    _register_subagent(
        {
            "subagent_id": sid,
            "parent_id": "root",
            "depth": 1,
            "goal": "test goal",
            "status": "running",
            "agent": agent,
            "owner_session_id": owner_session_id,
            "owner_transport": owner_transport,
            "owner_session_record": owner_session_record,
            "owner_agent_session_id": owner_agent_session_id,
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


def test_stale_agent_teardown_cannot_unregister_recycled_id():
    old_agent = _StubAgent()
    replacement = _StubAgent()
    _with_registered("sid-recycled-teardown", old_agent, owner_session_id="old-owner")
    _with_registered("sid-recycled-teardown", replacement, owner_session_id="new-owner")
    try:
        _unregister_subagent("sid-recycled-teardown", agent=old_agent)
        assert (
            steer_subagent(
                "sid-recycled-teardown",
                "replacement remains live",
            )
            is True
        )
        assert old_agent.steered == []
        assert replacement.steered == ["replacement remains live"]
    finally:
        _unregister_subagent("sid-recycled-teardown", agent=replacement)


def test_status_snapshot_never_leaks_owner_or_lifecycle_metadata():
    from tools.delegate_tool import list_active_subagents

    agent = _StubAgent()
    owner_transport = object()
    owner_session_record = {"session_key": "private-owner"}
    _with_registered(
        "sid-private-metadata",
        agent,
        owner_session_id="private-owner",
        owner_transport=owner_transport,
        owner_session_record=owner_session_record,
        owner_agent_session_id="durable-owner",
    )
    try:
        snapshot = next(
            item
            for item in list_active_subagents()
            if item["subagent_id"] == "sid-private-metadata"
        )
        assert snapshot["status"] == "running"
        assert snapshot["owner_agent_session_id"] == "durable-owner"
        assert "agent" not in snapshot
        assert "owner_session_id" not in snapshot
        assert "owner_transport" not in snapshot
        assert "owner_session_record" not in snapshot
        assert "accepting_steer" not in snapshot
        assert "private-owner" not in repr(snapshot)
        assert all(value is not owner_transport for value in snapshot.values())
        assert all(value is not owner_session_record for value in snapshot.values())
    finally:
        _unregister_subagent("sid-private-metadata", agent=agent)


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

    def test_accepted_steer_racing_completion_is_durably_retained(self):
        """Acceptance wins the registry race, so completion must retain its text."""
        from tools.delegate_tool import _run_single_child

        running = threading.Event()
        allow_return = threading.Event()
        steer_entered = threading.Event()
        allow_steer = threading.Event()
        pending: list[str] = []

        child = MagicMock()
        child._subagent_id = "sid-linearized-accept"
        child._delegate_depth = 1
        child.model = "test-model"
        child.session_prompt_tokens = 0
        child.session_completion_tokens = 0

        def run_conversation(**_kwargs):
            running.set()
            assert allow_return.wait(5)
            return {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [],
            }

        def steer(text: str) -> bool:
            steer_entered.set()
            assert allow_steer.wait(5)
            pending.append(text)
            return True

        def drain():
            if not pending:
                return None
            text = "\n".join(pending)
            pending.clear()
            return text

        child.run_conversation.side_effect = run_conversation
        child.steer.side_effect = steer
        child._drain_pending_steer.side_effect = drain
        parent = MagicMock()

        result_box: dict = {}
        runner = threading.Thread(
            target=lambda: result_box.setdefault(
                "result",
                _run_single_child(0, "race", child=child, parent_agent=parent),
            )
        )
        runner.start()
        assert running.wait(5)

        accepted_box: dict = {}
        steering = threading.Thread(
            target=lambda: accepted_box.setdefault(
                "accepted", steer_subagent(child._subagent_id, "retain this exact text")
            )
        )
        steering.start()
        assert steer_entered.wait(5)
        allow_return.set()
        allow_steer.set()
        steering.join(5)
        runner.join(5)

        assert not steering.is_alive()
        assert not runner.is_alive()
        assert accepted_box["accepted"] is True
        assert result_box["result"]["missed_steer"] == "retain this exact text"

    def test_steer_after_run_return_is_rejected_before_completion_callback(self):
        """Once the child returns, a blocked completion callback cannot extend acceptance."""
        from tools.delegate_tool import _run_single_child

        callback_entered = threading.Event()
        release_callback = threading.Event()

        def progress(_event: str, **_kwargs) -> None:
            return None

        def flush() -> None:
            callback_entered.set()
            assert release_callback.wait(5)

        progress._flush = flush  # type: ignore[attr-defined]
        child = MagicMock()
        child._subagent_id = "sid-closed-before-callback"
        child._delegate_depth = 1
        child.model = "test-model"
        child.tool_progress_callback = progress
        child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

        runner = threading.Thread(
            target=lambda: _run_single_child(0, "late", child=child, parent_agent=MagicMock())
        )
        runner.start()
        assert callback_entered.wait(5)
        try:
            assert steer_subagent(child._subagent_id, "too late") is False
            child.steer.assert_not_called()
        finally:
            release_callback.set()
            runner.join(5)
        assert not runner.is_alive()


class TestSubagentSteerRPC:
    """subagent.steer gateway RPC — the programmatic caller beside subagent.interrupt."""

    class _Transport:
        def __init__(self) -> None:
            self.frames: list[dict] = []

        def write(self, obj: dict) -> bool:
            self.frames.append(obj)
            return True

        def close(self) -> None:
            return None

    def _call(self, params: dict, *, transport=None, session_record=None) -> dict:
        import tui_gateway.server as srv

        session_id = params.get("session_id")
        if session_id:
            srv._sessions[session_id] = session_record or {
                "session_key": session_id,
                "history": [],
                "transport": transport,
            }
        try:
            return srv.dispatch(
                {"id": 1, "method": "subagent.steer", "params": params},
                transport=transport,
            )
        finally:
            if session_id:
                srv._sessions.pop(session_id, None)

    def test_missing_subagent_id_is_4000(self):
        envelope = self._call({"text": "hello"})
        assert envelope["error"]["code"] == 4000

    def test_empty_text_is_4002(self):
        envelope = self._call({"subagent_id": "sid-rpc-1", "text": "   "})
        assert envelope["error"]["code"] == 4002

    def test_live_child_queues_and_receives_text(self):
        owner_transport = self._Transport()
        owner_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        agent = _StubAgent()
        _with_registered(
            "sid-rpc-2",
            agent,
            owner_session_id="owner-session",
            owner_transport=owner_transport,
            owner_session_record=owner_record,
        )
        try:
            envelope = self._call(
                {
                    "session_id": "owner-session",
                    "subagent_id": "sid-rpc-2",
                    "text": "check the edge cases",
                },
                transport=owner_transport,
                session_record=owner_record,
            )
            assert envelope["result"] == {
                "status": "queued",
                "subagent_id": "sid-rpc-2",
                "text": "check the edge cases",
            }
            assert agent.steered == ["check the edge cases"]
        finally:
            _unregister_subagent("sid-rpc-2")

    def test_run_single_child_binds_exact_runtime_owner_artifacts(self):
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.delegate_tool import _run_single_child

        observed: dict[str, bool] = {}
        owner_transport = self._Transport()
        owner_session_record = {
            "session_key": "durable-parent",
            "history": [],
            "transport": owner_transport,
        }
        child = MagicMock()
        child._subagent_id = "sid-context-owner"
        child._delegate_depth = 1
        child.model = "test-model"
        child.steer.return_value = True

        def run_conversation(**_kwargs):
            observed["owner"] = steer_subagent(
                child._subagent_id,
                "owned steer",
                owner_session_id="ui-owner",
                owner_transport=owner_transport,
                owner_session_record=owner_session_record,
            )
            observed["foreign"] = steer_subagent(
                child._subagent_id,
                "foreign steer",
                owner_session_id="ui-owner",
                owner_transport=self._Transport(),
                owner_session_record=owner_session_record,
            )
            return {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [],
            }

        child.run_conversation.side_effect = run_conversation
        tokens = set_session_vars(
            session_key="durable-parent",
            session_id="durable-parent",
            ui_session_id="ui-owner",
        )
        try:
            _run_single_child(
                0,
                "owner binding",
                child=child,
                parent_agent=MagicMock(),
                owner_transport=owner_transport,
                owner_session_record=owner_session_record,
            )
        finally:
            clear_session_vars(tokens)

        assert observed == {"owner": True, "foreign": False}
        child.steer.assert_called_once_with("owned steer")

    def test_unknown_child_is_rejected_not_an_error(self):
        envelope = self._call(
            {
                "session_id": "owner-session",
                "subagent_id": "sid-rpc-gone",
                "text": "hello",
            }
        )
        assert envelope["result"]["status"] == "rejected"

    def test_foreign_session_cannot_steer_an_owned_child(self):
        agent = _StubAgent()
        _with_registered("sid-rpc-foreign", agent, owner_session_id="owner-session")
        try:
            envelope = self._call(
                {
                    "session_id": "foreign-session",
                    "subagent_id": "sid-rpc-foreign",
                    "text": "cross-session injection",
                }
            )
            assert envelope["result"]["status"] == "rejected"
            assert agent.steered == []
        finally:
            _unregister_subagent("sid-rpc-foreign")

    def test_foreign_transport_with_correct_session_id_is_denied(self):
        owner_transport = self._Transport()
        foreign_transport = self._Transport()
        owner_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        agent = _StubAgent()
        _with_registered(
            "sid-rpc-foreign-transport",
            agent,
            owner_session_id="owner-session",
            owner_transport=owner_transport,
            owner_session_record=owner_record,
        )
        try:
            envelope = self._call(
                {
                    "session_id": "owner-session",
                    "subagent_id": "sid-rpc-foreign-transport",
                    "text": "stolen identifier",
                },
                transport=foreign_transport,
                session_record=owner_record,
            )
            assert envelope["result"]["status"] == "rejected"
            assert agent.steered == []
        finally:
            _unregister_subagent("sid-rpc-foreign-transport")

    def test_recycled_session_record_with_same_id_is_denied(self):
        owner_transport = self._Transport()
        original_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        recycled_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        agent = _StubAgent()
        _with_registered(
            "sid-rpc-recycled-session",
            agent,
            owner_session_id="owner-session",
            owner_transport=owner_transport,
            owner_session_record=original_record,
        )
        try:
            envelope = self._call(
                {
                    "session_id": "owner-session",
                    "subagent_id": "sid-rpc-recycled-session",
                    "text": "new generation",
                },
                transport=owner_transport,
                session_record=recycled_record,
            )
            assert envelope["result"]["status"] == "rejected"
            assert agent.steered == []
        finally:
            _unregister_subagent("sid-rpc-recycled-session")

    def test_server_resolves_exact_runtime_authority_from_dispatch_context(self):
        import tui_gateway.server as srv

        owner_transport = self._Transport()
        owner_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        srv._sessions["owner-session"] = owner_record

        def capture(rid, _params):
            authority = srv._current_session_steer_authority("owner-session")
            return srv._ok(
                rid,
                {
                    "transport_matches": authority[0] is owner_transport,
                    "record_matches": authority[1] is owner_record,
                },
            )

        srv._methods["test.capture-steer-authority"] = capture
        try:
            envelope = srv.dispatch(
                {
                    "id": 1,
                    "method": "test.capture-steer-authority",
                    "params": {
                        "session_id": "owner-session",
                        "owner_transport": "spoof",
                        "owner_session_record": "spoof",
                    },
                },
                transport=owner_transport,
            )
        finally:
            srv._methods.pop("test.capture-steer-authority", None)
            srv._sessions.pop("owner-session", None)

        assert envelope["result"] == {
            "transport_matches": True,
            "record_matches": True,
        }

    def test_delegate_capture_uses_dispatch_runtime_artifacts(self):
        import tui_gateway.server as srv
        from tools import delegate_tool

        owner_transport = self._Transport()
        owner_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        srv._sessions["owner-session"] = owner_record

        def capture(rid, _params):
            transport, record = delegate_tool._capture_gateway_steer_authority(
                "owner-session"
            )
            return srv._ok(
                rid,
                {
                    "transport_matches": transport is owner_transport,
                    "record_matches": record is owner_record,
                },
            )

        srv._methods["test.capture-delegate-authority"] = capture
        try:
            envelope = srv.dispatch(
                {
                    "id": 1,
                    "method": "test.capture-delegate-authority",
                    "params": {"session_id": "owner-session"},
                },
                transport=owner_transport,
            )
        finally:
            srv._methods.pop("test.capture-delegate-authority", None)
            srv._sessions.pop("owner-session", None)

        assert envelope["result"] == {
            "transport_matches": True,
            "record_matches": True,
        }

    def test_commissioning_context_rejects_recycled_runtime_session_record(self):
        import tui_gateway.server as srv
        from tools import delegate_tool
        from tui_gateway.transport import bind_transport, reset_transport

        owner_transport = self._Transport()
        original_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        recycled_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        srv._sessions["owner-session"] = recycled_record
        transport_token = bind_transport(owner_transport)
        record_token = srv._current_runtime_session_record.set(original_record)
        try:
            assert delegate_tool._capture_gateway_steer_authority("owner-session") == (
                None,
                None,
            )
        finally:
            srv._current_runtime_session_record.reset(record_token)
            reset_transport(transport_token)
            srv._sessions.pop("owner-session", None)

    def test_rpc_params_cannot_spoof_runtime_artifacts(self):
        owner_transport = self._Transport()
        owner_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        agent = _StubAgent()
        _with_registered(
            "sid-rpc-param-spoof",
            agent,
            owner_session_id="owner-session",
            owner_transport=owner_transport,
            owner_session_record=owner_record,
        )
        try:
            envelope = self._call(
                {
                    "session_id": "owner-session",
                    "subagent_id": "sid-rpc-param-spoof",
                    "text": "ignore serialized capabilities",
                    "owner_transport": self._Transport(),
                    "owner_session_record": {"session_key": "owner-session"},
                    "owner_token": "forged",
                },
                transport=owner_transport,
                session_record=owner_record,
            )
            assert envelope["result"]["status"] == "queued"
            assert agent.steered == ["ignore serialized capabilities"]
        finally:
            _unregister_subagent("sid-rpc-param-spoof")

    def test_session_transport_rebinding_does_not_transfer_ownership(self):
        original_transport = self._Transport()
        rebound_transport = self._Transport()
        owner_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": original_transport,
        }
        agent = _StubAgent()
        _with_registered(
            "sid-rpc-rebound",
            agent,
            owner_session_id="owner-session",
            owner_transport=original_transport,
            owner_session_record=owner_record,
        )
        owner_record["transport"] = rebound_transport
        try:
            for transport in (original_transport, rebound_transport):
                envelope = self._call(
                    {
                        "session_id": "owner-session",
                        "subagent_id": "sid-rpc-rebound",
                        "text": "rebound authority",
                    },
                    transport=transport,
                    session_record=owner_record,
                )
                assert envelope["result"]["status"] == "rejected"
            assert agent.steered == []
        finally:
            _unregister_subagent("sid-rpc-rebound")

    def test_concurrent_sessions_cannot_cross_steer(self):
        transports = [self._Transport(), self._Transport()]
        records = [
            {"session_key": f"session-{i}", "history": [], "transport": transports[i]}
            for i in range(2)
        ]
        agents = [_StubAgent(), _StubAgent()]
        for i in range(2):
            _with_registered(
                f"sid-concurrent-{i}",
                agents[i],
                owner_session_id=f"session-{i}",
                owner_transport=transports[i],
                owner_session_record=records[i],
            )

        barrier = threading.Barrier(2)
        results: list[str] = []

        def cross_call(caller: int) -> None:
            barrier.wait(5)
            envelope = self._call(
                {
                    "session_id": f"session-{caller}",
                    "subagent_id": f"sid-concurrent-{1 - caller}",
                    "text": f"cross-{caller}",
                },
                transport=transports[caller],
                session_record=records[caller],
            )
            results.append(envelope["result"]["status"])

        threads = [threading.Thread(target=cross_call, args=(i,)) for i in range(2)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
            assert all(not thread.is_alive() for thread in threads)
            assert sorted(results) == ["rejected", "rejected"]
            assert agents[0].steered == []
            assert agents[1].steered == []
        finally:
            for i in range(2):
                _unregister_subagent(f"sid-concurrent-{i}")

    def test_owner_still_works_after_unrelated_dispatch_queries(self):
        import tui_gateway.server as srv

        owner_transport = self._Transport()
        owner_record = {
            "session_key": "owner-session",
            "history": [],
            "transport": owner_transport,
        }
        agent = _StubAgent()
        _with_registered(
            "sid-rpc-after-query",
            agent,
            owner_session_id="owner-session",
            owner_transport=owner_transport,
            owner_session_record=owner_record,
        )
        srv._methods["test.unrelated-query"] = lambda rid, _params: srv._ok(
            rid, {"ok": True}
        )
        try:
            assert srv.dispatch(
                {"id": 8, "method": "test.unrelated-query", "params": {}},
                transport=self._Transport(),
            )["result"] == {"ok": True}
            envelope = self._call(
                {
                    "session_id": "owner-session",
                    "subagent_id": "sid-rpc-after-query",
                    "text": "still mine",
                },
                transport=owner_transport,
                session_record=owner_record,
            )
            assert envelope["result"]["status"] == "queued"
            assert agent.steered == ["still mine"]
        finally:
            srv._methods.pop("test.unrelated-query", None)
            _unregister_subagent("sid-rpc-after-query")

    def test_record_missing_runtime_artifacts_cannot_be_steered_by_rpc(self):
        agent = _StubAgent()
        owner_transport = self._Transport()
        owner_record = {
            "session_key": "claiming-session",
            "history": [],
            "transport": owner_transport,
        }
        _with_registered(
            "sid-rpc-owner-missing",
            agent,
            owner_session_id="claiming-session",
        )
        try:
            envelope = self._call(
                {
                    "session_id": "claiming-session",
                    "subagent_id": "sid-rpc-owner-missing",
                    "text": "ambiguous authority",
                },
                transport=owner_transport,
                session_record=owner_record,
            )
            assert envelope["result"]["status"] == "rejected"
            assert agent.steered == []
        finally:
            _unregister_subagent("sid-rpc-owner-missing")

    def test_rpc_without_invoking_session_identity_is_denied(self):
        agent = _StubAgent()
        _with_registered("sid-rpc-no-caller", agent, owner_session_id="owner-session")
        try:
            envelope = self._call(
                {"subagent_id": "sid-rpc-no-caller", "text": "identity missing"}
            )
            assert envelope["error"]["code"] == 4001
            assert agent.steered == []
        finally:
            _unregister_subagent("sid-rpc-no-caller")

    def test_recycled_id_does_not_preserve_the_old_sessions_authority(self):
        old_agent = _StubAgent()
        new_agent = _StubAgent()
        _with_registered("sid-rpc-recycled", old_agent, owner_session_id="old-session")
        _with_registered("sid-rpc-recycled", new_agent, owner_session_id="new-session")
        try:
            envelope = self._call(
                {
                    "session_id": "old-session",
                    "subagent_id": "sid-rpc-recycled",
                    "text": "stale generation steer",
                }
            )
            assert envelope["result"]["status"] == "rejected"
            assert old_agent.steered == []
            assert new_agent.steered == []
        finally:
            _unregister_subagent("sid-rpc-recycled")
