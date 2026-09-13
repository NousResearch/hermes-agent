"""V1 persistent task-admission bridge and semantic-completion regressions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from hermes_cli import goals
from hermes_cli.goals import GoalContract, GoalManager, is_stale_goal_event
from tools.registry import registry
from tools.task_commit_tool import TASK_COMMIT_SCHEMA, task_commit
from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS


def call(session_id="task-commit-session", **kwargs):
    return json.loads(task_commit(session_id=session_id, **kwargs))


def create(session_id="task-commit-session", **overrides):
    payload = {
        "operation": "create",
        "objective": "Compare ten products",
        "outcome": "Ten verified products are compared and one is recommended",
        "verification": "A sourced comparison table and final recommendation exist",
        "constraints": ["Do not use unverified prices"],
        "boundaries": ["Public sources only"],
        "stop_when": ["Login or payment is required"],
    }
    payload.update(overrides)
    return call(session_id, **payload)


def test_01_schema_describes_persistent_admission_not_step_count():
    text = TASK_COMMIT_SCHEMA["description"]
    assert "persistent Goal lifecycle" in text
    assert "multiple actions" in text
    assert "three" not in text.lower()


def test_02_schema_excludes_discovery_and_short_work():
    text = TASK_COMMIT_SCHEMA["description"]
    assert "Do not use for discovery" in text
    assert "short work reliably completed in this turn" in text


def test_03_schema_preserves_goal_contract_semantics():
    text = TASK_COMMIT_SCHEMA["description"]
    assert "outcome + verification define DONE" in text
    assert "stop_when defines conditions that require BLOCKED" in text


def test_04_tool_is_registered_and_inline_bound():
    assert registry.get_entry("task_commit") is not None
    assert "task_commit" in INLINE_TOOL_EXECUTORS


def test_04b_subagents_cannot_admit_parent_goals():
    assert "task_commit" in DELEGATE_BLOCKED_TOOLS


def test_05_create_persists_complete_contract():
    result = create()
    assert result["success"] and result["result"] == "created"
    state = GoalManager("task-commit-session").state
    assert state and state.goal == "Compare ten products"
    assert state.contract.outcome.startswith("Ten verified")
    assert "Do not use unverified prices" in state.contract.constraints


def test_06_same_create_is_idempotent():
    first = create()
    second = create()
    assert second["success"] and second["result"] == "idempotent_noop"
    assert second["goal"]["created_at"] == first["goal"]["created_at"]


def test_07_different_create_returns_conflict_without_overwrite():
    create()
    result = create(objective="A different task")
    assert not result["success"] and result["result"] == "conflict"
    assert GoalManager("task-commit-session").state.goal == "Compare ten products"


def test_08_amend_appends_constraints_without_losing_other_fields():
    create()
    result = call("task-commit-session", operation="amend", constraints=["Exclude prices over 2000"])
    assert result["success"] and result["result"] == "amended"
    contract = GoalManager("task-commit-session").state.contract
    assert "Do not use unverified prices" in contract.constraints
    assert "Exclude prices over 2000" in contract.constraints
    assert "Public sources only" in contract.boundaries
    assert contract.verification.startswith("A sourced comparison")


def test_09_amend_cannot_change_objective():
    create()
    result = call("task-commit-session", operation="amend", objective="Research another market")
    assert not result["success"]
    assert "use replace" in result["error"]


def test_10_amend_is_visible_to_same_turn_fresh_judge():
    create()
    call("task-commit-session", operation="amend", constraints=["Exclude prices over 2000"])
    seen = {}

    def fake_judge(_goal, _response, **kwargs):
        seen["contract"] = kwargs["contract"]
        return "continue", "more work", False, None, False

    fresh = GoalManager("task-commit-session")
    with patch("hermes_cli.goals.judge_goal", side_effect=fake_judge):
        fresh.evaluate_after_turn("comparison still in progress")
    assert "Exclude prices over 2000" in seen["contract"].constraints


def test_11_replace_resets_execution_state_and_bumps_cutoff():
    create()
    old = GoalManager("task-commit-session")
    old.state.turns_used = 4
    old.state.subgoals = ["old criterion"]
    old.wait_for_seconds(30, "old wait")
    old_created = old.state.created_at
    result = call(
        "task-commit-session", operation="replace", objective="Research another market",
        outcome="A new market report exists", verification="The report has cited evidence",
        constraints=["Preserve source citations"], boundaries=["Public sources"],
        stop_when=["Account access is required"],
    )
    assert result["success"] and result["result"] == "replaced"
    state = GoalManager("task-commit-session").state
    assert state.created_at >= old_created
    assert state.turns_used == 0 and not state.subgoals and not state.gates
    assert state.waiting_until == 0 and state.waiting_on_session is None


def test_11b_replace_cutoff_stays_monotonic_if_wall_clock_moves_backwards():
    create()
    old_created = GoalManager("task-commit-session").state.created_at
    with patch("hermes_cli.goals.time.time", return_value=old_created - 10):
        result = call(
            "task-commit-session", operation="replace", objective="Clock-safe replacement",
            outcome="Replacement exists", verification="Replacement is verified",
        )
    assert result["success"]
    assert result["goal"]["created_at"] > old_created


def test_11c_create_after_done_keeps_cutoff_monotonic_if_wall_clock_moves_backwards():
    create()
    old = GoalManager("task-commit-session")
    old_created = old.state.created_at
    old.mark_done("first goal finished")
    with patch("hermes_cli.goals.time.time", return_value=old_created - 10):
        result = create(
            objective="Successor objective",
            outcome="Successor exists",
            verification="Successor is verified",
        )
    assert result["success"] and result["result"] == "created"
    assert result["goal"]["created_at"] > old_created


def test_12_empty_array_never_means_clear():
    create()
    result = call("task-commit-session", operation="amend", constraints=[])
    assert not result["success"] and "does not support clearing" in result["error"]
    assert "Do not use unverified prices" in GoalManager("task-commit-session").state.contract.constraints


def test_13_none_preserves_existing_contract_fields():
    create()
    before = GoalManager("task-commit-session").state.contract.to_dict()
    result = call("task-commit-session", operation="amend", constraints=None, boundaries=None, stop_when=None)
    assert result["success"] and result["result"] == "idempotent_noop"
    assert GoalManager("task-commit-session").state.contract.to_dict() == before


def test_14_stop_when_blocked_never_becomes_done():
    create()
    manager = GoalManager("task-commit-session")
    with patch("hermes_cli.goals.judge_goal", return_value=("blocked", "login required", False, None, False)):
        decision = manager.evaluate_after_turn("Cannot proceed without login")
    assert decision["verdict"] == "blocked"
    assert manager.state.status == "paused"


def test_15_missing_verification_continues():
    create()
    manager = GoalManager("task-commit-session")
    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "verification missing", False, None, False)):
        decision = manager.evaluate_after_turn("I found ten names but did not verify them")
    assert decision["should_continue"] is True
    assert manager.state.status == "active"


def test_16_budget_exhaustion_pauses_instead_of_done():
    manager = GoalManager("budget-session", default_max_turns=1)
    manager.set("Finish task", contract=GoalContract(outcome="done", verification="proof"))
    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "unfinished", False, None, False)):
        decision = manager.evaluate_after_turn("work stopped before proof")
    assert decision["status"] == "paused"
    assert manager.state.status == "paused"


def test_16b_incomplete_execution_overrides_a_done_judge_result():
    manager = GoalManager("incomplete-session", default_max_turns=3)
    manager.set("Finish task", contract=GoalContract(outcome="done", verification="proof"))
    with patch("hermes_cli.goals.judge_goal", return_value=("done", "claimed done", False, None, False)) as judge:
        decision = manager.evaluate_after_turn("TRUNCATED child summary", execution_incomplete=True)
    judge.assert_not_called()
    assert decision["verdict"] == "continue"
    assert decision["should_continue"] is True
    assert manager.state.status == "active"


def test_16c_incomplete_execution_at_goal_budget_pauses_never_done():
    manager = GoalManager("incomplete-budget", default_max_turns=1)
    manager.set("Finish task", contract=GoalContract(outcome="done", verification="proof"))
    decision = manager.evaluate_after_turn("budget summary", execution_incomplete=True)
    assert decision["status"] == "paused"
    assert manager.state.status == "paused"


def test_17_inline_executor_uses_agent_session_identity():
    agent = SimpleNamespace(session_id="inline-session")
    ctx = InlineToolContext(effective_task_id="turn")
    result = json.loads(INLINE_TOOL_EXECUTORS["task_commit"](agent, {
        "operation": "create", "objective": "Inline objective", "outcome": "Inline outcome",
        "verification": "Inline proof", "constraints": None, "boundaries": None, "stop_when": None,
    }, ctx))
    assert result["success"]
    assert GoalManager("inline-session").state.goal == "Inline objective"


def test_17b_subagent_completed_plus_truncated_is_marked_incomplete():
    from tools.process_registry_notifications import TimelineNotification

    notification = TimelineNotification.for_delegation("model text", {
        "results": [{"status": "completed", "truncated": True}],
    })
    assert notification.goal_execution_incomplete is True


def test_17c_cli_budget_exit_cannot_be_judged_done():
    import queue
    from hermes_cli.cli_loops_mixin import CLILoopsMixin

    manager = GoalManager("cli-incomplete", default_max_turns=3)
    manager.set("Finish", contract=GoalContract(outcome="done", verification="proof"))
    cli = CLILoopsMixin.__new__(CLILoopsMixin)
    cli._goal_manager = manager
    cli._get_goal_manager = lambda: manager
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli._last_turn_exit_reason = "max_iterations_reached(5/5)"
    cli._last_goal_execution_incomplete_event = False
    cli.conversation_history = [{"role": "assistant", "content": "budget summary"}]
    cli.agent = SimpleNamespace(session_id="cli-incomplete")
    with patch("hermes_cli.goals.judge_goal", return_value=("done", "wrong", False, None, False)) as judge:
        cli._maybe_continue_goal_after_turn()
    judge.assert_not_called()
    assert manager.state.status == "active"
    assert not cli._pending_input.empty()


def test_17d_tui_budget_exit_cannot_be_judged_done():
    from tui_gateway import prompt_turn

    manager = GoalManager("tui-incomplete", default_max_turns=3)
    manager.set("Finish", contract=GoalContract(outcome="done", verification="proof"))
    session = {"session_key": "tui-incomplete", "agent": SimpleNamespace(session_id="tui-incomplete")}
    result = {
        "final_response": "budget summary", "completed": False,
        "turn_exit_reason": "max_iterations_reached(5/5)",
    }
    with patch.object(prompt_turn, "_active_goal_manager", return_value=manager), \
         patch.object(prompt_turn, "_plan_goal_compression_recovery", return_value=(None, None)), \
         patch.object(prompt_turn, "_emit", create=True), \
         patch("hermes_cli.goals.judge_goal", return_value=("done", "wrong", False, None, False)) as judge:
        followup = prompt_turn._goal_followup_after_turn(
            "tab", session, result, "complete", "budget summary")
    judge.assert_not_called()
    assert manager.state.status == "active"
    assert followup and "Continuing toward your standing goal" in followup


def test_17e_gateway_forwards_incomplete_event_to_goal_hook():
    from gateway.run_goals import GatewayGoalsMixin

    class Store:
        async def get_or_create_session(self, _source, touch_activity=False):
            return SimpleNamespace(session_id="gateway-incomplete")

    runner = GatewayGoalsMixin.__new__(GatewayGoalsMixin)
    runner.async_session_store = Store()
    runner._post_turn_goal_continuation = AsyncMock()
    runner._post_turn_loop_completion = AsyncMock()
    runner._defer_wisdom_candidate_notice_after_delivery = AsyncMock()
    event = SimpleNamespace(metadata={"goal_execution_incomplete": True})
    asyncio.run(runner._run_post_turn_hooks(
        agent_result={"final_response": "child summary", "turn_exit_reason": "normal"},
        source=SimpleNamespace(), is_internal=True, event=event,
    ))
    assert runner._post_turn_goal_continuation.await_args.kwargs["execution_incomplete"] is True


def test_18_replaced_goal_classifies_old_delegation_as_stale():
    first = create()
    call(
        "task-commit-session", operation="replace", objective="New objective",
        outcome="New outcome", verification="New proof",
    )
    event = {"type": "async_delegation", "dispatched_at": first["goal"]["created_at"]}
    assert is_stale_goal_event("task-commit-session", event)


def test_19_replaced_goal_classifies_old_process_as_stale_but_not_new_process():
    first = create()
    replaced = call(
        "task-commit-session", operation="replace", objective="New objective",
        outcome="New outcome", verification="New proof",
    )
    assert is_stale_goal_event("task-commit-session", {
        "type": "completion", "started_at": first["goal"]["created_at"],
    })
    assert not is_stale_goal_event("task-commit-session", {
        "type": "completion", "started_at": replaced["goal"]["created_at"] + 1,
    })


def test_19b_paused_replacement_still_rejects_old_background_events():
    first = create()
    replaced = call(
        "task-commit-session", operation="replace", objective="Paused replacement",
        outcome="Replacement outcome", verification="Replacement proof",
    )
    manager = GoalManager("task-commit-session")
    manager.pause("waiting for user")
    assert is_stale_goal_event("task-commit-session", {
        "type": "async_delegation", "dispatched_at": first["goal"]["created_at"],
    })
    assert not is_stale_goal_event("task-commit-session", {
        "type": "async_delegation", "dispatched_at": replaced["goal"]["created_at"] + 1,
    })


def test_20_gateway_stale_completion_is_displayed_without_agent_injection(monkeypatch):
    create("gateway-goal")
    replaced = call(
        "gateway-goal", operation="replace", objective="New objective",
        outcome="New outcome", verification="New proof",
    )
    adapter = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True)))
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_db = SimpleNamespace(get_compression_tip=AsyncMock(return_value="gateway-goal"))
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = __import__("collections").OrderedDict()
    runner._completion_delivery_retention = 32
    runner._build_process_event_source = lambda _evt: SessionSource(
        platform=Platform.TELEGRAM, user_id="u", chat_id="c", chat_type="dm")
    runner._resolve_injection_adapter = lambda *_args: adapter
    runner._thread_metadata_for_source = lambda _source: None
    runner._preflight_completion_delivery = AsyncMock(return_value=SimpleNamespace(
        proceed=True, early_result=None, claim_id="", delegation_id=""))
    runner._inject_watch_notification = AsyncMock()
    event = {
        "type": "completion", "session_id": "old-process", "parent_session_id": "gateway-goal",
        "started_at": replaced["goal"]["created_at"] - 1,
    }
    assert asyncio.run(runner._deliver_completion_notification("old output", event)) is True
    adapter.send.assert_awaited_once()
    runner._inject_watch_notification.assert_not_awaited()


def test_21_tui_stale_completion_emits_status_without_claiming_agent_turn(monkeypatch):
    create("tui-goal")
    replacement = call(
        "tui-goal", operation="replace", objective="Replacement", outcome="Replacement done",
        verification="Replacement proof",
    )
    from tui_gateway import server

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_notification_event_requires_owner", lambda _evt: False)
    claim_turn = __import__("unittest.mock").mock.Mock(return_value=True)
    monkeypatch.setattr(server, "_notif_claim_turn", claim_turn)
    registry_stub = SimpleNamespace(
        is_completion_consumed=lambda _sid: False,
        completion_queue=__import__("queue").Queue(),
    )
    event = {
        "type": "completion", "session_id": "old-process", "parent_session_id": "tui-goal",
        "started_at": replacement["goal"]["created_at"] - 1,
    }
    assert server._notif_handle_event(
        "tab", {"session_key": "tui-goal"}, event, set(), registry_stub, lambda _evt: "old output", None,
    ) is True
    claim_turn.assert_not_called()
    assert any(args[0] == "status.update" for args in emitted)


# ──────────────────────────────────────────────────────────────────────
# Typed landing admission
# ──────────────────────────────────────────────────────────────────────


def test_22_landing_create_persists_typed_contract():
    result = call(
        "landing-session", operation="create",
        objective="Ship the landing page", outcome="Landing page is live",
        verification="curl probe passes",
        landing={
            "required_state": "LIVE_ACCEPTED",
            "targets": ["https://example.com/health", "release tag v2"],
            "live_probe": "curl -fsS https://example.com/health",
            "restart_required": True,
        },
    )
    assert result["success"] and result["result"] == "created"
    contract = GoalManager("landing-session").state.contract
    assert contract.landing is not None
    assert contract.landing.required_state == "LIVE_ACCEPTED"
    assert contract.landing.targets == ["https://example.com/health", "release tag v2"]
    assert contract.landing.live_probe.startswith("curl")
    assert contract.landing.restart_required is True
    assert result["goal"]["contract"]["landing"]["required_state"] == "LIVE_ACCEPTED"


def test_23_landing_rejects_shell_and_command_keys():
    for key in ("shell", "command"):
        result = call(
            "landing-bad", operation="create",
            objective="X", outcome="Y", verification="Z",
            landing={"required_state": "COMMITTED", "targets": ["a"], key: "rm -rf /"},
        )
        assert not result["success"], key
        assert "landing" in result["error"]


def test_24_landing_rejects_invalid_state():
    result = call(
        "landing-bad", operation="create",
        objective="X", outcome="Y", verification="Z",
        landing={"required_state": "MAYBE", "targets": ["a"]},
    )
    assert not result["success"]
    assert "landing" in result["error"]


def test_25_landing_rejects_empty_and_non_string_targets():
    for bad_targets in ([], [42], [None], "path"):
        result = call(
            "landing-bad", operation="create",
            objective="X", outcome="Y", verification="Z",
            landing={"required_state": "COMMITTED", "targets": bad_targets},
        )
        assert not result["success"], bad_targets
        assert "landing" in result["error"]


def test_26_landing_rejects_non_object():
    result = call(
        "landing-bad", operation="create",
        objective="X", outcome="Y", verification="Z",
        landing="COMMITTED",
    )
    assert not result["success"]
    assert "landing" in result["error"]


def test_27_landing_unknown_key_rejected():
    result = call(
        "landing-bad", operation="create",
        objective="X", outcome="Y", verification="Z",
        landing={"required_state": "COMMITTED", "targets": ["a"], "warp": True},
    )
    assert not result["success"]
    assert "warp" in result["error"]


def test_28_landing_amend_preserves_when_omitted_and_replaces_when_given():
    call(
        "landing-amend", operation="create",
        objective="Ship it", outcome="Live", verification="Probe",
        landing={"required_state": "STAGED", "targets": ["prod"]},
    )
    noop = call("landing-amend", operation="amend", constraints=["keep"])
    assert noop["success"] and noop["result"] == "amended"
    assert GoalManager("landing-amend").state.contract.landing.required_state == "STAGED"
    replaced = call(
        "landing-amend", operation="amend",
        landing={"required_state": "DEPLOYED", "targets": ["prod", "staging"]},
    )
    assert replaced["success"] and replaced["result"] == "amended"
    contract = GoalManager("landing-amend").state.contract
    assert contract.landing.required_state == "DEPLOYED"
    assert contract.landing.targets == ["prod", "staging"]
    assert "keep" in contract.constraints


def test_29_inline_executor_persists_landing():
    agent = SimpleNamespace(session_id="inline-landing-session")
    ctx = InlineToolContext(effective_task_id="turn")
    result = json.loads(INLINE_TOOL_EXECUTORS["task_commit"](agent, {
        "operation": "create", "objective": "Inline landing", "outcome": "Landed",
        "verification": "Probe passes", "constraints": None, "boundaries": None, "stop_when": None,
        "landing": {
            "required_state": "LIVE_ACCEPTED",
            "targets": ["https://example.com/health"],
            "live_probe": "curl -fsS https://example.com/health",
            "restart_required": True,
        },
    }, ctx))
    assert result["success"] and result["result"] == "created"
    landing = GoalManager("inline-landing-session").state.contract.landing
    assert landing is not None
    assert landing.required_state == "LIVE_ACCEPTED"
    assert landing.targets == ["https://example.com/health"]
    assert landing.restart_required is True
