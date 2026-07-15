"""Real-path exact-plan authority propagation through delegated workers."""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import pytest

from gateway import session_context
from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    set_session_vars,
)
from tools import approval
from tools import delegate_tool


SESSION_KEY = "delegation-authority-session"
PARENT_SESSION_ID = "conversation-parent"
CAPABILITY_EPOCH_SHA256 = "7" * 64
OWNER_ID = "owner-1"
PLAN_ID = "plan-delegated-exact-command"
EXACT_COMMAND = "rm -rf /tmp/delegated-approved-target"
MISMATCH_COMMAND = "rm -rf /tmp/delegated-other-target"


@pytest.fixture(autouse=True)
def _isolated_local_capability_runtime(monkeypatch):
    monkeypatch.setattr(approval, "_writer_boundary_policy_required", lambda: False)
    monkeypatch.setattr(approval, "_canonical_brain_required", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"approvals": {"plan_owner_user_ids": [OWNER_ID]}},
    )
    delegation_config = {
        "provider": "",
        "model": "",
        "base_url": "",
        "api_key": "",
        "api_mode": "",
        "reasoning_effort": "",
        "max_iterations": 90,
        "child_timeout_seconds": 0,
        "max_concurrent_children": 4,
        "max_spawn_depth": 2,
        "orchestrator_enabled": True,
        "subagent_auto_approve": False,
    }
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: delegation_config)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: None)
    approval.clear_session_local(SESSION_KEY)
    yield
    approval.clear_session_local(SESSION_KEY)


def _parent_agent():
    parent = MagicMock()
    parent.base_url = "https://chatgpt.com/backend-api/codex"
    parent.api_key = "not-a-real-secret"
    parent.provider = "openai-codex"
    parent.api_mode = "codex_responses"
    parent.model = "gpt-5.6-sol"
    parent.platform = "discord"
    parent.enabled_toolsets = ["terminal", "file", "todo", "delegation"]
    parent.valid_tool_names = {"terminal", "read_file", "todo", "delegate_task"}
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._current_task_id = None
    parent._current_turn_id = "parent-turn"
    parent._interrupt_requested = False
    parent._memory_manager = None
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.context_compressor = None
    parent.session_prompt_tokens = 0
    parent.session_estimated_cost_usd = 0.0
    parent.session_cost_source = "none"
    parent.session_cost_status = "unknown"
    parent.session_id = PARENT_SESSION_ID
    return parent


class _CapabilityProbeChild:
    def __init__(self, task_index: int):
        self.task_index = task_index
        self.tool_progress_callback = None
        self._credential_pool = None
        self._subagent_id = f"probe-child-{task_index}"
        self._parent_subagent_id = None
        self._delegate_depth = 1
        self._delegate_role = "leaf"
        self._delegate_saved_tool_names = []
        self.model = "gpt-5.6-sol"
        self.session_id = f"child-session-{task_index}"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.observed: dict[str, object] = {}
        self.done = threading.Event()

    def run_conversation(self, *, user_message, task_id, stream_callback):
        self.observed = {
            "session_key": approval.get_current_session_key(default=""),
            "session_id": get_session_env("HERMES_SESSION_ID", ""),
            "capability_epoch_sha256": get_session_env(
                "HERMES_CAPABILITY_EPOCH_SHA256", ""
            ),
            "consume_only": approval.is_delegated_exact_plan_consumer(),
            "mismatch_plan_id": approval.consume_plan_capability(
                SESSION_KEY, MISMATCH_COMMAND
            ),
            "parent_yolo_visible": approval.is_session_yolo_enabled(SESSION_KEY),
            "child_yolo_granted": approval.enable_session_yolo(SESSION_KEY),
            "child_session_approval_granted": approval.approve_session(
                SESSION_KEY, "recursive delete"
            ),
        }
        try:
            approval.grant_plan_capability(
                session_key=SESSION_KEY,
                plan_id="child-minted-plan",
                exact_commands=[MISMATCH_COMMAND],
                approved_by_user_id=OWNER_ID,
                ttl_seconds=3600,
                max_uses_per_command=1,
            )
        except Exception as exc:  # the exact fail-closed result is asserted below
            self.observed["grant_error"] = str(exc)
        else:  # pragma: no cover - explicit security regression sentinel
            self.observed["grant_error"] = ""
        exact_guard = approval.check_all_command_guards(
            EXACT_COMMAND,
            "local",
        )
        self.observed["exact_guard_approved"] = exact_guard.get("approved") is True
        self.observed["exact_plan_id"] = exact_guard.get("plan_capability")
        try:
            return {
                "final_response": "capability probe complete",
                "completed": True,
                "interrupted": False,
                "api_calls": 1,
                "messages": [],
            }
        finally:
            self.done.set()

    def close(self):
        return None

    def interrupt(self, *_args, **_kwargs):
        return None


def _run_delegation(
    monkeypatch,
    *,
    task_count: int,
    background: bool = False,
) -> list[_CapabilityProbeChild]:
    children: list[_CapabilityProbeChild] = []

    def _build_child(*, task_index, **_kwargs):
        child = _CapabilityProbeChild(task_index)
        children.append(child)
        # Simulate the real AIAgent constructor assigning its own session id.
        # The commissioning snapshot must still carry the parent's id into the
        # child execution/privileged-receipt context.
        session_context._SESSION_ID.set(child.session_id)
        return child

    monkeypatch.setattr(delegate_tool, "_build_child_agent", _build_child)
    parent = _parent_agent()
    if task_count == 1:
        result = delegate_tool.delegate_task(
            goal="run the exact approved command",
            background=background,
            parent_agent=parent,
        )
    else:
        result = delegate_tool.delegate_task(
            tasks=[
                {"goal": f"run approved command {index}"}
                for index in range(task_count)
            ],
            background=background,
            parent_agent=parent,
        )
    payload = json.loads(result)
    if background:
        assert payload["status"] == "dispatched"
        assert payload["count"] == task_count
        assert all(child.done.wait(5) for child in children)
    else:
        assert [entry["status"] for entry in payload["results"]] == [
            "completed"
        ] * task_count
    return children


def _bind_parent_turn():
    session_tokens = set_session_vars(
        source="gateway",
        session_key=SESSION_KEY,
        session_id=PARENT_SESSION_ID,
        capability_epoch_sha256=CAPABILITY_EPOCH_SHA256,
    )
    approval_token = approval.set_current_session_key(SESSION_KEY)
    return session_tokens, approval_token


def _clear_parent_turn(session_tokens, approval_token) -> None:
    approval.reset_current_session_key(approval_token)
    clear_session_vars(session_tokens)


def test_single_child_consumes_exact_parent_capability_without_minting(
    monkeypatch,
) -> None:
    session_tokens, approval_token = _bind_parent_turn()
    try:
        approval.grant_plan_capability(
            session_key=SESSION_KEY,
            plan_id=PLAN_ID,
            exact_commands=[EXACT_COMMAND],
            approved_by_user_id=OWNER_ID,
            ttl_seconds=3600,
            max_uses_per_command=1,
        )
        assert approval.enable_session_yolo(SESSION_KEY) is True

        (child,) = _run_delegation(monkeypatch, task_count=1)

        assert child.observed == {
            "session_key": SESSION_KEY,
            "session_id": PARENT_SESSION_ID,
            "capability_epoch_sha256": CAPABILITY_EPOCH_SHA256,
            "consume_only": True,
            "mismatch_plan_id": None,
            "parent_yolo_visible": False,
            "child_yolo_granted": False,
            "child_session_approval_granted": False,
            "grant_error": "delegated execution cannot grant or broaden plan authority",
            "exact_guard_approved": True,
            "exact_plan_id": PLAN_ID,
        }
        assert approval.consume_plan_capability(SESSION_KEY, EXACT_COMMAND) is None
    finally:
        _clear_parent_turn(session_tokens, approval_token)


def test_parallel_children_share_exact_use_counter_without_context_loss(
    monkeypatch,
) -> None:
    session_tokens, approval_token = _bind_parent_turn()
    try:
        approval.grant_plan_capability(
            session_key=SESSION_KEY,
            plan_id=PLAN_ID,
            exact_commands=[EXACT_COMMAND],
            approved_by_user_id=OWNER_ID,
            ttl_seconds=3600,
            max_uses_per_command=2,
        )

        children = _run_delegation(monkeypatch, task_count=2)

        assert [child.observed["session_key"] for child in children] == [
            SESSION_KEY,
            SESSION_KEY,
        ]
        assert [child.observed["capability_epoch_sha256"] for child in children] == [
            CAPABILITY_EPOCH_SHA256,
            CAPABILITY_EPOCH_SHA256,
        ]
        assert [child.observed["exact_plan_id"] for child in children] == [
            PLAN_ID,
            PLAN_ID,
        ]
        assert approval.consume_plan_capability(SESSION_KEY, EXACT_COMMAND) is None
    finally:
        _clear_parent_turn(session_tokens, approval_token)


def test_background_child_keeps_commissioning_authority_across_async_executor(
    monkeypatch,
) -> None:
    session_tokens, approval_token = _bind_parent_turn()
    try:
        approval.grant_plan_capability(
            session_key=SESSION_KEY,
            plan_id=PLAN_ID,
            exact_commands=[EXACT_COMMAND],
            approved_by_user_id=OWNER_ID,
            ttl_seconds=3600,
            max_uses_per_command=1,
        )

        (child,) = _run_delegation(
            monkeypatch,
            task_count=1,
            background=True,
        )

        assert child.observed["session_key"] == SESSION_KEY
        assert child.observed["session_id"] == PARENT_SESSION_ID
        assert child.observed["capability_epoch_sha256"] == CAPABILITY_EPOCH_SHA256
        assert child.observed["consume_only"] is True
        assert child.observed["exact_plan_id"] == PLAN_ID
        assert approval.consume_plan_capability(SESSION_KEY, EXACT_COMMAND) is None
    finally:
        _clear_parent_turn(session_tokens, approval_token)


def test_expired_parent_capability_stays_denied_inside_child(monkeypatch) -> None:
    session_tokens, approval_token = _bind_parent_turn()
    try:
        approval.grant_plan_capability(
            session_key=SESSION_KEY,
            plan_id=PLAN_ID,
            exact_commands=[EXACT_COMMAND],
            approved_by_user_id=OWNER_ID,
            ttl_seconds=3600,
            max_uses_per_command=1,
        )
        approval._plan_capabilities[SESSION_KEY][PLAN_ID]["expires_at"] = 0

        (child,) = _run_delegation(monkeypatch, task_count=1)

        assert child.observed["exact_plan_id"] is None
        assert child.observed["mismatch_plan_id"] is None
    finally:
        _clear_parent_turn(session_tokens, approval_token)
