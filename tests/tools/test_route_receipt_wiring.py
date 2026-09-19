from __future__ import annotations

import hashlib
from pathlib import Path

from agent.antigravity_delegate import AntigravityDelegateChild
from agent.antigravity_worker import AntigravityResult
from agent.gemini_route_receipts import GeminiReceiptStore
from agent.route_receipts import RouteReceiptChild, RouteReceiptStore


class FakeAntigravityWorker:
    def __init__(self, result: AntigravityResult):
        self.result = result
        self.closed = False

    def run(self, **kwargs):
        on_started = kwargs.get("on_process_started")
        if on_started:
            on_started()
        return self.result

    def close(self):
        self.closed = True


def gemini_result(*, ok: bool) -> AntigravityResult:
    return AntigravityResult(
        status="success" if ok else "failed",
        response="fake gemini answer" if ok else None,
        conversation_id="fixture-conversation",
        usage=(
            {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_tokens": 3,
                "cache_write_tokens": 4,
                "reasoning_tokens": 5,
                "total_tokens": 24,
            }
            if ok
            else {}
        ),
        raw_envelope={"status": "SUCCESS" if ok else "ERROR"},
        exit_code=0 if ok else 1,
        duration_ms=12,
        error_code=None if ok else "worker_failed",
        error_message=None if ok else "fixture failure",
    )


class FakeFrontierChild:
    def __init__(self):
        self.session_id = "frontier-child"
        self.provider = "openai-codex"
        self.model = "frontier-fixture"
        self.session_prompt_tokens = 7
        self.session_completion_tokens = 2
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 1
        self.calls = 0
        self.tool_progress_callback = None

    def run_conversation(self, **_kwargs):
        self.calls += 1
        return {
            "completed": True,
            "final_response": "fake frontier fallback",
            "api_calls": 1,
            "messages": [],
        }

    def close(self):
        return None


def make_adapter(tmp_path: Path, *, ok: bool, fallback=None):
    path = tmp_path / "routing.sqlite3"
    return AntigravityDelegateChild(
        worker=FakeAntigravityWorker(gemini_result(ok=ok)),
        fallback_child=fallback,
        store=GeminiReceiptStore(path),
        route_store=RouteReceiptStore(path),
        run_kind="synthetic",
        task_index=0,
        goal="fixture goal",
        context="fixture context",
        output_schema=None,
        output_contract="text",
        route_requested="auto",
        route_reason="fixture",
        data_classification="standard",
        requested_provider="antigravity-subscription",
        requested_model="gemini-fixture",
        requested_effort="high",
        parent_session_id="parent-fixture",
        parent_turn_id="turn-fixture",
    )


def test_gemini_adapter_writes_typed_route_neutral_receipt_with_fake_worker(tmp_path: Path):
    adapter = make_adapter(tmp_path, ok=True)

    result = adapter.run_conversation(user_message="fixture", task_id="fixture-task")

    assert result["final_response"] == "fake gemini answer"
    route_row = adapter.route_store.get_attempt(adapter.receipt_id)
    assert route_row["stable_call_id"] == adapter.receipt_id
    assert route_row["route_decision"] == "gemini"
    assert route_row["run_kind"] == "synthetic"
    assert route_row["status"] == "completed"
    assert route_row["total_tokens"] == 24
    legacy_row = adapter.store.get_attempt(adapter.receipt_id)
    assert legacy_row["goal_text"] == ""
    assert legacy_row["context_text"] == ""
    assert legacy_row["response_text"] is None
    assert legacy_row["response_sha256"] == hashlib.sha256(
        b"fake gemini answer"
    ).hexdigest()
    assert legacy_row["response_bytes"] == len(b"fake gemini answer")
    assert legacy_row["terminal_response_text"] is None
    assert legacy_row["raw_envelope_json"] is None
    assert legacy_row["error_message"] is None


def test_gemini_fallback_records_joined_frontier_receipt_with_fakes_only(tmp_path: Path):
    route_store = RouteReceiptStore(tmp_path / "routing.sqlite3")
    fallback = RouteReceiptChild(
        child=FakeFrontierChild(),
        store=route_store,
        parent_session_id="parent-fixture",
        parent_turn_id="turn-fixture",
        task_index=0,
        run_kind="synthetic",
        route_requested="auto",
        route_reason="fixture",
        data_classification="standard",
        output_contract="text",
    )
    adapter = make_adapter(tmp_path, ok=False, fallback=fallback)

    result = adapter.run_conversation(user_message="fixture", task_id="fixture-task")

    assert result["final_response"] == "fake frontier fallback"
    gemini_row = adapter.route_store.get_attempt(adapter.receipt_id)
    frontier_row = adapter.route_store.get_attempt(fallback.route_receipt_id)
    assert gemini_row["status"] == "failed"
    assert gemini_row["fallback_used"] == 1
    assert gemini_row["fallback_call_id"] == fallback.route_receipt_id
    assert frontier_row["route_decision"] == "frontier"
    assert frontier_row["fallback_from_call_id"] == adapter.receipt_id
    assert frontier_row["status"] == "completed"
    assert result["route_receipt_id"] == adapter.receipt_id
    assert result["fallback_route_receipt_id"] == fallback.route_receipt_id
    legacy_row = adapter.store.get_attempt(adapter.receipt_id)
    assert legacy_row["goal_text"] == ""
    assert legacy_row["context_text"] == ""
    assert legacy_row["response_text"] is None
    assert legacy_row["terminal_response_text"] is None
    assert legacy_row["raw_envelope_json"] is None
    assert legacy_row["error_message"] is None


def test_route_receipts_finish_and_link_when_legacy_writes_fail(
    tmp_path: Path, monkeypatch
):
    route_store = RouteReceiptStore(tmp_path / "routing.sqlite3")
    fallback = RouteReceiptChild(
        child=FakeFrontierChild(),
        store=route_store,
        parent_session_id="parent-fixture",
        parent_turn_id="turn-fixture",
        task_index=0,
        run_kind="synthetic",
        route_requested="auto",
        route_reason="fixture",
        data_classification="standard",
        output_contract="text",
    )
    adapter = make_adapter(tmp_path, ok=False, fallback=fallback)

    def legacy_write_failed(*_args, **_kwargs):
        raise RuntimeError("legacy receipt unavailable")

    monkeypatch.setattr(adapter.store, "complete_attempt", legacy_write_failed)
    monkeypatch.setattr(adapter.store, "record_fallback_outcome", legacy_write_failed)

    result = adapter.run_conversation(user_message="fixture", task_id="fixture-task")

    gemini_row = adapter.route_store.get_attempt(adapter.receipt_id)
    frontier_row = adapter.route_store.get_attempt(fallback.route_receipt_id)
    assert result["final_response"] == "fake frontier fallback"
    assert gemini_row["status"] == "failed"
    assert gemini_row["fallback_call_id"] == fallback.route_receipt_id
    assert frontier_row["fallback_from_call_id"] == adapter.receipt_id
    assert frontier_row["status"] == "completed"


def test_frontier_wrapper_builder_replaces_active_child_and_freezes_run_kind(tmp_path: Path):
    from tools.delegate_tool_gemini import wrap_frontier_delegate_child

    raw = FakeFrontierChild()
    parent = type("Parent", (), {})()
    parent.session_id = "parent-fixture"
    parent._current_turn_id = "turn-fixture"
    parent._active_children = [raw]
    parent._active_children_lock = None
    wrapped = wrap_frontier_delegate_child(
        child=raw,
        parent_agent=parent,
        task_index=0,
        task={
            "route": "sol",
            "run_kind": "canary",
            "data_classification": "standard",
            "output_contract": "text",
        },
        routing_cfg={"receipt_db": "routing/fixture.sqlite3"},
        route_reason="explicit",
        receipt_path=tmp_path / "routing.sqlite3",
    )

    assert parent._active_children == [wrapped]
    assert wrapped.run_kind == "canary"
    wrapped.run_conversation(user_message="fixture", task_id="fixture-task")
    assert wrapped.store.get_attempt(wrapped.route_receipt_id)["run_kind"] == "canary"


def test_delegate_task_contract_advertises_and_validates_run_kind():
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA
    from tools.delegate_tool_tasks import _normalize_task_list

    run_kind = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]["run_kind"]
    assert run_kind["enum"] == ["production", "canary", "synthetic", "evaluation"]
    assert DELEGATE_TASK_SCHEMA["parameters"]["properties"]["run_kind"]["enum"] == run_kind["enum"]

    normalized, error = _normalize_task_list(
        "A complete fake task",
        None,
        None,
        None,
        "leaf",
        3,
        run_kind="synthetic",
    )
    assert error is None
    assert normalized == [
        {"goal": "A complete fake task", "role": "leaf", "run_kind": "synthetic"}
    ]

    tasks, error = _normalize_task_list(
        None,
        None,
        [{"goal": "A complete fake task", "run_kind": "mystery"}],
        None,
        "leaf",
        3,
    )
    assert tasks is None
    assert error == "Task 0 run_kind must be one of: production, canary, synthetic, evaluation."
