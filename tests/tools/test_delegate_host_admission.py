"""Real delegation adapters against an isolated authority, with no inference."""

import json
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.waiting import wait_until
from tools import async_delegation, delegate_tool, host_policy
from tools.delegate_tool_routes import TaskRoute
from tools.worker_admission import admission_status, reserve_workers

pytestmark = pytest.mark.platforms("posix")


class ControlledChild:
    def __init__(self, ident):
        self.session_id = self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self._delegate_saved_tool_names = []
        self._credential_pool = self.tool_progress_callback = None
        self.provider, self.model = "test", "fixture-model"
        self.started, self.release, self.closed = (threading.Event() for _ in range(3))
        self.stopped = threading.Event()
        self.close_count = self.calls = 0
        self.run_hook = lambda: None

    def run_conversation(self, **_kwargs):
        self.calls += 1
        self.started.set()
        self.run_hook()
        assert self.release.wait(20)
        return {
            "final_response": "completed", "completed": not self.stopped.is_set(),
            "interrupted": self.stopped.is_set(),
            "api_calls": 1, "messages": [],
        }

    def hard_interrupt(self, *_args, **_kwargs):
        self.stopped.set()

    def get_activity_summary(self):
        return {"api_call_count": 1}

    def close(self):
        self.close_count += 1
        self.closed.set()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = (tmp_path / "authority").resolve()
    root.mkdir(mode=0o700)
    monkeypatch.setattr(host_policy, "policy_root", lambda: root)
    host_policy.atomic_private_json(root / "policy.json", {
        "version": 1, "revision": "integration-v1",
        "admission": {"mode": "enforce", "max_workers": 2}, "approvals": {},
    })
    host_policy.atomic_private_json(root / "ledger.json", {"version": 1, "reservations": {}})
    with host_policy.policy_lock():
        pass
    parent = SimpleNamespace(
        session_id="parent", _session_db=None, _delegate_depth=0,
        _active_children=[], _active_children_lock=threading.Lock(),
        enabled_toolsets=["file"], tools=[], quiet_mode=True,
    )
    cfg = {"max_concurrent_children": 3, "max_spawn_depth": 3}
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: cfg)
    creds = dict.fromkeys(("model", "provider", "base_url", "api_key", "api_mode"))
    route = TaskRoute(creds=creds, routing_cfg={})
    monkeypatch.setattr(delegate_tool, "_routes_for", lambda tasks, *_args: ([route] * len(tasks), None))
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda *_args: creds)
    children = []

    def build(**kwargs):
        child = ControlledChild(f"child-{uuid.uuid4().hex}")
        children.append(child)
        delegate_tool._attach_child(kwargs["parent_agent"], child)
        return child

    monkeypatch.setattr(delegate_tool, "_build_child_agent", build)
    async_delegation._reset_for_tests()
    completion = queue.Queue()
    monkeypatch.setattr("tools.process_registry.process_registry.completion_queue", completion)
    yield SimpleNamespace(root=root, parent=parent, children=children, cfg=cfg, completion=completion)
    for child in children:
        child.release.set()
    for child in children:
        if child.started.is_set():
            assert child.closed.wait(20)
    if async_delegation._executor is not None:
        async_delegation._executor.shutdown(wait=True)
    async_delegation._reset_for_tests()


@pytest.mark.parametrize("entrypoint", ["sync", "background", "registry", "lifecycle"])
def test_all_entrypoints_reject_before_construction(runtime, monkeypatch, entrypoint):
    from agent.subagent_lifecycle import SubagentLifecycleError, SubagentLifecycleService
    from tools.registry import registry

    occupied = reserve_workers(2)
    transcript = Mock()
    monkeypatch.setattr("tools.delegation_live_log.create_live_transcripts", transcript)
    try:
        if entrypoint == "lifecycle":
            with pytest.raises(SubagentLifecycleError, match="capacity"):
                SubagentLifecycleService(lambda: runtime.parent).launch({"goal": "inspect safely"})
        elif entrypoint == "registry":
            result = json.loads(registry.dispatch("delegate_task", {"goal": "inspect safely"}, parent_agent=runtime.parent))
            assert result["admission"] == {"code": "worker_capacity", "occupancy": 2, "limit": 2}
        else:
            result = json.loads(delegate_tool.delegate_task(
                goal="inspect safely", parent_agent=runtime.parent, background=entrypoint == "background",
            ))
            assert result["admission"]["code"] == "worker_capacity"
        assert runtime.children == []
        transcript.assert_not_called()
    finally:
        for reservation in occupied:
            reservation.finish()


def test_batch_refusal_is_all_or_none_before_any_child_exists(runtime):
    occupied = reserve_workers(1)[0]
    try:
        result = json.loads(delegate_tool.delegate_task(
            tasks=[{"goal": "Inspect the parser contracts"}, {"goal": "Inspect the transport contracts"}],
            parent_agent=runtime.parent,
        ))
        assert "capacity" in result["error"]
        assert runtime.children == []
        assert admission_status()["reserved_workers"] == 1
    finally:
        occupied.finish()


def test_schema_retry_retains_the_original_worker_reservation(runtime):
    child = delegate_tool._build_child_preserving_parent_tools(parent_agent=runtime.parent)
    child._delegate_output_schema = {"type": "object"}
    retry_started, retry_release = threading.Event(), threading.Event()
    child.release.set()

    def during_retry():
        if child.calls == 2:
            retry_started.set()
            assert retry_release.wait(20)

    child.run_hook = during_retry
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(delegate_tool._run_single_child, 0, "inspect", child, runtime.parent)
        try:
            assert retry_started.wait(10)
            assert admission_status()["reserved_workers"] == 1
            assert not child.closed.is_set()
        finally:
            retry_release.set()
        assert result.result(timeout=10)["schema_retries"] == 1
    assert child.calls == 2
    assert admission_status()["reserved_workers"] == 0


def test_lifecycle_cancel_does_not_free_a_running_child(runtime):
    from agent.subagent_lifecycle import SubagentLifecycleService, SubagentState

    service = SubagentLifecycleService(lambda: runtime.parent)
    handle = service.launch({"goal": "inspect", "correlation_id": "cancellation"})
    child = runtime.children[0]
    assert child.started.wait(10)
    assert service.cancel(handle, reason="test").accepted
    assert child.stopped.is_set()
    assert admission_status()["reserved_workers"] == 1
    child.release.set()
    assert service.wait(handle, timeout_seconds=10).state is SubagentState.CANCELLED
    assert admission_status()["reserved_workers"] == 0


def test_accepted_queued_background_child_survives_outer_dispatch_failure(runtime, monkeypatch):
    from tools.delegate_tool_admission import construction_batch

    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    pending = []

    def queue_without_start(**kwargs):
        pending.append(kwargs["runner"])
        return {"status": "dispatched", "delegation_id": "queued-admission-test"}

    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch", queue_without_start)
    with pytest.raises(RuntimeError, match="after accepted dispatch"):
        with construction_batch(1, runtime.parent):
            child = delegate_tool._build_child_preserving_parent_tools(parent_agent=runtime.parent)
            import time
            batch = delegate_tool._Batch(
                [{"goal": "inspect"}], [(0, {"goal": "inspect"}, child)], runtime.parent, {"model": None},
                None, "leaf", 1, None, [], [], "", "", None, None, False, time.monotonic(),
            )
            assert json.loads(delegate_tool._run_batch(batch, True))["status"] == "dispatched"
            raise RuntimeError("after accepted dispatch")
    assert admission_status()["reserved_workers"] == 1
    assert child.close_count == 0
    child.release.set()
    assert pending.pop()()["results"][0]["status"] == "completed"
    assert child.close_count == 1
    assert admission_status()["reserved_workers"] == 0


def test_sync_and_split_background_share_capacity_until_real_exit(runtime, monkeypatch):
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    runtime.cfg["independent_completions"] = True
    result = json.loads(delegate_tool.delegate_task(
        tasks=[{"goal": "Inspect the parser contracts"}, {"goal": "Inspect the transport contracts"}],
        parent_agent=runtime.parent, background=True,
    ))
    assert result["status"] == "dispatched"
    assert len(result["units"]) == 2
    assert all(child.started.wait(10) for child in runtime.children)
    assert admission_status()["reserved_workers"] == 2
    rejected = json.loads(delegate_tool.delegate_task(goal="another sync task", parent_agent=runtime.parent))
    assert "capacity" in rejected["error"]
    runtime.children[0].release.set()
    wait_until(lambda: admission_status()["reserved_workers"] == 1, timeout=10)
    assert not runtime.children[1].closed.is_set()
    runtime.children[1].release.set()
    wait_until(lambda: admission_status()["reserved_workers"] == 0, timeout=10)


@pytest.mark.parametrize("failure", [ValueError, RuntimeError, KeyboardInterrupt])
def test_mid_batch_construction_failure_releases_only_unstarted_children(runtime, monkeypatch, failure):
    build = delegate_tool._build_child_agent

    def fail_second(**kwargs):
        if runtime.children:
            raise failure("construction refused")
        return build(**kwargs)

    monkeypatch.setattr(delegate_tool, "_build_child_agent", fail_second)
    arguments = dict(
        tasks=[{"goal": "Inspect the parser contracts"}, {"goal": "Inspect the transport contracts"}],
        parent_agent=runtime.parent,
    )
    if failure is ValueError:
        assert "construction refused" in json.loads(delegate_tool.delegate_task(**arguments))["error"]
    else:
        with pytest.raises(failure):
            delegate_tool.delegate_task(**arguments)
    assert admission_status()["reserved_workers"] == 0
    assert runtime.parent._active_children == []
    assert runtime.children[0].close_count == 1


def test_policy_replacement_before_claim_cleans_unstarted_child(runtime):
    child = delegate_tool._build_child_preserving_parent_tools(parent_agent=runtime.parent)
    (runtime.root / "policy.json").write_text("{broken", encoding="utf-8")
    result = delegate_tool._run_single_child(0, "inspect", child, runtime.parent)
    assert result["admission"]["code"] == "host_policy_invalid"
    assert not child.started.is_set()
    assert child.close_count == 1
    assert runtime.parent._active_children == []
    assert json.loads((runtime.root / "ledger.json").read_text())["reservations"] == {}


def test_duplicate_run_never_closes_or_releases_running_worker(runtime):
    child = delegate_tool._build_child_preserving_parent_tools(parent_agent=runtime.parent)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(delegate_tool._run_single_child, 0, "inspect", child, runtime.parent)
        try:
            assert child.started.wait(10)
            duplicate = delegate_tool._run_single_child(0, "duplicate", child, runtime.parent)
            assert duplicate["status"] == "error"
            assert child.close_count == 0
            assert admission_status()["reserved_workers"] == 1
        finally:
            child.release.set()
        assert pending.result(timeout=10)["status"] == "completed"
    assert child.calls == child.close_count == 1
    assert admission_status()["reserved_workers"] == 0


def test_timeout_keeps_reservation_through_abandoned_conversation_and_close(runtime, monkeypatch):
    child = delegate_tool._build_child_preserving_parent_tools(parent_agent=runtime.parent)

    def abandon(run, *_args):
        assert run.child.started.wait(10)
        # End the parent wait deterministically, without waiting on a clock.

    monkeypatch.setattr("tools.delegate_tool_child_run._ChildRun.wait_liveness_aware", abandon)
    result = delegate_tool._run_single_child(0, "inspect", child, runtime.parent)
    assert result["status"] == "timeout"
    assert not child.closed.is_set()
    assert admission_status()["reserved_workers"] == 1
    child.release.set()
    wait_until(lambda: admission_status()["reserved_workers"] == 0, timeout=10)
    assert child.close_count == 1


def test_lifecycle_submission_failure_rolls_back_registry_and_reservation(runtime, monkeypatch):
    from agent import subagent_lifecycle as lifecycle

    executor = Mock()
    executor.submit.side_effect = RuntimeError("executor refused")
    monkeypatch.setattr(lifecycle, "_EXECUTOR", executor)
    before = dict(lifecycle._REGISTRY.records)
    with pytest.raises(RuntimeError, match="executor refused"):
        lifecycle.SubagentLifecycleService(lambda: runtime.parent).launch({
            "goal": "inspect", "correlation_id": "submission-failure",
        })
    assert lifecycle._REGISTRY.records == before
    assert runtime.parent._active_children == []
    assert runtime.children[0].close_count == 1
    assert admission_status()["reserved_workers"] == 0


def test_lifecycle_submit_exception_does_not_close_an_already_running_child(runtime, monkeypatch):
    from agent import subagent_lifecycle as lifecycle

    pending = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        def accepted_then_raise(function, *args):
            pending.append(pool.submit(function, *args))
            assert runtime.children[0].started.wait(10)
            raise RuntimeError("failure after accepted submission")

        executor = Mock()
        executor.submit.side_effect = accepted_then_raise
        monkeypatch.setattr(lifecycle, "_EXECUTOR", executor)
        try:
            with pytest.raises(RuntimeError, match="after accepted submission"):
                lifecycle.SubagentLifecycleService(lambda: runtime.parent).launch({"goal": "inspect"})
            assert runtime.children[0].close_count == 0
            assert admission_status()["reserved_workers"] == 1
        finally:
            for child in runtime.children:
                child.release.set()
        pending[0].result(timeout=10)
    assert runtime.children[0].close_count == 1
    assert admission_status()["reserved_workers"] == 0


def test_nested_delegation_keeps_parent_claim_while_refusing_descendant(runtime):
    child = delegate_tool._build_child_preserving_parent_tools(parent_agent=runtime.parent)
    occupied = reserve_workers(1)[0]
    nested_results = []

    def nested():
        nested_results.append(json.loads(delegate_tool.delegate_task(goal="nested task", parent_agent=child)))
        assert admission_status()["reserved_workers"] == 2
        child.release.set()

    child.run_hook = nested
    try:
        result = delegate_tool._run_single_child(0, "orchestrate", child, runtime.parent)
        assert result["status"] == "completed"
        assert "capacity" in nested_results[0]["error"]
        assert admission_status()["reserved_workers"] == 1
    finally:
        occupied.finish()
