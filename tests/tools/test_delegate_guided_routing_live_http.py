"""Real host-owned child construction + inference test for the delegation guided-model-routing
adapter (plans/2026-09-15_141016-guided-model-routing.md §6 "Delegation").

Drives ``tools.delegate_tool.delegate_task`` end-to-end against a REAL ``AIAgent`` child
(`_build_child_agent` is not mocked) making a REAL HTTP call to a local loopback fake OpenAI-
compatible server -- not the selector called directly, not a mocked constructor. Complements:

  * ``tests/agent/test_managed_route_guard_live_http.py`` (Kanban-shaped agent, per-request guard).
  * ``tests/hermes_cli/test_kanban_worker_cli_route_enforcement_integration.py`` (real subprocess).

This file proves the DELEGATION adapter specifically:
  1. a task with ``routing_role`` set resolves a receipted route and the constructed child's
     actual request reaches the receipted (approved) fake endpoint with real content;
  2. a denied/mismatched managed route (no active policy) never reaches ANY endpoint -- the
     spawn fails before content is sent, zero requests recorded;
  3. a managed parent's nested delegation cannot escape to an unmanaged route or widen its role
     -- rejected before any child is constructed.

External paid inference is never used; the endpoint is an ephemeral 127.0.0.1 http.server.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _CapturingHandler(BaseHTTPRequestHandler):
    requests: list
    refused_models = ()
    reported_model = None

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode()) if length else {}
        if not self.path.rstrip("/").endswith("chat/completions"):
            self._send_json({"ok": True})
            return
        type(self).requests.append(req)
        if req.get("model") in self.refused_models:
            self._send_json({"error": {"message": "Insufficient credits", "type": "insufficient_quota"}}, status=402)
            return
        if req.get("stream"):
            self._send_stream()
        else:
            self._send_json({
                "id": "m",
                "model": self.reported_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "child done"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            })

    def _send_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "child done"},
                                      "finish_reason": None}]},
            {"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        ]
        for chunk in chunks:
            chunk["model"] = self.reported_model
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _send_json(self, payload: dict, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


def _start_server():
    handler_cls = type("Handler", (_CapturingHandler,), {"requests": []})
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, handler_cls


def _policy(endpoint: str) -> dict:
    return {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": "fake-route", "route_revision": 1, "provider": "custom",
            "model": "test-model", "endpoint": endpoint, "maker": "test",
            "model_family": "test-model", "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {"builder": {"deep": ["fake-route"], "shallow": ["fake-route"]}},
    }


@pytest.fixture()
def routed_home():
    server, handler = _start_server()
    test_home = tempfile.mkdtemp(prefix="hermes_delegation_routing_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    prev = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = hermes_home

    saved = dict(sys.modules)
    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith(("agent.", "tools.", "hermes_")):
            del sys.modules[mod]

    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/v1"
    try:
        yield {"hermes_home": hermes_home, "handler": handler, "url": url}
    finally:
        server.shutdown()
        os.environ.pop("HERMES_HOME", None)
        if prev is not None:
            os.environ["HERMES_HOME"] = prev
        shutil.rmtree(test_home, ignore_errors=True)
        sys.modules.clear()
        sys.modules.update(saved)


def _publish_active(hermes_home, url):
    from agent.model_selection_store import activate_policy, publish_policy
    record = publish_policy(hermes_home, _policy(url), approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])


def _make_parent(hermes_home, endpoint="http://127.0.0.1:1/v1"):
    from run_agent import AIAgent
    parent = AIAgent(
        api_key="parent-key", base_url=endpoint, provider="custom", model="parent-model",
        max_iterations=5, enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False, platform="cli",
    )
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _patch_custom_provider(monkeypatch, url):
    """Make the 'custom' provider name resolve through the runtime provider system to our fake
    endpoint with a harmless api key, mirroring how a real configured custom provider resolves."""
    from hermes_cli import runtime_provider as rp

    def _fake_resolve(*, requested, target_model):
        return {"provider": "custom", "base_url": url, "api_key": "fake-managed-key",
                "api_mode": "chat_completions", "request_overrides": {}, "command": None, "args": []}

    monkeypatch.setattr(rp, "resolve_runtime_provider", _fake_resolve)
    import tools.delegate_tool_config as dtc
    monkeypatch.setattr(dtc, "resolve_runtime_provider", _fake_resolve, raising=False)


def test_managed_delegation_task_reaches_real_child_and_pins_receipt(routed_home, monkeypatch):
    from tools.delegate_tool import delegate_task

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url)
    _patch_custom_provider(monkeypatch, url)
    parent = _make_parent(hermes_home)

    result_json = delegate_task(
        tasks=[{
            "goal": "Summarize the managed routing design in one sentence.",
            "routing_role": "builder",
            "routing_requirements": {"task_class": "established-pattern", "input_tokens": 1000, "reserve_tokens": 8192},
        }],
        parent_agent=parent,
    )
    result = json.loads(result_json)
    assert result["results"][0]["status"] == "completed", result
    assert len(handler.requests) == 1, "the real child must reach the approved endpoint exactly once"
    sent = json.dumps(handler.requests[0])
    assert "Summarize the managed routing design" in sent


@pytest.mark.parametrize("reported,status", [("test-model", "matching"), (None, "missing"), ("TEST-model", "changed")])
@pytest.mark.parametrize("entrypoint", ["tool", "lifecycle"])
def test_delegated_response_identity_stays_with_child(routed_home, monkeypatch, reported, status, entrypoint):
    from agent.model_selection_store import _connect
    from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService
    from tools.delegate_tool import delegate_task

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    handler.reported_model = reported
    parent = _make_parent(home, endpoint=url)
    task = dict(goal="private child identity probe", routing_role="builder",
                routing_requirements=dict(input_tokens=1000, reserve_tokens=8192))
    try:
        if entrypoint == "tool":
            result = json.loads(delegate_task(tasks=[task], parent_agent=parent))["results"][0]
            assert result["status"] == "completed", result
        else:
            service = SubagentLifecycleService(lambda: parent)
            handle = service.launch(SubagentLaunchRequest(**task))
            assert service.wait(handle, timeout_seconds=30).completed
            assert service.result(handle).terminal_state.value == "SUCCEEDED"
        assert len(handler.requests) == 1
        with _connect(home) as conn:
            health = [json.loads(row[0]) for row in conn.execute(
                "SELECT payload_json FROM routing_outcomes WHERE kind='routing_health'")]
        assert len(health) == 1
        assert health[0]["reported_model"] == reported
        assert health[0]["identity_status"] == status
        assert health[0]["status"] == "healthy" and health[0]["replay_safe"] is False
        assert "private child identity probe" not in json.dumps(health)
        assert parent._active_children == []
    finally:
        parent.close()


@pytest.mark.parametrize("first_mode", ["enforced", "shadow"])
def test_separate_calls_have_host_owned_receipts(routed_home, monkeypatch, first_mode):
    from agent.model_selection_store import get_receipt
    from tools.delegate_tool import delegate_task

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    parent = _make_parent(home, endpoint=url)
    task = {"goal": "Complete a separately launched task", "routing_role": "builder",
            "_delegation_id": "untrusted-caller-id",
            "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}}
    try:
        first = json.loads(delegate_task(tasks=[{**task, "routing_mode": first_mode}], parent_agent=parent))
        second = json.loads(delegate_task(tasks=[task], parent_agent=parent))
        entries = [result["results"][0] for result in (first, second)]
        assert all(entry["status"] == "completed" for entry in entries), (first, second)
        ids = [entry.get("routing_receipt_id") or entry["routing_shadow_receipt_id"] for entry in entries]
        assert ids[0] != ids[1]
        executions = [get_receipt(home, receipt)["requirements"]["execution_id"] for receipt in ids]
        assert executions[0] != executions[1]
        assert "untrusted-caller-id" not in executions
        assert task["_delegation_id"] == "untrusted-caller-id", "do not mutate caller intake"
        assert len(handler.requests) == 2
        assert parent._active_children == []
    finally:
        parent.close()


@pytest.mark.parametrize("failure", ["malformed_first", "malformed_last", "constructor", "storage"])
def test_batch_fatal_failure_releases_real_children(routed_home, monkeypatch, failure):
    import sqlite3

    import agent.managed_route_runtime as runtime
    import tools.delegate_tool as dt

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    parent = _make_parent(home)
    built, closed, clients = [], [], []
    build = dt._build_child_preserving_parent_tools

    def record_child(**kwargs):
        if failure == "constructor" and kwargs["task_index"] == 1:
            raise ValueError("fixture fatal constructor failure")
        child = build(**kwargs)
        built.append(child)
        clients.append(child.client)
        close = child.close
        def record_close():
            closed.append(child)
            return close()
        monkeypatch.setattr(child, "close", record_close)
        return child

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", record_child)
    persist = runtime.persist_receipt

    def fail_later_receipt(*args, **kwargs):
        if built:
            raise sqlite3.OperationalError("fixture late receipt storage failure")
        return persist(*args, **kwargs)

    if failure == "storage":
        monkeypatch.setattr(runtime, "persist_receipt", fail_later_receipt)
    tasks = [{"goal": "Complete eligible batch member", "routing_role": "builder",
              "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}}
             for _ in range(2)]
    if failure.startswith("malformed"):
        tasks[0 if failure == "malformed_first" else 1]["routing_requirements"]["unexpected"] = True
    try:
        if failure == "storage":
            with pytest.raises(sqlite3.OperationalError, match="fixture late receipt storage failure"):
                dt.delegate_task(tasks=tasks, parent_agent=parent)
        else:
            result = json.loads(dt.delegate_task(tasks=tasks, parent_agent=parent))
            assert "error" in result, result
        assert handler.requests == []
        assert parent._active_children == []
        assert closed == built
        assert all(client is not None and client.is_closed() for client in clients)
        if failure.startswith("malformed"):
            assert built == [], "validate every member before constructing any child"
        else:
            assert len(built) == 1, "exercise cleanup of an actually constructed child"
    finally:
        for child in built:
            if child not in closed:
                child.close()
        parent.close()


@pytest.mark.parametrize("denied_first", [True, False])
@pytest.mark.parametrize("denial", ["policy", "credentials"])
def test_policy_denial_does_not_cancel_eligible_batch_member(routed_home, monkeypatch, denied_first, denial):
    from tools.delegate_tool import delegate_task
    from agent.model_selection_store import get_receipt

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _patch_custom_provider(monkeypatch, url)
    if denial == "credentials":
        import tools.delegate_tool_config as config
        from agent.model_selection_store import activate_policy, publish_policy
        policy = _policy(url)
        policy["routes"].append({**policy["routes"][0], "route_id": "unavailable",
                                 "model": "unavailable", "allowed_roles": ["unapproved-role"]})
        policy["rankings"]["unapproved-role"] = {"deep": ["unavailable"]}
        publish_policy(home, policy, approval_ref=policy["approval_ref"])
        activate_policy(home, policy["policy_id"], 1)
        resolve = config._runtime_provider_credentials
        def missing_credentials(cfg, parent):
            if cfg.get("model") == "unavailable":
                raise ValueError("fixture credential unavailable")
            return resolve(cfg, parent)
        monkeypatch.setattr(config, "_runtime_provider_credentials", missing_credentials)
    else:
        _publish_active(home, url)
    parent = _make_parent(home)
    eligible = {"goal": "eligible content", "routing_role": "builder",
                "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}}
    denied = {"goal": "denied content", "routing_role": "unapproved-role",
              "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192}}
    tasks = [denied, eligible] if denied_first else [eligible, denied]
    try:
        result = json.loads(delegate_task(tasks=tasks, parent_agent=parent))
        assert "results" in result, result
        assert [entry["task_index"] for entry in result["results"]] == [0, 1]
        blocked, completed = (result["results"] if denied_first else reversed(result["results"]))
        assert blocked["status"] == "error"
        assert blocked["routing_reason"] == ("no_qualified_route" if denial == "policy" else "provider_unavailable")
        assert completed["status"] == "completed"
        receipt = get_receipt(home, completed["routing_receipt_id"])
        assert receipt is not None
        assert receipt["requirements"]["role"] == "builder"
        assert len(handler.requests) == 1
        assert handler.requests[0]["model"] == receipt["selected"]["model"]
        assert "eligible content" in json.dumps(handler.requests)
        assert "denied content" not in json.dumps(handler.requests)
        assert getattr(parent, "_active_children") == []
    finally:
        parent.close()


@pytest.mark.parametrize("independent", [False, True])
def test_mixed_background_batch_delivers_each_member_once(routed_home, monkeypatch, independent):
    import queue
    from gateway.session_context import set_session_vars
    from tools import async_delegation
    from tools.delegate_tool import delegate_task
    from tools.process_registry import process_registry
    import tools.delegate_tool_config as config
    from agent.model_selection_store import activate_policy, publish_policy

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    policy = _policy(url)
    policy["routes"].append({**policy["routes"][0], "route_id": "second-route",
                             "model": "second-model", "allowed_roles": ["writer"]})
    policy["rankings"]["writer"] = {"deep": ["second-route"], "shallow": ["second-route"]}
    publish_policy(home, policy, approval_ref="operator:test")
    activate_policy(home, "kanban-default", 1)
    _patch_custom_provider(monkeypatch, url)
    monkeypatch.setattr(config, "_get_independent_completions", lambda: independent)
    completions = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", completions)
    async_delegation._reset_for_tests()
    set_session_vars(platform="cli", session_id="mixed-background", async_delivery=True)
    parent = _make_parent(home)
    tasks = [
        {"goal": "eligible first", "routing_role": "builder"},
        {"goal": "denied content", "routing_role": "unapproved-role"},
        {"goal": "eligible last", "routing_role": "writer"},
    ]
    for task in tasks:
        task["routing_requirements"] = {"input_tokens": 1000, "reserve_tokens": 8192}
    tasks[0]["reasoning_effort"] = "high"
    tasks[2]["reasoning_effort"] = "low"
    try:
        result = json.loads(delegate_task(tasks=tasks, parent_agent=parent, background=True))
        assert result["status"] == "dispatched", result
        assert result["count"] == 2
        assert result["goals"] == ["eligible first", "eligible last"]
        assert [(entry["task_index"], entry["routing_reason"]) for entry in result["results"]] == [
            (1, "no_qualified_route"),
        ]
        events = [completions.get(timeout=30) for _ in range(2 if independent else 1)]
        entries = [entry for event in events for entry in event["results"]]
        assert sorted(entry["task_index"] for entry in entries) == [0, 2]
        assert all(entry["status"] == "completed" for entry in entries)
        assert len({entry["routing_receipt_id"] for entry in entries}) == 2
        assert len(handler.requests) == 2
        assert {(sent["model"], sent["reasoning_effort"]) for sent in handler.requests} == {
            ("test-model", "high"), ("second-model", "low"),
        }
        assert "denied content" not in json.dumps(handler.requests)
        assert parent._active_children == []
    finally:
        if async_delegation._executor is not None:
            async_delegation._executor.shutdown(wait=True)
        async_delegation._reset_for_tests()
        parent.close()


def test_managed_delegation_denied_route_reaches_zero_endpoints(routed_home, monkeypatch):
    """No active policy published -> RoutingBlocked -> the spawn must fail BEFORE any child is
    constructed or any content transmitted. Zero requests at the endpoint, ever."""
    from tools.delegate_tool import delegate_task

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    # Deliberately do NOT publish/activate a policy.
    _patch_custom_provider(monkeypatch, url)
    parent = _make_parent(hermes_home)

    result_json = delegate_task(
        tasks=[{"goal": "This must never run.", "routing_role": "builder"}],
        parent_agent=parent,
    )
    assert "error" in result_json.lower() or "routing" in result_json.lower()
    assert len(handler.requests) == 0, "a denied managed route must never reach the endpoint"


@pytest.mark.parametrize("mode", [None, "shadow", "enforced"])
def test_global_request_overrides_do_not_cross_enforced_boundary(routed_home, monkeypatch, mode):
    from pathlib import Path
    from tools.delegate_tool import delegate_task
    from hermes_cli import runtime_provider

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    resolve = runtime_provider.resolve_runtime_provider

    def with_selected_settings(**kwargs):
        runtime = resolve(**kwargs)
        runtime["request_overrides"] = {"extra_body": {"selected_setting": "selected"}}
        return runtime

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", with_selected_settings)
    (Path(home) / "config.yaml").write_text(
        "delegation:\n  request_overrides:\n    extra_body:\n      global_setting: global\n"
    )
    parent = _make_parent(home, endpoint=url)
    task = {"goal": "override boundary probe"}
    if mode:
        task.update(routing_role="builder", routing_mode=mode,
                    routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192})
    try:
        result = json.loads(delegate_task(tasks=[task], parent_agent=parent))
        assert result["results"][0]["status"] == "completed", result
        assert len(handler.requests) == 1
        sent = handler.requests[0]
        if mode == "enforced":
            assert sent["selected_setting"] == "selected"
            assert "global_setting" not in sent
        else:
            assert sent["global_setting"] == "global"
            assert "selected_setting" not in sent
    finally:
        parent.close()


@pytest.mark.parametrize("fault", [None, "provenance", "store", "outcome"])
def test_shadow_delegation_records_decision_but_keeps_legacy_child_route(routed_home, monkeypatch, fault):
    """A shadow task is observational only: selection may recommend the
    policy route, but construction and inference stay on the legacy parent
    route and no managed receipt is stamped on the child."""
    from tools.delegate_tool import delegate_task

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(hermes_home, url)
    parent = _make_parent(hermes_home, endpoint=url)

    if fault in ("store", "outcome"):
        import sqlite3
        import agent.model_selection_store as store
        def fail(*args, **kwargs):
            raise sqlite3.OperationalError("private storage failure")
        if fault == "store":
            from pathlib import Path
            database = Path(hermes_home) / "model_routing.db"
            database.rename(database.with_suffix(".saved"))
            database.mkdir()
        else:
            monkeypatch.setattr(store, "append_outcome", fail)

    result = json.loads(delegate_task(
        tasks=[{
            "goal": "Run through the legacy route while recording the recommendation.",
            "routing_role": "reviewquality" if fault == "provenance" else "builder", "routing_mode": "shadow",
            "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192},
        }],
        parent_agent=parent,
    ))

    assert result["results"][0]["status"] == "completed", result
    assert handler.requests[0]["model"] == "parent-model"
    assert result["results"][0].get("routing_receipt_id") is None
    if fault:
        assert result["results"][0]["routing_shadow_error"]
        assert "private storage failure" not in str(result)
    else:
        assert result["results"][0]["routing_shadow_receipt_id"].startswith("rr_")


def test_shadow_lifecycle_constructor_retains_legacy_fallback(routed_home, monkeypatch):
    from agent.subagent_lifecycle import SubagentLaunchRequest
    from tools.delegate_tool_routing import build_lifecycle_child
    import tools.delegate_tool as dt

    home, url = routed_home["hermes_home"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    parent = _make_parent(home, endpoint=url)
    original = dt._resolve_child_runtime
    def with_fallback(*args, **kwargs):
        runtime = original(*args, **kwargs)
        runtime["fallback_model"] = [{"provider": "custom", "model": "fallback", "base_url": url}]
        return runtime
    monkeypatch.setattr(dt, "_resolve_child_runtime", with_fallback)
    children = []
    try:
        for mode in (None, "shadow", "enforced"):
            request = SubagentLaunchRequest(goal="construct", routing_role="builder" if mode else None,
                routing_mode=mode, routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192} if mode else None)
            children.append(build_lifecycle_child(request, parent))
        assert children[0]._fallback_chain
        assert children[1]._fallback_chain == children[0]._fallback_chain
        assert children[2]._fallback_chain == []
    finally:
        for child in children:
            child.close()
        parent.close()


@pytest.mark.parametrize("entrypoint", ["tool", "lifecycle"])
@pytest.mark.parametrize("mode", [None, "shadow", "shadow_error", "enforced"])
def test_shadow_preserves_actual_failure_recovery(routed_home, monkeypatch, entrypoint, mode):
    from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService
    import tools.delegate_tool as dt

    home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    _publish_active(home, url)
    _patch_custom_provider(monkeypatch, url)
    handler.refused_models = ("parent-model", "test-model")
    parent = _make_parent(home, endpoint=url)
    original = dt._resolve_child_runtime

    def with_fallback(*args, **kwargs):
        runtime = original(*args, **kwargs)
        runtime["fallback_model"] = [{"provider": "custom", "model": "fallback-model",
                                      "base_url": url, "api_key": "fixture-key"}]
        return runtime

    monkeypatch.setattr(dt, "_resolve_child_runtime", with_fallback)
    task = {"goal": "recover after provider refusal"}
    if mode:
        task.update(routing_role="reviewquality" if mode == "shadow_error" else "builder",
                    routing_mode="shadow" if mode == "shadow_error" else mode,
                    routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192})
    try:
        if entrypoint == "tool":
            result = json.loads(dt.delegate_task(tasks=[task], parent_agent=parent))["results"][0]
            completed = result["status"] == "completed"
            summary = result.get("summary", "")
            if mode == "shadow_error":
                assert result["routing_shadow_error"]
        else:
            service = SubagentLifecycleService(lambda: parent)
            handle = service.launch(SubagentLaunchRequest(**task))
            assert service.wait(handle, timeout_seconds=30).completed
            result = service.result(handle)
            completed = result.terminal_state.value == "SUCCEEDED"
            summary = result.summary or ""
        assert handler.requests
        if mode == "enforced":
            assert not completed
            assert all(sent["model"] == "test-model" for sent in handler.requests)
        else:
            assert completed, result
            assert "child done" in summary
            assert handler.requests[0]["model"] == "parent-model"
            assert handler.requests[-1]["model"] == "fallback-model"
        assert parent._active_children == []
    finally:
        parent.close()


def test_nested_managed_delegation_cannot_widen_role(routed_home, monkeypatch):
    """A parent that is ITSELF a managed child (carries its own receipt) cannot ask a nested
    delegation for a wider/different role than its own authorized ceiling."""
    from agent.model_selection import select
    from agent.model_selection_store import activate_policy, persist_receipt, publish_policy
    from tools.delegate_tool import delegate_task

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    policy = _policy(url)
    policy["routes"][0]["allowed_roles"] = ["builder", "reviewquality"]
    policy["rankings"]["reviewquality"] = {"deep": ["fake-route"], "shallow": ["fake-route"]}
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])
    _patch_custom_provider(monkeypatch, url)

    requirements = {
        "schema_version": 1, "role": "builder", "execution_kind": "delegation",
        "execution_id": "parent-task", "attempt_id": "0", "slot_id": "",
        "task_class": "established-pattern", "required_capabilities": [],
        "input_tokens": 1000, "reserve_tokens": 8192, "reasoning": "medium",
        "provenance": {"frozen_sha": "x", "verified_by": "t", "complete": True, "contributors": []},
    }
    decision = select(requirements, policy, {}, now=1000)
    receipt_id = persist_receipt(hermes_home, decision)

    parent = _make_parent(hermes_home)
    parent._managed_routing_receipt_id = receipt_id
    parent._managed_routing_home = hermes_home

    result_json = delegate_task(
        tasks=[{"goal": "Try to widen my own mandate to a review role.", "routing_role": "reviewquality"}],
        parent_agent=parent,
    )
    assert len(handler.requests) == 0, "a widened nested mandate must never reach a real child/endpoint"
    assert "widen" in result_json.lower() or "error" in result_json.lower()


def test_nested_managed_delegation_cannot_omit_role_to_demote(routed_home, monkeypatch):
    """The actual child constructor inherits a managed parent's receipted
    ceiling even when the nested task omits every routing field."""
    from agent.model_selection import select
    from agent.model_selection_store import activate_policy, persist_receipt, publish_policy
    from tools.delegate_tool import delegate_task

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    policy = _policy(url)
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])
    _patch_custom_provider(monkeypatch, url)
    requirements = {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": "managed-parent", "attempt_id": "1", "slot_id": "",
        "task_class": "established-pattern", "required_capabilities": [],
        "input_tokens": 1000, "reserve_tokens": 8192, "reasoning": "medium",
        "provenance": {"frozen_sha": "x", "verified_by": "host", "complete": True, "contributors": []},
    }
    parent = _make_parent(hermes_home, endpoint=url)
    parent.model = "test-model"
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect_closing

    kb.init_db()
    with connect_closing() as conn:
        tid = kb.create_task(conn, title="managed-parent", assignee="test", routing_role="builder")
        claimed = kb.claim_task(conn, tid, claimer="test")
        assert claimed is not None and claimed.current_run_id is not None
        requirements.update(execution_id=tid, attempt_id=str(claimed.current_run_id))
        receipt_id = persist_receipt(hermes_home, select(requirements, policy, {}, now=1000))
        assert kb.set_routing_receipt(conn, tid, receipt_id, expected_run_id=claimed.current_run_id)
    monkeypatch.setenv("HERMES_KANBAN_ROUTING_RECEIPT", receipt_id)
    monkeypatch.setenv("HERMES_KANBAN_ROUTING_ORIGIN_HOME", hermes_home)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    from hermes_cli import kanban_worker_routing as cli_module

    cli = type("ManagedWorkerCLI", (), {"agent": parent, "reasoning_config": "medium"})()
    monkeypatch.setattr(
        cli_module, "requested_effort_for_kanban_guard", lambda _cli: "medium",
    )
    assert cli_module._enforce_kanban_routing_receipt(cli) is True

    result = json.loads(delegate_task(
        tasks=[{"goal": "Nested task deliberately omits routing_role."}],
        parent_agent=parent,
    ))

    assert "error" in result
    assert "input" in result["error"].lower() or "routing" in result["error"].lower()
    assert handler.requests == [], "omission must fail closed, never launch an unmanaged child"


def test_real_managed_moa_aggregator_authority_blocks_unmanaged_nested_child(
    routed_home, monkeypatch,
):
    """The real native aggregator call stamps turn-scoped authority on its
    acting AIAgent; a following delegation with no routing fields cannot escape
    through the legacy child constructor."""
    from agent.moa_loop import MoAChatCompletions
    from agent.model_selection_store import activate_policy, publish_policy
    from tools.delegate_tool import delegate_task

    hermes_home, handler, url = routed_home["hermes_home"], routed_home["handler"], routed_home["url"]
    policy = _policy(url)
    policy["routes"][0]["allowed_roles"] = ["moaaggregator"]
    policy["rankings"] = {
        "moaaggregator": {"deep": ["fake-route"], "shallow": ["fake-route"]},
    }
    record = publish_policy(hermes_home, policy, approval_ref="operator:test")
    activate_policy(hermes_home, "kanban-default", record["revision"])
    _patch_custom_provider(monkeypatch, url)
    parent = _make_parent(hermes_home)
    parent._current_turn_id = "managed-moa-turn"

    facade = MoAChatCompletions.__new__(MoAChatCompletions)
    facade._agent = parent
    facade.preset_name = "managed"
    facade._pending_trace = None
    facade._plan_aggregator_cache = lambda messages, tools, guidance, runtime: (messages, tools)
    facade._call_prepared_aggregator(
        {
            "aggregator": {
                "provider": "custom", "model": "legacy-ignored",
                "routing_role": "moaaggregator",
                "routing_requirements": {"input_tokens": 1000, "reserve_tokens": 8192},
            },
            "messages": [{"role": "user", "content": "Act as managed aggregator."}],
            "guidance": "", "aggregator_temperature": None,
        },
        {"stream": False},
    )
    assert len(handler.requests) == 1

    result = json.loads(delegate_task(
        tasks=[{"goal": "Try to become an unmanaged child."}], parent_agent=parent,
    ))

    assert "error" in result
    assert len(handler.requests) == 1, "the nested omission must not reach a second unmanaged request"
