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

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode()) if length else {}
        if not self.path.rstrip("/").endswith("chat/completions"):
            self._send_json({"ok": True})
            return
        type(self).requests.append(req)
        if req.get("stream"):
            self._send_stream()
        else:
            self._send_json({
                "id": "m",
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
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _send_json(self, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(200)
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


def _make_parent(hermes_home):
    from run_agent import AIAgent
    parent = AIAgent(
        api_key="parent-key", base_url="http://127.0.0.1:1/v1", provider="custom", model="parent-model",
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
            "routing_requirements": {"task_class": "established-pattern"},
        }],
        parent_agent=parent,
    )
    result = json.loads(result_json)
    assert result["results"][0]["status"] == "completed", result
    assert len(handler.requests) == 1, "the real child must reach the approved endpoint exactly once"
    sent = json.dumps(handler.requests[0])
    assert "Summarize the managed routing design" in sent


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
        "input_tokens": 0, "reserve_tokens": 0, "reasoning": "medium",
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
