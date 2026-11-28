"""A top-level ``profile`` on delegate_task is refused at every entry point, never run on the default route.

``profile`` is a per-task parameter declared inside ``tasks[]`` by the active delegation backend. Accepting it at
the top level silently delegated on the default route, so the guard refuses the call before any worker, ledger
row or route is built. Stubs sit at the lowest safe boundary: the child builders, the batch runner, the ledger
writer, and the profile-home/credential resolver. No live delegation, no network, no spawn.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

import run_agent
import tools.delegate_tool  # imported first so the registry entry exists
from model_tools import handle_function_call
from tools.delegate_tool_routes import TaskRoute, resolve_task_routes
from tools.registry import registry

GOOD = "Refactor the login handler to use the new session helper"
REFUSAL_MARKERS = ("profile", "PER-TASK", "NO delegation was started", '"profile": "vega"')


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _args(profile):
    return {"tasks": [{"goal": GOOD}], "profile": profile}


def _via_agent(args, parent=None):
    return run_agent.AIAgent._dispatch_delegate_task(parent or _make_mock_parent(), dict(args))


def _via_registry(args, parent=None):
    return registry.dispatch("delegate_task", dict(args), parent_agent=parent or _make_mock_parent())


def _completed(idx=0):
    return {"task_index": idx, "status": "completed", "summary": "ok",
            "api_calls": 1, "duration_seconds": 1.0, "_child_role": None}


class _ProfileBackend:
    """Stub delegation backend that declares the per-task ``profile`` parameter."""

    name = "profile-declaring"

    def task_parameters(self, config):
        return {"profile": {"type": "string"}}

    def route_task(self, task, config):
        chosen = task.get("profile")
        return {"profile": chosen, "label": chosen} if chosen else {}


@pytest.fixture
def stub_backend(monkeypatch):
    # Lowest safe boundary for routing: no credential load, no profile-home resolution. The backend is stamped
    # because the real _resolve_route always stamps it; the real _route_to_task_route plumbing (label, cache) runs.
    monkeypatch.setattr("tools.delegate_tool_routes._active_provider", lambda config: _ProfileBackend())
    monkeypatch.setattr("tools.delegate_tool_routes._resolve_route",
                        lambda route, parent_agent, backend: TaskRoute({}, {}, backend=backend))


@pytest.fixture
def no_worker(monkeypatch):
    """Every child-building and batch-running seam, so a refused call that leaks through fails loudly."""
    mocks = {}
    for name in ("_run_batch", "_build_child_agent", "_build_child_preserving_parent_tools", "_run_single_child"):
        mocks[name] = MagicMock(name=name, return_value=_completed())
        monkeypatch.setattr(f"tools.delegate_tool.{name}", mocks[name])
    return mocks


@pytest.fixture
def no_ledger(monkeypatch):
    mocks = {}
    for name in ("_persist_dispatch", "record_unit_child"):
        mocks[name] = MagicMock(name=name)
        monkeypatch.setattr(f"tools.async_delegation.{name}", mocks[name])
    return mocks


@pytest.mark.parametrize("value", ["vega", None])
def test_top_level_profile_is_refused(value):
    payload = json.loads(_via_agent(_args(value)))

    assert "error" in payload, "a refusal, not results"
    assert "results" not in payload
    for marker in REFUSAL_MARKERS:
        assert marker in payload["error"], marker


@pytest.mark.parametrize("value", ["vega", None])
def test_refusal_on_every_entry_point(value):
    args = _args(value)

    via_agent = json.loads(_via_agent(args))
    via_registry = json.loads(_via_registry(args))
    for payload in (via_agent, via_registry):
        for marker in REFUSAL_MARKERS:
            assert marker in payload["error"], marker

    # handle_function_call cannot reach registry handlers for this tool: model_tools short-circuits every
    # delegate_task call at _AGENT_LOOP_TOOLS (model_tools.py) before dispatch, so the refusal never runs and
    # no delegation starts. The fail-closed property holds by construction; the test pins that behaviour.
    with patch("tools.delegate_tool._run_batch") as run_batch:
        via_entry = json.loads(handle_function_call("delegate_task", dict(args), task_id="t"))
    assert via_entry == {"error": "delegate_task must be handled by the agent loop"}
    run_batch.assert_not_called()


def test_no_worker_is_created_after_rejection(no_worker):
    _via_agent(_args("vega"))
    _via_registry(_args("vega"))

    for mock in no_worker.values():
        mock.assert_not_called()


def test_no_delegation_ledger_write_after_rejection(no_ledger):
    _via_agent(_args("vega"))
    _via_registry(_args("vega"))

    for mock in no_ledger.values():
        mock.assert_not_called()


def test_nested_profile_still_routes_to_that_profile(stub_backend):
    routes, error = resolve_task_routes([{"goal": GOOD, "profile": "vega"}], {}, _make_mock_parent())

    assert error is None
    assert routes[0].error is None
    assert routes[0].request == {"profile": "vega"}
    assert routes[0].label == "vega"
    assert routes[0].backend == "profile-declaring"
    assert routes[0].report() == {"backend": "profile-declaring", "label": "vega"}


def test_unpinned_task_still_runs(stub_backend):
    routes, error = resolve_task_routes([{"goal": GOOD}], {}, _make_mock_parent())

    assert error is None
    assert routes[0].error is None
    assert routes[0].request == {}
    assert routes[0].label is None
    assert "label" not in routes[0].report()


def test_tasks_with_different_profiles_resolve_independently(stub_backend):
    routes, error = resolve_task_routes(
        [{"goal": GOOD, "profile": "vega"}, {"goal": GOOD, "profile": "coder"}], {}, _make_mock_parent())

    assert error is None
    assert [r.request for r in routes] == [{"profile": "vega"}, {"profile": "coder"}]
    assert [r.label for r in routes] == ["vega", "coder"]
    assert [r.report() for r in routes] == [
        {"backend": "profile-declaring", "label": "vega"},
        {"backend": "profile-declaring", "label": "coder"},
    ]


def test_other_unknown_top_level_key_still_warn_and_runs():
    # Only the single-child runner is stubbed; the real batch path runs. depth=1 keeps it synchronous, and
    # max_spawn_depth is raised to 2 so the depth-1 parent's child is within budget (the default of 1 refuses it).
    with patch("tools.delegate_tool._get_max_spawn_depth", return_value=2), \
            patch("tools.delegate_tool._run_single_child", return_value=_completed()) as run_child:
        result = json.loads(_via_registry({"tasks": [{"goal": GOOD}], "escape": "x"}, _make_mock_parent(depth=1)))

    assert "results" in result
    assert "escape" in result["_warning"]
    run_child.assert_called_once()


def test_other_tools_keep_warn_and_run(monkeypatch):
    received = {}
    monkeypatch.setattr("tools.process_registry_handoff.handoff_process",
                        lambda session_id, args, task_id: received.update(args) or {"status": "ok"})
    call = {"action": "handoff", "session_id": "proc_1", "sesion_id": "proc_2"}

    result = json.loads(handle_function_call("process_manage", dict(call), task_id="t"))

    assert received.get("sesion_id") == "proc_2"
    assert "sesion_id" in result["_warning"]
