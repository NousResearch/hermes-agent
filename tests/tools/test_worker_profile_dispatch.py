"""Behavioral dispatch contracts for provider-independent worker profiles."""

import json
from types import SimpleNamespace

from tools import delegate_tool as delegate
from tools.delegate_tool_config import _resolve_delegation_credentials


def _runtime(**kwargs):
    provider = kwargs.get("requested") or "parent"
    return {
        "provider": provider,
        "model": kwargs.get("target_model"),
        "base_url": f"https://{provider}.example.invalid/v1",
        "api_key": "fixture-credential-reference",
        "api_mode": "chat_completions",
    }


def test_two_workers_resolve_distinct_profiles_before_construction(monkeypatch):
    import agent.models_dev as models_dev
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", _runtime)
    monkeypatch.setattr(
        models_dev,
        "get_model_capabilities",
        lambda *args, **kwargs: SimpleNamespace(supports_tools=True),
    )
    cfg = {
        "profiles": {
            "small": {
                "provider": "provider-a",
                "model": "small-model",
                "reasoning_effort": "low",
                "tool_policy": {"allowed_toolsets": ["file"]},
            },
            "review": {
                "provider": "provider-b",
                "model": "review-model",
                "reasoning_effort": "high",
                "execution_limits": {"max_iterations": 12},
            },
        },
    }
    parent = SimpleNamespace(model="parent-model")
    base = _resolve_delegation_credentials(cfg, parent, "small")
    tasks = [
        {"goal": "Collect the bounded evidence", "profile": "small"},
        {"goal": "Review the resulting candidate", "profile": "review"},
    ]
    task_creds, error = delegate._resolve_task_credentials(tasks, base, cfg, parent)
    assert error is None

    built = []

    def build_child(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(tool_progress_callback=None)

    monkeypatch.setattr(delegate, "_build_child_preserving_parent_tools", build_child)
    children, error = delegate._build_children(
        tasks,
        [None, None],
        base,
        top_role="leaf",
        max_iterations=50,
        parent_agent=parent,
        routing_cfg=cfg,
        live_deleg_id=None,
        live_writers=[],
        task_creds=task_creds,
    )
    assert error is None
    assert len(children) == 2
    assert [call["model"] for call in built] == ["small-model", "review-model"]
    assert [call["requested_profile"] for call in built] == ["small", "review"]
    assert built[0]["profile_tool_policy"].allowed_toolsets == ("file",)
    assert built[1]["max_iterations"] == 12
    assert built[0]["profile_route_receipt"]["transmitted_model"] is None


def test_invalid_second_profile_preflight_returns_no_routes(monkeypatch):
    import agent.models_dev as models_dev
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", _runtime)
    monkeypatch.setattr(
        models_dev,
        "get_model_capabilities",
        lambda *args, **kwargs: SimpleNamespace(supports_tools=True),
    )
    cfg = {"profiles": {"small": {"provider": "provider-a", "model": "small-model"}}}
    parent = SimpleNamespace(model="parent-model")
    base = _resolve_delegation_credentials(cfg, parent, "small")
    routes, error = delegate._resolve_task_credentials(
        [
            {"goal": "Collect the bounded evidence", "profile": "small"},
            {"goal": "Review the resulting candidate", "profile": "missing"},
        ],
        base,
        cfg,
        parent,
    )
    assert routes == []
    assert "missing" in error


def test_explicitly_unavailable_route_fails_batch_preflight_before_construction(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    calls = []

    def runtime(**kwargs):
        calls.append(kwargs.get("target_model"))
        if kwargs.get("target_model") == "unavailable-model":
            raise ValueError("configured route is unavailable")
        return _runtime(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", runtime)
    cfg = {
        "profiles": {
            "small": {"provider": "provider-a", "model": "small-model"},
            "offline": {"provider": "provider-b", "model": "unavailable-model"},
        },
    }
    parent = SimpleNamespace(model="parent-model")
    base = _resolve_delegation_credentials(cfg, parent, "small")
    routes, error = delegate._resolve_task_credentials(
        [
            {"goal": "Collect", "profile": "small"},
            {"goal": "Must reject", "profile": "offline"},
        ],
        base,
        cfg,
        parent,
    )
    assert routes == []
    assert "unavailable" in error
    assert calls == ["small-model", "small-model", "unavailable-model"]


def test_two_worker_dispatch_receipts_match_transport_requests(monkeypatch):
    """The fan-out executor must preserve each resolved route through the real request hook."""
    import agent.models_dev as models_dev
    import hermes_cli.runtime_provider as runtime_provider
    from agent.turn_api_call import perform_api_call
    from agent.worker_receipts import observe_worker_response
    from tools import delegation_live_log

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", _runtime)
    monkeypatch.setattr(
        models_dev,
        "get_model_capabilities",
        lambda *args, **kwargs: SimpleNamespace(supports_tools=True),
    )
    monkeypatch.setattr(delegation_live_log, "create_live_transcripts", lambda *args, **kwargs: (None, [], []))
    cfg = {
        "profiles": {
            "small": {
                "provider": "provider-a",
                "model": "small-model",
                "reasoning_effort": "low",
            },
            "review": {
                "provider": "provider-b",
                "model": "review-model",
                "reasoning_effort": "high",
            },
        },
        "max_concurrent_children": 2,
    }
    monkeypatch.setattr(delegate, "_load_config", lambda: cfg)
    wire = []

    class FixtureChild:
        def __init__(self, **kwargs):
            receipt = dict(kwargs["profile_route_receipt"])
            self.provider = receipt["resolved_provider"]
            self.model = kwargs["model"]
            self.api_mode = "chat_completions"
            self.base_url = None
            self.platform = "test"
            self.session_id = f"fixture-{self.provider}"
            self._worker_route_receipt = receipt
            self._reasoning_effort = receipt["resolved_reasoning_effort"]
            self._disable_streaming = True
            self._delegate_role = "leaf"
            self._delegate_depth = 1
            self.tool_progress_callback = None
            self.session_estimated_cost_usd = 0.0
            self.session_cost_status = "unknown"

        def _has_pending_redirect(self):
            return False

        def _interruptible_api_call(self, request):
            wire.append(dict(request))
            return SimpleNamespace(model=f"reported-{self.model}")

        def run_conversation(self, **kwargs):
            verdict = perform_api_call(
                self,
                api_kwargs={"model": self.model, "reasoning_effort": self._reasoning_effort},
                _original_api_kwargs={},
                _llm_middleware_trace=[],
                _moa_prepared_request=None,
                _retry=SimpleNamespace(),
                thinking_spinner=None,
                retry_count=0,
                api_call_count=0,
                api_request_id=f"request-{self.provider}",
                effective_task_id="fixture",
                turn_id="turn",
                interrupted=False,
            )
            observe_worker_response(self, verdict.response)
            return {
                "completed": True,
                "final_response": f"done-{self.provider}",
                "api_calls": 1,
                "messages": [{"role": "assistant", "content": f"done-{self.provider}"}],
            }

        def close(self):
            return None

    monkeypatch.setattr(delegate, "_build_child_preserving_parent_tools", lambda **kwargs: FixtureChild(**kwargs))
    parent = SimpleNamespace(
        model="parent-model",
        session_id="parent-session",
        _active_children=[],
        _interrupt_requested=False,
    )
    result = json.loads(
        delegate.delegate_task(
            tasks=[
                {"goal": "Collect the bounded evidence", "profile": "small"},
                {"goal": "Review the resulting candidate", "profile": "review"},
            ],
            profile="small",
            background=False,
            credentials_cfg=cfg,
            parent_agent=parent,
        )
    )
    wire_by_model = {item["model"]: item for item in wire}
    assert wire_by_model["small-model"]["reasoning_effort"] == "low"
    assert wire_by_model["review-model"]["reasoning_effort"] == "high"
    receipts = {item["route"]["resolved_model"]: item["route"] for item in result["results"]}
    assert receipts["small-model"]["transmitted_model"] == "small-model"
    assert receipts["small-model"]["transmitted_reasoning_effort"] == "low"
    assert receipts["small-model"]["provider_reported_model"] == "reported-small-model"
    assert receipts["review-model"]["transmitted_model"] == "review-model"
    assert receipts["review-model"]["transmitted_reasoning_effort"] == "high"
    assert receipts["review-model"]["provider_reported_model"] == "reported-review-model"


def test_execute_code_nested_tools_obey_the_worker_tool_ceiling(monkeypatch):
    from tools import code_execution_tool

    assert code_execution_tool._sandbox_tools_for(["execute_code"]) == frozenset()
    assert code_execution_tool._sandbox_tools_for(["execute_code", "read_file"]) == frozenset({"read_file"})
    assert code_execution_tool._sandbox_tools_for(None)  # legacy callers without a session ceiling retain behavior
    captured = {}
    monkeypatch.setattr(
        code_execution_tool,
        "execute_code",
        lambda **kwargs: captured.update(kwargs) or "{}",
    )
    code_execution_tool._execute_code_handler(
        {"code": "print('ok')"},
        task_id="task",
        enabled_tools=["execute_code", "read_file"],
        worker_max_tool_calls=2,
    )
    assert captured["worker_max_tool_calls"] == 2
    assert captured["enabled_tools"] == ["execute_code", "read_file"]
