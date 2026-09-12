"""Pin x delegation.fallback_providers decision table for child agents (#80450, #65038)."""

import json
from types import SimpleNamespace

from unittest.mock import MagicMock, patch

import pytest

from tools.delegate_tool import _build_child_agent
from tools.delegate_tool_config import _resolve_child_fallback_chain
from tests.tools.test_delegate import _make_mock_parent

PARENT_CHAIN = [
    {"provider": "openrouter", "model": "gpt-4o-mini", "api_key": "sk-or-parent"}
]
DECLARED_CHAIN = [
    {"provider": "deepseek", "model": "deepseek-chat", "api_key": "sk-ds-child"}
]


def _parent(chain=None):
    parent = _make_mock_parent(depth=0)
    parent._fallback_chain = chain
    return parent



# (pinned, declared fallback_providers, expected) — the six cells plus the malformed edges.
@pytest.mark.parametrize(
    ("pinned", "declared", "expected"),
    [
        (True, "absent", None),                 # #80450: a pinned child fails loudly, never reroutes
        (True, None, None),
        (True, [], None),
        (True, DECLARED_CHAIN, DECLARED_CHAIN),
        (False, "absent", PARENT_CHAIN),        # historical default preserved
        (False, None, PARENT_CHAIN),
        (False, [], None),                      # explicit [] disables fallback
        (False, DECLARED_CHAIN, DECLARED_CHAIN),  # #65038: delegation.fallback_providers reaches the child
        (True, "not-a-list", None),
        (False, "not-a-list", PARENT_CHAIN),
        (True, [{"provider": "deepseek"}], None),
        (False, [{"provider": "deepseek"}], PARENT_CHAIN),
        (False, [{"provider": "deepseek", "model": "x"}, {"provider": "deepseek"}],
         [{"provider": "deepseek", "model": "x"}]),  # a valid route survives a malformed neighbour
    ],
)
def test_child_fallback_chain_matrix(pinned, declared, expected):
    cfg = {} if declared == "absent" else {"fallback_providers": declared}
    assert _resolve_child_fallback_chain(_parent(list(PARENT_CHAIN)), cfg, pinned=pinned) == expected


def _spawn_kwargs(parent, cfg, **overrides):
    model = overrides.pop("model", None)
    with patch("tools.delegate_tool._load_config", return_value=cfg), patch("run_agent.AIAgent") as MockAgent:
        MockAgent.return_value = MagicMock()
        _build_child_agent(task_index=0, goal="matrix wiring", context=None, toolsets=None, model=model,
                           max_iterations=10, parent_agent=parent, task_count=1, **overrides)
    return MockAgent.call_args[1]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, PARENT_CHAIN),                                                    # unpinned inherits
        ({"model": "deepseek-chat"}, None),                                    # model-only pin (#80450 model arm)
        ({"override_provider": "minimax", "override_base_url": "https://api.minimax.example/v1",
          "override_api_key": "sk-mm"}, None),                                 # provider pin
    ],
)
def test_pin_is_derived_from_provider_base_url_or_model(overrides, expected):
    assert _spawn_kwargs(_parent(list(PARENT_CHAIN)), {}, **overrides)["fallback_model"] == expected


def test_declared_chain_flows_through_real_profile_config_loader(
    tmp_path, monkeypatch
):
    """The public key must survive DEFAULT_CONFIG/profile loading without
    patching ``_load_config`` and reach the child constructor."""
    import yaml

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    token = set_hermes_home_override(tmp_path)
    try:
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {"delegation": {"fallback_providers": list(DECLARED_CHAIN)}}
            ),
            encoding="utf-8",
        )
        with patch("run_agent.AIAgent") as mock_agent:
            mock_agent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="real config loader",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=_parent(list(PARENT_CHAIN)),
                task_count=1,
            )
    finally:
        reset_hermes_home_override(token)

    child_kwargs = mock_agent.call_args.kwargs
    assert child_kwargs["fallback_model"] == DECLARED_CHAIN


def test_explicit_empty_chain_survives_real_profile_config_loader(tmp_path, monkeypatch):
    """An explicit [] remains an authoritative disable after config loading."""
    import yaml

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    token = set_hermes_home_override(tmp_path)
    try:
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"delegation": {"fallback_providers": []}}),
            encoding="utf-8",
        )
        with patch("run_agent.AIAgent") as mock_agent:
            mock_agent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="explicit disable",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=_parent(list(PARENT_CHAIN)),
                task_count=1,
            )
    finally:
        reset_hermes_home_override(token)

    assert mock_agent.call_args.kwargs["fallback_model"] is None


def test_pinned_review_does_not_borrow_general_worker_chain(tmp_path, monkeypatch):
    """The public /review route owns its fallback policy as well as its model."""
    import yaml

    from agent.review_engine import start_review
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "delegation": {
                    "fallback_providers": [
                        {"provider": "deepseek", "model": "worker-fallback"}
                    ]
                },
                "auxiliary": {
                    "review": {
                        "provider": "custom",
                        "model": "review-model",
                        "base_url": "http://127.0.0.1:18479/v1",
                        "api_key": "test-only",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    parent = _parent(list(PARENT_CHAIN))
    parent.session_id = "review-80479-parent"
    captured = {}

    class ReachedConstructor(RuntimeError):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise ReachedConstructor()

    token = set_hermes_home_override(tmp_path)
    try:
        with patch("run_agent.AIAgent", side_effect=capture):
            with pytest.raises(ReachedConstructor):
                start_review(
                    parent,
                    [{"role": "user", "content": "Check the last result"}],
                )
    finally:
        reset_hermes_home_override(token)

    assert captured["model"] == "review-model"
    assert captured["base_url"] == "http://127.0.0.1:18479/v1"
    assert captured["fallback_model"] is None


def test_declared_child_chain_activates_on_primary_failure():
    """The selected chain is accepted by the real fallback activation rail."""
    from agent.error_classifier import FailoverReason
    from run_agent import AIAgent

    chain = _resolve_child_fallback_chain(
        _parent(list(PARENT_CHAIN)),
        {"fallback_providers": list(DECLARED_CHAIN)},
        pinned=True,
    )
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        child = AIAgent(
            api_key="primary-test-key",
            base_url="https://primary.example/v1",
            model="primary-model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=chain,
        )
    fallback_client = MagicMock()
    fallback_client.base_url = "https://fallback.example/v1"
    fallback_client.api_key = "fallback-test-key"
    with (
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "deepseek-chat"),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda model, _provider: model,
        ),
    ):
        assert child._try_activate_fallback(FailoverReason.rate_limit) is True

    assert child.model == "deepseek-chat"
    assert child.provider == "deepseek"


def test_routed_child_fallback_is_one_terminal_attempt_without_native_regression(tmp_path):
    from agent.execution_router import (
        ExecutionKind,
        ExecutionRouteDecisionV1,
        ExecutionRouterProviderDescriptorV1,
        RoutedAttemptRestartRequired,
    )
    from hermes_cli.execution_router_runtime import ExecutionRouterRegistration
    from hermes_state import SessionDB
    from tools.delegate_tool import delegate_task

    decisions = []

    class Provider:
        descriptor = ExecutionRouterProviderDescriptorV1(
            plugin_id="router-plugin",
            plugin_version="1.0.0",
            provider_id="router-provider",
            contract_version="1.0",
            supported_execution_kinds=(ExecutionKind.NATIVE_CHILD,),
        )

        def resolve_execution_route(self, request, _cancellation):
            decisions.append(request)
            if request.instruction.text == "pass sibling":
                return ExecutionRouteDecisionV1.pass_through(
                    request_id=request.request_id,
                    attempt_id=request.attempt_id,
                )
            return ExecutionRouteDecisionV1.route(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                candidate_id="fallback-0",
            )

    registration = ExecutionRouterRegistration(Provider(), 1, lambda generation: generation == 1)
    manager = MagicMock()
    manager.get_execution_router_registration.return_value = registration
    parent = _make_mock_parent()
    parent.session_id = "slice-b-parent"
    parent._fallback_chain = [{"provider": "target-provider", "model": "target-model"}]
    db = SessionDB(tmp_path / "state.db")
    db.create_session(parent.session_id, source="cli")
    parent._session_db = db
    cfg = {
        "max_iterations": 9,
        "fallback_providers": list(parent._fallback_chain),
    }
    calls = {"credentials": 0, "build": 0, "submit": 0, "schema": 0}
    children = []
    cleanup_counts = {}
    completed = []

    def credentials(child_cfg, _parent):
        calls["credentials"] += 1
        return {
            "provider": child_cfg.get("provider"),
            "model": child_cfg.get("model"),
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "request_overrides": None,
        }

    def build(**kwargs):
        calls["build"] += 1
        child = SimpleNamespace(
            goal=kwargs["goal"],
            requested_provider=kwargs["override_provider"] or "native-provider",
            provider=kwargs["override_provider"] or "native-provider",
            model=kwargs["model"] or "native-model",
            reasoning_config=None,
            tool_progress_callback=None,
            _delegate_role="leaf",
            _subagent_id=f"slice-b-{len(children)}",
            session_id=f"slice-b-child-{len(children)}",
            _fallback_chain=list(parent._fallback_chain),
            _fallback_index=0,
        )
        children.append(child)
        return child

    class FakeHeartbeat:
        def start(self):
            return None

    class FakeRun:
        def __init__(self, child, _parent, task_index, goal, *_args):
            self.child = child
            self.task_index = task_index
            self.goal = goal
            self.child_task_id = child.session_id
            self.relay_text = None

        def seed_workspace(self):
            return None

        def await_child(self):
            calls["submit"] += 1
            if self.goal == "ordinary error":
                raise RuntimeError("ordinary child failure")
            if self.goal == "routed fallback":
                consumed = self.child._fallback_index
                self.child._fallback_index += 1
                self.child._routed_restart_required = RoutedAttemptRestartRequired(
                    "sk-raw-secret", consumed
                )
            return {"completed": True, "summary": f"done: {self.goal}"}, None, False

        def elapsed(self):
            return 0.01

        def append_sibling_write_reminder(self, _entry):
            return None

        def account_background_processes(self, _entry):
            return None

        def emit_complete(self, _result, entry, _duration):
            completed.append((self.goal, entry["status"]))

        def close_steering(self):
            return None

        def finish_failed(self, entry, _late_steer, **_kwargs):
            completed.append((self.goal, entry["status"]))
            return entry

        def attach_worktree(self, entry):
            return entry

        def cleanup(self, **_kwargs):
            cleanup_counts[id(self.child)] = cleanup_counts.get(id(self.child), 0) + 1

    def validate_schema(child, *_args):
        calls["schema"] += 1
        return child.goal

    def result_entry(child, result, task_index, duration, _schema):
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": result["summary"],
            "api_calls": 1,
            "duration_seconds": duration,
            "_child_role": child._delegate_role,
        }

    common_patches = (
        patch("hermes_cli.plugins.get_plugin_manager", return_value=manager),
        patch("tools.delegate_tool._load_config", return_value=cfg),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=credentials),
        patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=build),
        patch("tools.delegate_tool._ChildRun", FakeRun),
        patch("tools.delegate_tool._start_heartbeat", return_value=FakeHeartbeat()),
        patch("tools.delegate_tool._lease_child_credential", return_value=(None, None)),
        patch("tools.delegate_tool._register_child", return_value=None),
        patch("tools.delegate_tool._validate_child_output_schema", side_effect=validate_schema),
        patch("tools.delegate_tool._build_result_entry", side_effect=result_entry),
        patch("tools.delegate_tool._merge_late_steer"),
        patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])),
    )

    with common_patches[0], common_patches[1], common_patches[2], common_patches[3], \
            common_patches[4], common_patches[5], common_patches[6], common_patches[7], \
            common_patches[8], common_patches[9], common_patches[10], common_patches[11]:
        first = json.loads(delegate_task(
            tasks=[
                {"goal": "routed fallback"},
                {"goal": "pass sibling"},
                {"goal": "ordinary error"},
            ],
            parent_agent=parent,
        ))
        assert [entry["task_index"] for entry in first["results"]] == [0, 1, 2]
        assert [entry["status"] for entry in first["results"]] == ["error", "completed", "error"]
        routed_failure = json.dumps(first["results"][0])
        assert len(routed_failure) < 700
        assert "target-provider" not in routed_failure
        assert "target-model" not in routed_failure
        assert "sk-raw" not in routed_failure
        assert children[0]._fallback_index == 1
        assert hasattr(children[1], "_execution_router_native_child_attempt")
        assert not hasattr(children[1], "_execution_router_selected_attempt")
        assert calls == {"credentials": 3, "build": 3, "submit": 3, "schema": 1}

        first_route_request = decisions[0]
        route_events = [
            event for event in db.execution_route_lifecycle(parent.session_id).read_events(limit=100)
            if event.request_id == first_route_request.request_id
            and event.attempt_id == first_route_request.attempt_id
        ]
        assert sorted(event.event_type.value for event in route_events) == [
            "route_accepted", "route_finished", "route_requested", "route_started",
        ]
        assert next(
            event for event in route_events if event.event_type.value == "route_finished"
        ).terminal_state == "routed_restart_required"

        pass_request = decisions[1]
        pass_events = [
            event for event in db.execution_route_lifecycle(parent.session_id).read_events(limit=100)
            if event.request_id == pass_request.request_id
            and event.attempt_id == pass_request.attempt_id
        ]
        assert sorted(event.event_type.value for event in pass_events) == [
            "route_finished", "route_requested", "route_started",
        ]

        error_request = decisions[2]
        error_events = [
            event for event in db.execution_route_lifecycle(parent.session_id).read_events(limit=100)
            if event.request_id == error_request.request_id
            and event.attempt_id == error_request.attempt_id
        ]
        assert sorted(event.event_type.value for event in error_events) == [
            "route_accepted", "route_finished", "route_requested", "route_started",
        ]
        assert next(
            event for event in error_events if event.event_type.value == "route_finished"
        ).terminal_state == "error"

        later = json.loads(delegate_task(goal="later routed child", parent_agent=parent))
        assert later["results"][0]["status"] == "completed"
        assert len(decisions) == 4
        assert decisions[3].request_id != first_route_request.request_id
        assert decisions[3].attempt_id != first_route_request.attempt_id
        later_events = [
            event for event in db.execution_route_lifecycle(parent.session_id).read_events(limit=100)
            if event.request_id == decisions[3].request_id
            and event.attempt_id == decisions[3].attempt_id
        ]
        assert sorted(event.event_type.value for event in later_events) == [
            "route_accepted", "route_finished", "route_requested", "route_started",
        ]
        assert sum(event.event_type.value == "route_finished" for event in later_events) == 1

        manager.get_execution_router_registration.return_value = None
        native = json.loads(delegate_task(goal="no router child", parent_agent=parent))
        assert native["results"][0]["status"] == "completed"
        assert len(decisions) == 4
        assert not hasattr(children[-1], "_execution_router_native_child_attempt")

    from agent.execution_router import ExecutionRouteCandidateV1
    from hermes_cli.execution_router_runtime import (
        _prepare_native_child_attempt,
        _start_native_child_attempt,
    )
    from tools.delegate_tool import _run_single_child

    class SetupPool:
        def __init__(self):
            self.releases = []

        def release_lease(self, lease_id):
            self.releases.append(lease_id)

    class SetupHeartbeat:
        def __init__(self):
            self.stops = 0

        def stop(self):
            self.stops += 1

    setup_candidates = (
        ExecutionRouteCandidateV1("native", "native-provider", "native-model", None),
        ExecutionRouteCandidateV1("fallback-0", "target-provider", "target-model", None),
    )
    setup_failures = {
        "_lease_child_credential": "setup lease failed",
        "_start_heartbeat": "setup heartbeat failed",
        "_register_child": "setup register failed",
        "_ChildRun": "setup run failed",
    }
    for seam in ("_lease_child_credential", "_start_heartbeat", "_register_child", "_ChildRun"):
        prepared = _prepare_native_child_attempt(
            raw_instruction=f"setup failure at {seam}",
            surface_class="delegate_task",
            session_id=parent.session_id,
            task_index=20 + len(decisions),
            native_candidate_id="native",
            eligible_candidates=setup_candidates,
            registration=registration,
            lifecycle=db.execution_route_lifecycle(parent.session_id),
        )
        pool, heartbeat = SetupPool(), SetupHeartbeat()
        active_child = SimpleNamespace(
            requested_provider="target-provider",
            provider="target-provider",
            model="target-model",
            reasoning_config=None,
            tool_progress_callback=None,
            close=MagicMock(),
        )
        _start_native_child_attempt(prepared, active_child)
        unregister = MagicMock()
        seam_patches = {
            "_lease_child_credential": patch(
                "tools.delegate_tool._lease_child_credential",
                side_effect=RuntimeError(setup_failures[seam]),
            ),
            "_start_heartbeat": patch(
                "tools.delegate_tool._start_heartbeat",
                side_effect=RuntimeError(setup_failures[seam]),
            ),
            "_register_child": patch(
                "tools.delegate_tool._register_child",
                side_effect=RuntimeError(setup_failures[seam]),
            ),
            "_ChildRun": patch(
                "tools.delegate_tool._ChildRun",
                side_effect=RuntimeError(setup_failures[seam]),
            ),
        }
        with (
            patch("tools.delegate_tool._lease_child_credential", return_value=(pool, "lease")),
            patch("tools.delegate_tool._start_heartbeat", return_value=heartbeat),
            patch("tools.delegate_tool._register_child", return_value="setup-child"),
            patch("tools.delegate_tool._unregister_subagent", unregister),
            seam_patches[seam],
        ):
            with pytest.raises(RuntimeError, match=setup_failures[seam]):
                _run_single_child(
                    task_index=20,
                    goal="setup failure",
                    child=active_child,
                    parent_agent=parent,
                )

        setup_events = [
            event for event in db.execution_route_lifecycle(parent.session_id).read_events(limit=100)
            if event.request_id == prepared.request.request_id
            and event.attempt_id == prepared.request.attempt_id
        ]
        assert sorted(event.event_type.value for event in setup_events) == [
            "route_accepted", "route_finished", "route_requested", "route_started",
        ]
        assert next(
            event for event in setup_events if event.event_type.value == "route_finished"
        ).terminal_state == "child_setup_failed"
        assert sum(event.event_type.value == "route_finished" for event in setup_events) == 1
        active_child.close.assert_called_once_with()
        assert pool.releases == ([] if seam == "_lease_child_credential" else ["lease"])
        assert heartbeat.stops == (1 if seam in {"_register_child", "_ChildRun"} else 0)
        assert unregister.call_count == (1 if seam == "_ChildRun" else 0)

        native_pool, native_heartbeat = SetupPool(), SetupHeartbeat()
        native_child = SimpleNamespace(tool_progress_callback=None, close=MagicMock())
        native_unregister = MagicMock()
        native_seam_patches = {
            "_lease_child_credential": patch(
                "tools.delegate_tool._lease_child_credential",
                side_effect=RuntimeError(setup_failures[seam]),
            ),
            "_start_heartbeat": patch(
                "tools.delegate_tool._start_heartbeat",
                side_effect=RuntimeError(setup_failures[seam]),
            ),
            "_register_child": patch(
                "tools.delegate_tool._register_child",
                side_effect=RuntimeError(setup_failures[seam]),
            ),
            "_ChildRun": patch(
                "tools.delegate_tool._ChildRun",
                side_effect=RuntimeError(setup_failures[seam]),
            ),
        }
        with (
            patch("tools.delegate_tool._lease_child_credential", return_value=(native_pool, "native-lease")),
            patch("tools.delegate_tool._start_heartbeat", return_value=native_heartbeat),
            patch("tools.delegate_tool._register_child", return_value="native-child"),
            patch("tools.delegate_tool._unregister_subagent", native_unregister),
            native_seam_patches[seam],
        ):
            with pytest.raises(RuntimeError, match=setup_failures[seam]):
                _run_single_child(
                    task_index=21,
                    goal="native setup failure",
                    child=native_child,
                    parent_agent=parent,
                )
        native_child.close.assert_not_called()
        assert native_pool.releases == []
        assert native_heartbeat.stops == 0
        native_unregister.assert_not_called()

    assert calls == {"credentials": 5, "build": 5, "submit": 5, "schema": 3}
    assert all(count == 1 for count in cleanup_counts.values())
    assert sorted(completed) == sorted([
        ("routed fallback", "error"),
        ("pass sibling", "completed"),
        ("ordinary error", "error"),
        ("later routed child", "completed"),
        ("no router child", "completed"),
    ])
    db.close()


if __name__ == "__main__":
    unittest.main()
