"""Worker-policy seam: default-worker-only without a policy; admission before credentials; leases."""
from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools import worker_policy as W

BASE = dict(provider="example-provider", model="default-model", base_url="", fallback_providers=[])


class ExamplePolicy:
    """Two named lanes; only parents whose session_id starts with 'trusted' may pick them."""

    LANES = {"big": {"provider": "example-provider", "model": "big-model"}}

    def __init__(self):
        self.leases = []
        self.lock = threading.Lock()

    def admit(self, tasks, routing_cfg, delegation_cfg, *, parent, home):
        trusted = bool(parent and str(parent.session_id or "").startswith("trusted") and parent._delegate_depth == 0)
        for task in tasks:
            lane = task.get("worker", "default")
            if lane != "default" and (lane not in self.LANES or not trusted):
                raise ValueError("Worker refused: lane %r not allowed for this parent" % lane)
        return routing_cfg, tasks, None

    def task_route(self, task, base_cfg):
        cfg = copy.deepcopy(base_cfg)
        if task.get("worker") in self.LANES:
            cfg.update(self.LANES[task["worker"]])
            cfg["fallback_providers"] = []
        return cfg

    def validate_credentials(self, task, credentials):
        if task.get("worker") in self.LANES and credentials.get("model") != self.LANES[task["worker"]]["model"]:
            raise ValueError("Worker refused: credentials do not match the lane")

    def child_routing(self, task, routing_cfg):
        return dict(routing_cfg, fallback_providers=[]) if task.get("worker") in self.LANES else routing_cfg

    def configure_child(self, child, task, *, parent, home):
        if task.get("worker") in self.LANES:
            child._delegate_worker = task["worker"]

    def run_child(self, child, **kwargs):
        with self.lock:
            self.leases.append(("acquire", getattr(child, "_delegate_worker", None)))
        try:
            return child.run_conversation(**kwargs)
        finally:
            with self.lock:
                self.leases.append(("release", getattr(child, "_delegate_worker", None)))

    def refresh_effort(self, child):
        pass

    def parent_authorized(self, parent, home=None):
        return bool(parent and str(parent.session_id or "").startswith("trusted"))

    def task_schema(self):
        return {"worker": {"type": "string", "enum": ["default", *self.LANES], "description": "Named lane."}}


class Raising:
    def __getattr__(self, name):
        if name in W.REQUIRED_METHODS:
            def boom(*a, **k):
                raise RuntimeError("policy bug")
            return boom
        raise AttributeError(name)


@pytest.fixture
def registry(monkeypatch):
    import hermes_cli.plugins as plugins
    manager = PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    def install(policy, name="example-workers", key="example", enabled=True):
        manifest = PluginManifest(name=name)
        PluginContext(manifest, manager).register_worker_policy(key, policy)
        manager._plugins[name] = NS(enabled=enabled, error=None, manifest=manifest)
    return manager, install


@pytest.mark.parametrize("task", [{"worker": "big"}, {"effort": "high"}, {"work_class": "x"}])
@pytest.mark.parametrize("mode", ["missing", "raising", "disabled", "duplicate"])
def test_without_a_healthy_policy_every_selector_is_refused(registry, task, mode):
    _, install = registry
    if mode == "raising":
        install(Raising())
    elif mode == "disabled":
        install(ExamplePolicy(), enabled=False)
    elif mode == "duplicate":
        install(ExamplePolicy(), name="a", key="a")
        install(ExamplePolicy(), name="b", key="b")
    with pytest.raises(ValueError, match="no worker policy is available"):
        W.admit([dict(task, goal="x")], dict(BASE), dict(BASE), parent_agent=NS(_delegate_depth=0, session_id="trusted"))


def test_default_task_runs_unchanged_without_policy(registry):
    policy, routing, tasks, used = W.admit([{"goal": "x"}], dict(BASE), dict(BASE), parent_agent=None)
    assert policy is None and routing == BASE and tasks == [{"goal": "x"}] and used is None
    assert W.parent_authorized(NS(session_id="trusted")) is False and W.task_schema_properties() == {}


def test_policy_refusal_propagates_and_trusted_parent_gets_lane(registry):
    _, install = registry
    install(ExamplePolicy())
    with pytest.raises(ValueError, match="not allowed"):
        W.admit([{"goal": "x", "worker": "big"}], dict(BASE), dict(BASE), parent_agent=NS(_delegate_depth=0, session_id="other"))
    with pytest.raises(ValueError, match="not allowed"):
        W.admit([{"goal": "x", "worker": "big"}], dict(BASE), dict(BASE), parent_agent=NS(_delegate_depth=1, session_id="trusted"))
    policy, routing, tasks, _ = W.admit([{"goal": "x", "worker": "big"}], dict(BASE), dict(BASE),
                                        parent_agent=NS(_delegate_depth=0, session_id="trusted-1"))
    assert W.task_route(policy, tasks[0], routing)["model"] == "big-model"


def test_policy_receives_frozen_provenance_not_the_agent(registry):
    _, install = registry
    seen = {}

    class Spy(ExamplePolicy):
        def admit(self, tasks, routing_cfg, delegation_cfg, *, parent, home):
            seen["parent"] = parent
            tasks[0]["goal"] = "mutated"
            return routing_cfg, tasks, None
    install(Spy())
    agent = MagicMock()
    agent._delegate_depth = 0
    agent.session_id = "s1"
    original = [{"goal": "x"}]
    W.admit(original, dict(BASE), dict(BASE), parent_agent=agent)
    parent = seen["parent"]
    assert isinstance(parent, W.ParentOrigin) and parent.session_id == "s1"
    assert not hasattr(parent, "run_conversation")
    with pytest.raises(Exception):
        parent.session_id = "forged"
    assert original == [{"goal": "x"}]


def test_named_lane_without_lease_owner_is_refused_and_pins_survive_unload():
    child = NS(_delegate_worker="big", run_conversation=MagicMock(return_value="ran"))
    with pytest.raises(RuntimeError, match="lease owner"):
        W.run_child(child, user_message="x")
    child.run_conversation.assert_not_called()
    policy = ExamplePolicy()
    pinned = NS(run_conversation=lambda **kw: "done")
    W.configure_child(policy, pinned, {"worker": "big"}, parent_agent=None)
    assert W.run_child(pinned, user_message="x") == "done"
    assert policy.leases == [("acquire", "big"), ("release", "big")]
    mock_child = MagicMock()
    mock_child.run_conversation.return_value = "plain"
    assert W.run_child(mock_child) == "plain"  # auto-mock attributes never count as a pinned policy


def test_registration_validated_and_load_owned():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="example"), manager)
    with pytest.raises(ValueError):
        ctx.register_worker_policy("bad", object())
    registration = ctx.register_worker_policy("w", ExamplePolicy())
    with pytest.raises(ValueError):
        ctx.register_worker_policy("w", ExamplePolicy())
    registration.dispose()
    assert manager._worker_policies == {}


def test_schema_selectors_only_from_policy(registry):
    from tools import delegate_tool
    items = delegate_tool._build_dynamic_schema_overrides()["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "worker" not in items
    _, install = registry
    install(ExamplePolicy())
    items = delegate_tool._build_dynamic_schema_overrides()["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert items["worker"]["enum"] == ["default", "big"]


def _dispatch(monkeypatch, tasks, parent):
    from tools import delegate_tool as delegate
    monkeypatch.setattr(delegate, "_load_config", lambda: copy.deepcopy(BASE))
    monkeypatch.setattr(delegate, "_get_max_spawn_depth", lambda: 2)
    monkeypatch.setattr(delegate, "_get_max_concurrent_children", lambda: 8)
    monkeypatch.setattr(delegate, "is_spawn_paused", lambda: False)
    monkeypatch.setattr(delegate, "_oneshot_spawn_budget", lambda *a: None)
    resolved = []
    from tools import delegate_tool_routes as routes
    def resolve(cfg, p):
        resolved.append(copy.deepcopy(cfg))
        return routes.TaskRoute(dict(cfg), dict(cfg))
    monkeypatch.setattr(routes, "route_for_config", resolve)
    built = {}
    def fake_build(*a, **k):
        built.update(k)
        built["routes"] = a[2]
        return [], "offline boundary"
    monkeypatch.setattr(delegate, "_build_children", fake_build)
    out = json.loads(delegate.delegate_task(tasks=tasks, parent_agent=parent))
    return out, resolved, built


def test_dispatch_refuses_selector_without_policy_before_child_construction(monkeypatch, registry):
    parent = MagicMock()
    parent._delegate_depth = 0
    out, resolved, built = _dispatch(monkeypatch, [{"goal": "default worker task"}, {"goal": "named worker task", "worker": "big"}], parent)
    assert "no worker policy is available" in json.dumps(out) and built == {}
    assert len(resolved) <= 1  # only the pre-existing batch default resolution may have run


def test_dispatch_resolves_lane_credentials_per_task_after_admission(monkeypatch, registry):
    _, install = registry
    install(ExamplePolicy())
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "trusted-1"
    out, resolved, built = _dispatch(monkeypatch, [{"goal": "default worker task"}, {"goal": "named worker task", "worker": "big"}], parent)
    assert "offline boundary" in json.dumps(out)
    assert [r.creds["model"] for r in built["routes"]] == ["default-model", "big-model"]
    assert built["worker_policy"] is not None
