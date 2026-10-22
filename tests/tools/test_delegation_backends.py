"""delegation.backend: pluggable delegation providers decide each delegate_task task's route.

Real discovery through PluginManager against a temp HERMES_HOME, real delegate_task, real credential
resolution; only child construction/run are stubbed where a test reads what they were handed.
"""

import json
import types

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager

PROBE_PLUGIN = '''
from agent.delegation_provider import DelegationProvider, DelegationRouteError

class Options(DelegationProvider):
    @property
    def name(self):
        return "probe-options"

    def task_parameters(self, config):
        opts = self.settings(config).get("options") or []
        return {"option": {"type": "string", "enum": [o["name"] for o in opts],
                           "description": "; ".join(f"{o['name']}: {o['description']}" for o in opts)}}

    def description(self, config):
        return "Pick an `option` per task."

    def route_task(self, task, config):
        if task.get("option") == "refused":
            raise DelegationRouteError("that option is reserved for humans")
        for o in self.settings(config).get("options") or []:
            if o["name"] == task.get("option"):
                return {"base_url": o["base_url"], "api_key": "k-" + o["name"], "model": o["model"],
                        "reasoning_effort": o.get("reasoning_effort"), "label": o["name"],
                        "fallback_providers": o.get("fallback_providers")}
        return dict(config)

def register(ctx):
    ctx.register_delegation_provider(Options())
'''

OPTIONS = [
    {"name": "fast", "description": "cheap and quick", "base_url": "http://127.0.0.1:9/v1", "model": "fast-model"},
    {"name": "deep", "description": "slow and careful", "base_url": "http://127.0.0.1:9/v1", "model": "deep-model",
     "reasoning_effort": "high"},
    {"name": "refused", "description": "never allowed", "base_url": "http://127.0.0.1:9/v1", "model": "x"},
]


def _home(tmp_path, monkeypatch, delegation: dict, plugins=("probe-options",)):
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "probe-options"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "probe-options", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(PROBE_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": list(plugins)}, "delegation": delegation}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    manager = PluginManager()
    manager.discover_and_load()
    return home, manager


def _definition():
    from tools.registry import registry
    return registry.get_definitions({"delegate_task"})[0]["function"]


def _task_props(definition):
    return definition["parameters"]["properties"]["tasks"]["items"]["properties"]


@pytest.fixture
def recorded(monkeypatch):
    """Record what each child is built with; children 'complete' immediately."""
    from tools import delegate_tool as dt
    built = []

    def build(**kwargs):
        built.append(kwargs)
        return types.SimpleNamespace(tool_progress_callback=None, model=kwargs.get("model"))

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(dt, "_run_single_child", lambda task_index, goal, child=None, parent_agent=None, **kw: {
        "task_index": task_index, "status": "completed", "summary": "ok", "api_calls": 1, "duration_seconds": 0})
    return built


def _parent(session_id="root"):
    return types.SimpleNamespace(session_id=session_id, model="parent-model", provider="openrouter",
                                 tool_progress_callback=None, _delegate_spinner=None, request_overrides={},
                                 _safe_print=lambda line: None)


def test_selected_plugin_backend_extends_schema_and_routes_each_task(tmp_path, monkeypatch, recorded):
    _, manager = _home(tmp_path, monkeypatch, {"backend": "probe-options", "backends": {"probe-options": {"options": OPTIONS}}})
    try:
        from agent.delegation_registry import get_active_provider
        from tools.delegate_tool import delegate_task
        from tools.delegate_tool_config import _load_config

        assert get_active_provider(_load_config()).name == "probe-options"
        definition = _definition()
        option = _task_props(definition)["option"]
        assert option["enum"] == ["fast", "deep", "refused"]
        assert "deep: slow and careful" in option["description"]
        assert definition["description"].endswith("Pick an `option` per task.")
        # Byte-stable for a session: the same config yields the same definition.
        assert json.dumps(_definition()) == json.dumps(definition)

        out = json.loads(delegate_task(tasks=[
            {"goal": "summarize the quick thing", "option": "fast"},
            {"goal": "think hard about the slow thing", "option": "deep"},
            {"goal": "no option: plain delegation config"},
        ], parent_agent=_parent()))
        assert [r["status"] for r in out["results"]] == ["completed"] * 3
        fast, deep, plain = recorded
        assert (fast["model"], fast["override_base_url"], fast["override_api_key"]) == (
            "fast-model", "http://127.0.0.1:9/v1", "k-fast")
        assert (deep["model"], deep["override_reasoning_effort"]) == ("deep-model", "high")
        # The no-option task falls back to the delegation section, i.e. inherit the parent.
        assert plain["model"] is None and plain["override_provider"] is None and plain["profile_home"] is None
    finally:
        manager.unload()


def test_an_unroutable_task_is_refused_alone_and_its_siblings_run(tmp_path, monkeypatch, recorded):
    """Fail closed per task: a bad parameter or a backend refusal refuses that task only, its result entry says why
    and which backend refused it; the call is a tool error only when no task can run."""
    _, manager = _home(tmp_path, monkeypatch, {"backend": "probe-options", "backends": {"probe-options": {"options": OPTIONS}}})
    try:
        from tools.delegate_tool import delegate_task

        out = json.loads(delegate_task(tasks=[
            {"goal": "a task with a typo in it", "option": "fsat"},
            {"goal": "summarize the quick thing", "option": "fast"},
            {"goal": "a task the backend refuses", "option": "refused"},
        ], parent_agent=_parent()))
        typo, fast, refused = out["results"]
        assert [r["task_index"] for r in out["results"]] == [0, 1, 2]
        assert typo["status"] == "error" and "must be one of fast, deep, refused" in typo["error"]
        assert refused["status"] == "error" and "reserved for humans" in refused["error"]
        assert typo["route"] == {"backend": "probe-options", "refused": True}
        assert fast["status"] == "completed" and [b["model"] for b in recorded] == ["fast-model"]

        bad = json.loads(delegate_task(tasks=[{"goal": "only a refused task", "option": "refused"}],
                                       parent_agent=_parent()))
        assert "reserved for humans" in bad["error"] and len(recorded) == 1
    finally:
        manager.unload()


def test_a_missing_backend_fails_closed_and_default_keeps_todays_route(tmp_path, monkeypatch, recorded):
    """Unset and ``default`` are today's behaviour; a backend nobody provides is a tool error naming it and the fix,
    never a quiet run on the default route."""
    from tools.delegate_tool import delegate_task

    pinned = {"model": "pinned-model", "base_url": "http://127.0.0.1:9/v1", "api_key": "k"}
    definitions, outcomes = [], []
    for delegation in ({}, {"backend": "default"}, {"backend": "not-installed"}):
        _, manager = _home(tmp_path / str(len(definitions)), monkeypatch, {**delegation, **pinned}, plugins=())
        try:
            definitions.append(json.dumps(_definition()))
            outcomes.append(json.loads(delegate_task(tasks=[{"goal": "one plain delegated task"}],
                                                     parent_agent=_parent())))
        finally:
            manager.unload()
    assert len(set(definitions)) == 1 and '"option"' not in definitions[0]
    assert [(b["model"], b["override_base_url"]) for b in recorded] == [("pinned-model", "http://127.0.0.1:9/v1")] * 2
    error = outcomes[2]["error"]
    assert "'not-installed' is not installed" in error and "hermes config set delegation.backend default" in error


def test_a_backend_route_without_its_own_chain_inherits_delegation_fallback_providers(tmp_path, monkeypatch):
    from tools.delegate_tool_config import _resolve_child_fallback_chain
    from tools.delegate_tool_routes import resolve_task_routes

    chain = [{"provider": "openrouter", "model": "backup-model"}]
    options = [{**OPTIONS[0]}, {**OPTIONS[1], "fallback_providers": []}]
    _, manager = _home(tmp_path, monkeypatch, {"backend": "probe-options", "fallback_providers": chain,
                                               "backends": {"probe-options": {"options": options}}})
    try:
        from tools.delegate_tool_config import _load_config

        routes, err = resolve_task_routes([{"goal": "inherits", "option": "fast"}, {"goal": "own empty chain",
                                                                                     "option": "deep"}],
                                          _load_config(), _parent())
        assert err is None
        inherited, disabled = (_resolve_child_fallback_chain(_parent(), r.routing_cfg, pinned=True) for r in routes)
        assert [(e["provider"], e["model"]) for e in inherited] == [("openrouter", "backup-model")]
        assert disabled is None
    finally:
        manager.unload()


def test_plugin_parameters_never_redefine_core_or_route_fields(tmp_path, monkeypatch):
    from agent.delegation_provider import DelegationProvider
    from tools.delegate_tool_routes import schema_extension

    class Greedy(DelegationProvider):
        @property
        def name(self):
            return "greedy"

        def task_parameters(self, config):
            return {"goal": {"type": "string"}, "model": {"type": "string"}, "api_key": {"type": "string"},
                    "tier": {"type": "string", "enum": ["a", "b"]}}

        def route_task(self, task, config):
            return {}

    from agent.delegation_registry import _registry

    greedy = Greedy()
    _registry.register(greedy)
    try:
        params, _ = schema_extension({"backend": "greedy"})
        assert set(params) == {"tier"}
    finally:
        _registry.restore_registration("greedy", greedy, None)


def test_route_task_always_sees_the_core_task_position(tmp_path, monkeypatch):
    """``index`` is core's: a provider declaring it never lets the model's value replace the real position."""
    from agent.delegation_provider import DelegationProvider
    from agent.delegation_registry import _registry
    from tools.delegate_tool_routes import resolve_task_routes, schema_extension

    seen = []

    class Positional(DelegationProvider):
        @property
        def name(self):
            return "positional"

        def task_parameters(self, config):
            return {"index": {"type": "integer"}}

        def route_task(self, task, config):
            seen.append(task["index"])
            return {}

    positional = Positional()
    _registry.register(positional)
    try:
        assert schema_extension({"backend": "positional"})[0] == {}
        _, err = resolve_task_routes([{"goal": "first", "index": 7}, {"goal": "second", "index": 0}],
                                     {"backend": "positional"}, _parent())
        assert err is None and seen == [0, 1]
    finally:
        _registry.restore_registration("positional", positional, None)


def test_backend_key_reads_through_the_real_loader(tmp_path, monkeypatch):
    """``delegation.backend`` defaults to ``default`` and a configured value reaches the registry."""
    from agent.delegation_registry import configured_backend
    from hermes_cli.config import load_config
    from tools.delegate_tool_config import _load_config

    _, manager = _home(tmp_path / "unset", monkeypatch, {}, plugins=())
    try:
        assert load_config()["delegation"]["backend"] == "default"
        assert configured_backend(_load_config()) == "default"
    finally:
        manager.unload()
    _, manager = _home(tmp_path / "set", monkeypatch, {"backend": " Probe-Options "})
    try:
        assert configured_backend(_load_config()) == "probe-options"
    finally:
        manager.unload()


def test_backend_settings_come_from_its_own_backends_block(tmp_path, monkeypatch):
    """A backend's settings are ``delegation.backends.<its name>``: the same keys beside the core delegation keys
    or under another backend's name are not its settings."""
    _, manager = _home(tmp_path, monkeypatch, {"backend": "probe-options", "options": OPTIONS,
                                               "backends": {"model_map": {"options": OPTIONS}}})
    try:
        assert _task_props(_definition())["option"]["enum"] == []
    finally:
        manager.unload()
