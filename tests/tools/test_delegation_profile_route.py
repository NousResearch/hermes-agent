"""A profile route runs the child AS that profile: its config, model, credentials and SOUL, never the
launching profile's — through the same scope binding a multiplexed gateway uses for a routed turn."""

import json
import threading
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest


def _write_profile(home, *, host, model, key, secret, effort, soul=None):
    # A hosted (non-loopback) endpoint: the key must come from the profile's own secrets.
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": f"https://{host}/v1", "default": model, "key_env": "PROFILE_KEY"},
        "agent": {"reasoning_effort": effort},
    }), encoding="utf-8")
    (home / ".env").write_text(f"PROFILE_KEY={key}\nPROBE_SECRET={secret}\n", encoding="utf-8")
    if soul:
        (home / "SOUL.md").write_text(soul, encoding="utf-8")


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    launch = tmp_path / "os-home" / ".hermes"
    _write_profile(launch, host="launch.invalid", model="launch-model", key="launch-key", secret="launch-secret",
                   effort="medium")
    coder = launch / "profiles" / "coder"
    _write_profile(coder, host="coder.invalid", model="coder-model", key="coder-key", secret="coder-secret",
                   effort="low", soul="I am the coder profile.")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    # The launch profile's secrets live in the process env, as after a normal dotenv load.
    monkeypatch.setenv("PROFILE_KEY", "launch-key")
    monkeypatch.setenv("PROBE_SECRET", "launch-secret")
    return launch, coder


def _snapshot():
    from agent.secret_scope import get_secret
    from hermes_cli.config import load_config_readonly
    from hermes_constants import get_hermes_home
    return {"home": str(get_hermes_home()), "secret": get_secret("PROBE_SECRET"),
            "model": load_config_readonly()["model"]["default"]}


def _parent():
    parent = MagicMock()
    parent.base_url, parent.api_key, parent.provider = "https://openrouter.ai/api/v1", "parent-key", "openrouter"
    parent.api_mode, parent.model, parent.platform = "chat_completions", "parent-model", "cli"
    parent.providers_allowed = parent.providers_ignored = parent.providers_order = parent.provider_sort = None
    parent._session_db, parent._delegate_depth, parent._active_children = None, 0, []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = parent.tool_progress_callback = parent.thinking_callback = None
    parent._credential_pool = MagicMock(name="launch-pool")
    return parent


def test_profile_route_resolves_the_target_profiles_model_and_key(two_profiles):
    from tools.delegate_tool_routes import _route_to_task_route

    launch, coder = two_profiles
    route = _route_to_task_route({"profile": "coder", "label": "coder"}, _parent(), {})
    assert route.profile_home == coder and route.label == "coder"
    assert (route.creds["model"], route.creds["base_url"], route.creds["api_key"]) == (
        "coder-model", "https://coder.invalid/v1", "coder-key")
    assert route.reasoning_effort.config == {"enabled": True, "effort": "low"}
    assert _snapshot() == {"home": str(launch), "secret": "launch-secret", "model": "launch-model"}


def test_unknown_profile_is_refused_with_an_instruction(two_profiles):
    from agent.delegation_provider import DelegationRouteError
    from tools.delegate_tool_routes import _route_to_task_route

    with pytest.raises(DelegationRouteError, match="hermes profile list"):
        _route_to_task_route({"profile": "nobody"}, _parent(), {})


def test_child_is_built_and_converses_inside_the_target_profile_scope(two_profiles):
    from tools.delegate_tool import _build_child_agent, _route_overrides
    from tools.delegate_tool_child_run import _ChildRun
    from tools.delegate_tool_routes import _route_to_task_route

    launch, coder = two_profiles
    parent = _parent()
    route = _route_to_task_route({"profile": "coder"}, parent, {})
    seen = {}

    def construct(**kwargs):
        seen["built"] = {**_snapshot(), **{k: kwargs.get(k) for k in ("model", "base_url", "api_key")},
                         "soul": kwargs.get("load_soul_identity")}
        return MagicMock(session_id="child-1")

    with patch("run_agent.AIAgent", side_effect=construct):
        child = _build_child_agent(task_index=0, goal="work as coder", context=None, toolsets=None,
                                   model=route.creds["model"], max_iterations=5, task_count=1,
                                   parent_agent=parent, **_route_overrides(route))
    assert seen["built"] == {"home": str(coder), "secret": "coder-secret", "model": "coder-model",
                             "base_url": "https://coder.invalid/v1", "api_key": "coder-key",
                             "soul": True}
    assert child._credential_pool is not parent._credential_pool  # never the launch profile's pool
    assert _snapshot()["secret"] == "launch-secret"

    class _Child:
        provider, model, _delegate_profile_home = "custom", "coder-model", coder

        def run_conversation(self, user_message, **kw):
            from agent.prompt_builder import load_soul_md
            seen["ran"] = {**_snapshot(), "soul": load_soul_md()}
            return {"final_response": "ok", "completed": True}

    class _Parent:
        _current_task_id = None

    run = _ChildRun(_Child(), _Parent(), 0, "work as coder", None, None)
    with patch("tools.delegate_tool_child_run._create_isolated_worktree", return_value=None):
        run.seed_workspace()
        result, failure, _ = run.await_child()
    assert failure is None and result["final_response"] == "ok"
    assert seen["ran"]["home"] == str(coder) and seen["ran"]["secret"] == "coder-secret"
    assert seen["ran"]["model"] == "coder-model" and "coder profile" in (seen["ran"]["soul"] or "")
    assert _snapshot() == {"home": str(launch), "secret": "launch-secret", "model": "launch-model"}


def test_secret_missing_from_target_profile_never_falls_back_to_the_launch_env(two_profiles):
    from agent.secret_scope import get_secret
    from tools.delegate_tool_routes import delegated_profile_scope

    _, coder = two_profiles
    (coder / ".env").write_text("PROFILE_KEY=coder-key\n", encoding="utf-8")
    with delegated_profile_scope(coder):
        assert get_secret("PROBE_SECRET") is None  # launch value sits in os.environ; never borrowed
    assert get_secret("PROBE_SECRET") == "launch-secret"
    assert json.loads(json.dumps(_snapshot()))["model"] == "launch-model"


def _set_config(home, **sections):
    cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    cfg.update(sections)
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _build(route, parent):
    from tools.delegate_tool import _build_child_agent, _route_overrides
    seen = {}

    def construct(**kwargs):
        seen.update(kwargs)
        return MagicMock(session_id="child-1")

    with patch("run_agent.AIAgent", side_effect=construct):
        child = _build_child_agent(task_index=0, goal="work as coder", context=None, toolsets=None,
                                   model=route.creds["model"], max_iterations=5, task_count=1,
                                   parent_agent=parent, **_route_overrides(route))
    return child, seen


def test_profile_whose_default_is_an_alias_takes_the_alias_route(two_profiles):
    """Same owner as CLI startup: ``model.default: <alias>`` is the alias's model and endpoint."""
    from tools.delegate_tool_routes import _route_to_task_route

    _, coder = two_profiles
    _set_config(coder, model={"default": "localqwen"},
                model_aliases={"localqwen": {"model": "qwen3-27b", "base_url": "http://127.0.0.1:8099/v1"}})
    route = _route_to_task_route({"profile": "coder"}, _parent(), {})
    assert (route.creds["model"], route.creds["base_url"]) == ("qwen3-27b", "http://127.0.0.1:8099/v1")


def test_named_custom_profile_child_leases_its_own_account_on_a_shared_url(two_profiles):
    from tools.delegate_tool_routes import _route_to_task_route

    _, coder = two_profiles
    url = "https://gateway.invalid/v1"
    _set_config(coder, model={"provider": "open-ai", "default": "coder-model"},
                providers={"claude-ai": {"api": url, "api_key": "key-a"}, "open-ai": {"api": url, "api_key": "key-b"}})

    def entry(cid, key):
        return {"id": cid, "label": cid, "auth_type": "api_key", "priority": 0, "source": "manual",
                "access_token": key, "base_url": url}
    (coder / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {
        "claude-ai": [entry("a", "pool-key-a")], "open-ai": [entry("b", "pool-key-b")]}}), encoding="utf-8")

    parent = _parent()
    child, _ = _build(_route_to_task_route({"profile": "coder"}, parent, {}), parent)
    assert [e.access_token for e in child._credential_pool.entries()] == ["pool-key-b"]


@pytest.mark.parametrize("agent_cfg, expected", [
    ({"reasoning_effort": "high", "reasoning_overrides": {"coder-model": False}}, {"enabled": False}),
    ({}, None),  # unset in the target profile: its default, never the launching parent's level
])
def test_profile_child_reasoning_is_the_target_profiles_resolved_setting(two_profiles, agent_cfg, expected):
    from tools.delegate_tool_routes import _route_to_task_route

    _, coder = two_profiles
    _set_config(coder, agent=agent_cfg)
    parent = _parent()
    parent.reasoning_config = {"enabled": True, "effort": "xhigh"}
    with patch("tools.delegate_tool._load_config", return_value={"reasoning_effort": "xhigh"}):
        _, built = _build(_route_to_task_route({"profile": "coder"}, parent, {}), parent)
    assert built["reasoning_config"] == expected


def test_profile_child_lease_and_schema_retry_run_inside_the_target_profile(two_profiles):
    """The startup lease (``_swap_credential`` reloads TLS/headers from config) and the output_schema retry turn
    are child work: they run as the routed profile, like the first conversation."""
    from types import SimpleNamespace

    from tools.delegate_tool_child_run import _lease_child_credential, _validate_child_output_schema

    launch, coder = two_profiles
    seen = {}

    class _Pool:
        def acquire_lease(self, credential_id=None):
            return "e1"

        def entries(self):
            return [SimpleNamespace(id="e1", base_url="https://coder.invalid/v1", last_status=None)]

    class _Child:
        base_url, session_id, _delegate_profile_home = "https://coder.invalid/v1", "child-1", coder
        _credential_pool, _delegate_output_schema = _Pool(), {"type": "object", "required": ["ok"]}

        def _swap_credential(self, entry):
            seen["lease"] = _snapshot()
            return True

        def run_conversation(self, user_message, **kw):
            seen["retry"] = _snapshot()
            return {"final_response": '{"ok": true}', "api_calls": 1}

    child = _Child()
    _lease_child_credential(child)
    outcome = _validate_child_output_schema(child, {"final_response": "not json"}, 0, "t-1", None)
    assert outcome.valid and outcome.retries == 1
    assert seen["lease"]["home"] == str(coder) and seen["lease"]["model"] == "coder-model"
    assert seen["retry"] == {"home": str(coder), "secret": "coder-secret", "model": "coder-model"}
    assert _snapshot() == {"home": str(launch), "secret": "launch-secret", "model": "launch-model"}


@pytest.mark.parametrize("failing", ["agent.secret_scope.set_secret_scope",
                                     "tools.terminal_scope.install_profile_terminal_scope"])
def test_profile_scope_that_fails_to_enter_leaves_the_parent_in_its_own_profile(two_profiles, failing):
    """A step of the scope binding that raises unwinds the steps already bound: the parent never stays
    inside the child's profile (home override, secret scope, multiplex flag)."""
    from agent.secret_scope import current_secret_scope_home, is_multiplex_active
    from tools.delegate_tool_routes import delegated_profile_scope

    _, coder = two_profiles
    before = (_snapshot(), current_secret_scope_home(), is_multiplex_active())
    with patch(failing, side_effect=RuntimeError("scope step failed")), pytest.raises(RuntimeError):
        with delegated_profile_scope(coder):
            pass
    assert (_snapshot(), current_secret_scope_home(), is_multiplex_active()) == before


class _ProfileProvider:
    """A backend that routes every task to the ``coder`` profile."""
    name = "profile_routes"

    def task_parameters(self, config):
        return {}

    def route_task(self, task, config):
        return {"profile": "coder"}


def _resolve_coder_task():
    from tools.delegate_tool_routes import resolve_task_routes
    with patch("tools.delegate_tool_routes._active_provider", return_value=_ProfileProvider()):
        return resolve_task_routes([{"goal": "work as coder"}], {}, _parent())


def test_profile_route_is_refused_under_a_managed_worker_config_snapshot(two_profiles, monkeypatch):
    """A managed worker's config readers all return its frozen session config, so the target profile's
    config.yaml is unreachable there: refuse instead of pairing the launch model with the target's keys."""
    import os

    from agent import safe_worker_policy
    from agent.delegation_provider import DelegationRouteError
    from tools.delegate_tool_routes import _route_to_task_route

    launch, _ = two_profiles
    frozen = yaml.safe_load((launch / "config.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(safe_worker_policy, "_policy", safe_worker_policy._SafeWorkerPolicy(
        os.getpid(), False, False, json.dumps(frozen)))

    (route,), error = _resolve_coder_task()
    assert error is None and "managed worker" in (route.error or "")
    with pytest.raises(DelegationRouteError):
        _route_to_task_route({"profile": "coder"}, _parent(), {})


def test_profile_pinned_to_a_missing_acp_command_is_refused_at_routing(two_profiles, monkeypatch):
    """Same rule as ``delegation.provider`` (#80450): a profile whose provider is pinned to an ACP binary
    that is not on PATH refuses the spawn, naming the profile, instead of building a broken child."""
    _, coder = two_profiles
    _set_config(coder, model={"provider": "copilot-acp", "default": "coder-model"})
    # An acp+tcp endpoint keeps the configured command unresolved (the runtime resolver only refuses a missing
    # binary when it would spawn one); the delegation pin rule still refuses it.
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "hermes-test-missing-acp-binary")
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp+tcp://127.0.0.1:9")

    (route,), error = _resolve_coder_task()
    assert error is None and route.creds == {}
    assert "hermes-test-missing-acp-binary" in (route.error or "") and "coder" in route.error


def test_one_profile_under_two_labels_resolves_once_and_keeps_each_label(two_profiles):
    from hermes_cli import runtime_provider
    from tools.delegate_tool_routes import _route_to_task_route

    cache, parent = {}, _parent()
    with patch.object(runtime_provider, "resolve_runtime_provider", wraps=runtime_provider.resolve_runtime_provider) as rp:
        first = _route_to_task_route({"profile": "coder", "label": "reviewer"}, parent, cache)
        second = _route_to_task_route({"profile": "coder", "label": "fixer"}, parent, cache)
    assert rp.call_count == 1
    assert (first.label, second.label) == ("reviewer", "fixer")
    assert first.creds == second.creds and first.creds["model"] == "coder-model"


@pytest.mark.parametrize("worktree", [None, {"path": "/work/wt-1", "branch": "subagent/wt-1"}])
def test_profile_child_reads_the_starting_directory_seeded_for_it(two_profiles, worktree):
    """cwd records are keyed by the active profile on a multiplexed host: the child's starting directory (the
    parent's, or its isolated worktree) is recorded where the child, running as the target profile, reads it."""
    from tools.delegate_tool_child_run import _ChildRun
    from tools.delegate_tool_routes import child_profile_scope
    from tools.terminal_tool import clear_session_cwd, get_session_cwd, record_session_cwd

    _, coder = two_profiles

    class _Child:
        _delegate_profile_home = coder

    class _Parent:
        _current_task_id = "parent-task"

    record_session_cwd("parent-task", "/work/repo")
    run = _ChildRun(_Child(), _Parent(), 0, "work as coder", None, None)
    try:
        with patch("tools.delegate_tool_child_run._create_isolated_worktree", return_value=worktree):
            run.seed_workspace()
        with child_profile_scope(run.child):
            assert get_session_cwd(run.child_task_id) == (worktree or {}).get("path", "/work/repo")
    finally:
        clear_session_cwd("parent-task")


def test_profile_child_uses_the_target_profiles_persistent_container(two_profiles):
    """Persistent Docker keys its container by profile: a child routed to ``coder`` inside a launch-profile
    session runs in coder's container (coder's image, mounts, env), while the session's delivery identity
    (HERMES_SESSION_PROFILE, read by notifications) stays the parent's."""
    from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
    from tools import terminal_tool
    from tools.delegate_tool_routes import delegated_profile_scope

    _, coder = two_profiles
    _set_config(coder, terminal={"backend": "docker", "container_persistent": True})
    tokens = set_session_vars(session_key="agent:main:telegram:dm:1", profile="default")
    try:
        with delegated_profile_scope(coder):
            assert terminal_tool._resolve_container_task_id("child-task") == "profile:coder"
            assert get_session_env("HERMES_SESSION_PROFILE", "") == "default"
    finally:
        clear_session_vars(tokens)


def test_child_routed_back_to_the_launch_profile_uses_the_launch_profiles_container(two_profiles):
    """A ``coder`` chat on a default-launched gateway (its turn bound to coder, HERMES_SESSION_PROFILE=coder) routing
    a child to ``default`` runs that child in default's persistent container, not the parent chat's."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import terminal_tool
    from tools.delegate_tool_routes import delegated_profile_scope

    launch, coder = two_profiles
    for home in (launch, coder):
        _set_config(home, terminal={"backend": "docker", "container_persistent": True})
    tokens = set_session_vars(session_key="agent:coder:telegram:dm:1", profile="coder")
    try:
        with delegated_profile_scope(coder):
            assert terminal_tool._resolve_container_task_id("parent-task") == "profile:coder"
            with delegated_profile_scope(launch):
                assert terminal_tool._resolve_container_task_id("child-task") == "default"
            assert terminal_tool._resolve_container_task_id("parent-task") == "profile:coder"
    finally:
        clear_session_vars(tokens)
