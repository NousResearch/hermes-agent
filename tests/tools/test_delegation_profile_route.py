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
    assert route.reasoning_effort == "low"
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
