"""Resolution-time fallback for delegation routes (Fixes #146645).

A route whose primary provider refuses at credential resolution with an auth or quota error (an Anthropic pool
in 429 cooldown, a Codex pool at its usage limit) runs the task on the first entry of the route's own fallback
chain that resolves, instead of refusing the task. Misconfiguration still refuses with today's message.

The resolver below is the only fake: route selection, the credential bundle, the fallback-chain rules and child
construction are the real code paths.
"""

import logging
import types
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.auth import AuthError

PRIMARY = {"provider": "anthropic", "model": "claude-opus-5-5"}
BEDROCK = {"provider": "bedrock", "model": "global.anthropic.claude-opus-5-5"}
OPENROUTER = {"provider": "openrouter", "model": "backup-model"}
XAI = {"provider": "xai-oauth", "model": "grok-4.7"}

RUNTIMES = {
    "anthropic": {"provider": "anthropic", "base_url": "https://api.anthropic.com", "api_key": "sk-ant",
                  "api_mode": "anthropic_messages"},
    "bedrock": {"provider": "bedrock", "base_url": "https://bedrock-runtime.us-west-2.amazonaws.com",
                "api_key": "aws-sdk", "api_mode": "anthropic_messages"},
    "openrouter": {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk-or",
                   "api_mode": "chat_completions"},
    "xai-oauth": {"provider": "xai-oauth", "base_url": "https://api.x.ai/v1", "api_key": "xai-token",
                  "api_mode": "codex_responses"},
}


def _exhausted_anthropic(model):
    """The real error an Anthropic pool raises when its only entry is benched for this model."""
    from hermes_cli.runtime_provider import _anthropic_token_or_raise
    with patch("agent.anthropic_credentials.resolve_anthropic_token",
               lambda model=None: None if model else "sk-ant-other-model"):
        _anthropic_token_or_raise(model=model)


def _codex_quota_exhausted(model):
    """The real error a Codex pool raises while its weekly reset is in the future."""
    from hermes_cli.auth_codex import _codex_quota_exhausted_error
    raise _codex_quota_exhausted_error(295723)


@pytest.fixture
def resolver(monkeypatch):
    """Fake ``resolve_runtime_provider``: ``resolver.failing[provider]`` is a callable(model) that raises;
    unknown providers go to the real resolver. ``resolver.calls`` records ``(provider, model)`` and
    ``resolver.explicit`` the other kwargs of each call."""
    import hermes_cli.runtime_provider as rp

    real = rp.resolve_runtime_provider
    state = types.SimpleNamespace(calls=[], failing={}, explicit=[])

    def resolve(requested=None, target_model=None, **kwargs):
        state.calls.append((requested, target_model))
        state.explicit.append(kwargs)
        if requested in state.failing:
            state.failing[requested](target_model)
        if requested not in RUNTIMES:
            return real(requested=requested, target_model=target_model, **kwargs)
        return {**RUNTIMES[requested], "model": target_model}

    monkeypatch.setattr(rp, "resolve_runtime_provider", resolve)
    return state


def _parent(chain=None):
    return types.SimpleNamespace(session_id="root", model="parent-model", provider="openrouter",
                                 _fallback_chain=chain, request_overrides={}, tool_progress_callback=None)


class _Options:
    """A ``model_map``-style backend: each task names an option; no option is the plain delegation section."""
    name = "probe-options"

    def __init__(self, options):
        self.options = options

    def task_parameters(self, config):
        return {"option": {"type": "string", "enum": list(self.options)}}

    def route_task(self, task, config):
        option = self.options.get(task.get("option"))
        return dict(option, label=task["option"]) if option else dict(config)


def _routes(tasks, config, provider=None, parent=None):
    from plugins.delegation.default import DefaultDelegationProvider
    from tools.delegate_tool_routes import resolve_task_routes
    with patch("tools.delegate_tool_routes._active_provider", return_value=provider or DefaultDelegationProvider()):
        routes, error = resolve_task_routes(tasks, config, parent or _parent())
    assert error is None
    return routes


def _fallback_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "Falling back to" in r.getMessage()]


def test_default_route_resolve_fallback_on_exhausted_anthropic_pool(resolver, caplog):
    """The reported case: delegation.provider anthropic in 429 cooldown, delegation.fallback_providers bedrock."""
    resolver.failing["anthropic"] = _exhausted_anthropic
    config = {**PRIMARY, "fallback_providers": [BEDROCK], "request_overrides": {"extra_body": {"tag": "review"}}}
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        (route,) = _routes([{"goal": "review the diff"}], config)
    assert route.error is None
    assert (route.creds["provider"], route.creds["model"]) == ("bedrock", "global.anthropic.claude-opus-5-5")
    assert route.creds["base_url"] == RUNTIMES["bedrock"]["base_url"]
    assert route.creds["request_overrides"] == {"extra_body": {"tag": "review"}}
    assert resolver.calls == [("anthropic", "claude-opus-5-5"), ("bedrock", "global.anthropic.claude-opus-5-5")]
    (warning,) = _fallback_warnings(caplog)
    assert warning.startswith("Primary provider auth failed (Anthropic credentials are rate-limited for "
                              "claude-opus-5-5")
    assert warning.endswith("Falling back to bedrock/global.anthropic.claude-opus-5-5")


def test_model_map_option_resolve_fallback_walks_its_own_chain(resolver, caplog):
    """#146645: a Codex option at its weekly limit tries its own chain in order; delegation.fallback_providers
    is not consulted because the option declares a chain of its own."""
    resolver.failing["openai-codex"] = _codex_quota_exhausted
    quick = {"provider": "openai-codex", "model": "codex-a",
             "fallback_providers": [{"provider": "openai-codex", "model": "codex-b"}, XAI]}
    config = {"fallback_providers": [BEDROCK]}
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        (route,) = _routes([{"goal": "quick look", "option": "quick"}], config, _Options({"quick": quick}))
    assert route.error is None and route.label == "quick"
    assert (route.creds["provider"], route.creds["model"]) == ("xai-oauth", "grok-4.7")
    assert resolver.calls == [("openai-codex", "codex-a"), ("openai-codex", "codex-b"), ("xai-oauth", "grok-4.7")]
    (warning,) = _fallback_warnings(caplog)
    assert warning.startswith("Primary provider rate-limited (429) (Codex provider quota exhausted (429)")
    assert warning.endswith("Falling back to xai-oauth/grok-4.7")


def test_pinned_route_resolve_fallback_ignores_the_parent_chain(resolver):
    """A route naming its provider never borrows the parent session's chain (same rule as the built child)."""
    resolver.failing["anthropic"] = _exhausted_anthropic
    (route,) = _routes([{"goal": "g"}], dict(PRIMARY), parent=_parent(chain=[dict(OPENROUTER)]))
    assert route.error.startswith("Cannot resolve delegation provider 'anthropic': Anthropic credentials are "
                                  "rate-limited for claude-opus-5-5")
    assert "fallback" not in route.error.lower()
    assert resolver.calls == [("anthropic", "claude-opus-5-5")]


def test_explicit_empty_chain_resolve_fallback_still_refuses(resolver):
    """``fallback_providers: []`` on an option disables fallback even with a delegation-level chain."""
    resolver.failing["anthropic"] = _exhausted_anthropic
    option = {**PRIMARY, "fallback_providers": []}
    (route,) = _routes([{"goal": "g", "option": "solo"}], {"fallback_providers": [BEDROCK]},
                       _Options({"solo": option}))
    assert route.error.startswith("Cannot resolve delegation provider 'anthropic': Anthropic credentials")
    assert resolver.calls == [("anthropic", "claude-opus-5-5")]


def _missing_cli(model):
    raise AuthError("Could not find the 'acme-acp' CLI command 'acmecli'. Install it.", provider="acme-acp",
                    code="missing_external_process_cli")


def _missing_key(model):
    raise AuthError("No usable credentials found for provider 'anthropic'.", provider="anthropic",
                    code="missing_api_key")


@pytest.mark.parametrize("primary, failing, expected", [
    pytest.param({"provider": "anthropc", "model": "claude-opus-5-5"}, {},
                 "Cannot resolve delegation provider 'anthropc': Unknown provider 'anthropc'.", id="unknown-provider"),
    pytest.param(dict(PRIMARY), {"anthropic": _missing_key},
                 "Cannot resolve delegation provider 'anthropic': No usable credentials found", id="missing-key"),
    pytest.param({"provider": "ollama", "model": "llama4"}, {},
                 "Cannot resolve delegation provider 'ollama': provider 'ollama' has no endpoint configured",
                 id="no-endpoint"),
    pytest.param({"provider": "acme-acp", "model": "acme-1"}, {"acme-acp": _missing_cli},
                 "Cannot resolve delegation provider 'acme-acp': Could not find the 'acme-acp' CLI", id="missing-cli"),
])
def test_misconfigured_primary_resolve_fallback_refuses_with_todays_message(resolver, primary, failing, expected):
    """A typo or a missing key is configuration, not a credential state: the task refuses and names it."""
    from tools.delegate_tool_config import _resolve_delegation_credentials
    resolver.failing.update(failing)
    with pytest.raises(ValueError) as direct:
        _resolve_delegation_credentials(primary, _parent())  # today's message, without the walk
    resolver.calls.clear()
    (route,) = _routes([{"goal": "g"}], {**primary, "fallback_providers": [BEDROCK]})
    assert route.error == str(direct.value) and route.error.startswith(expected)
    assert [p for p, _ in resolver.calls] == [primary["provider"]]


def test_resolved_without_key_resolve_fallback_refuses_with_todays_message(resolver, monkeypatch):
    monkeypatch.setitem(RUNTIMES, "anthropic", {**RUNTIMES["anthropic"], "api_key": ""})
    (route,) = _routes([{"goal": "g"}], {**PRIMARY, "fallback_providers": [BEDROCK]})
    assert route.error == ("Delegation provider 'anthropic' resolved but has no API key. "
                           "Set the appropriate environment variable or run 'hermes auth'.")
    assert resolver.calls == [("anthropic", "claude-opus-5-5")]


def _raises(exc):
    def fail(model):
        raise exc
    return fail


def test_unreachable_primary_resolve_fallback_walks_the_chain(resolver, caplog):
    """A transient network failure at resolution walks the chain like an auth error."""
    import httpx
    resolver.failing["anthropic"] = _raises(httpx.ConnectError("[Errno 111] Connection refused"))
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        (route,) = _routes([{"goal": "g"}], {**PRIMARY, "fallback_providers": [BEDROCK]})
    assert route.error is None and route.creds["provider"] == "bedrock"
    assert _fallback_warnings(caplog) == ["Primary provider unreachable ([Errno 111] Connection refused). "
                                          "Falling back to bedrock/global.anthropic.claude-opus-5-5"]


RELAY = {"provider": "openrouter", "model": "backup-model", "base_url": "https://relay.example/v1",
         "key_env": "RELAY_KEY"}


@pytest.mark.parametrize("via_profile", [False, True], ids=["route", "profile"])
def test_entry_endpoint_and_key_resolve_fallback_reach_the_resolver(resolver, monkeypatch, via_profile):
    """A chain entry resolves with its own base_url and key, never the provider's default endpoint and key."""
    from hermes_constants import get_hermes_home
    from tools.delegate_tool_routes import _profile_route
    resolver.failing["anthropic"] = _exhausted_anthropic
    monkeypatch.setenv("RELAY_KEY", "relay-key")
    if via_profile:
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"fallback_providers": [RELAY]})
        monkeypatch.setattr("tools.delegate_tool_routes._profile_model_request",
                            lambda cfg: {"target_model": "claude-opus-5-5", "requested": "anthropic"})
        route = _profile_route(get_hermes_home())
    else:
        (route,) = _routes([{"goal": "g"}], {**PRIMARY, "fallback_providers": [RELAY]})
    assert route.creds["provider"] == "openrouter"
    assert resolver.explicit[-1] == {"explicit_base_url": "https://relay.example/v1", "explicit_api_key": "relay-key"}


def test_all_fail_resolve_fallback_error_names_every_failure(resolver, caplog):
    resolver.failing["anthropic"] = _exhausted_anthropic
    resolver.failing["bedrock"] = _raises(AuthError("No AWS credentials found."))
    resolver.failing["xai-oauth"] = _raises(AuthError("xAI OAuth token expired."))
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        (route,) = _routes([{"goal": "g"}], {**PRIMARY, "fallback_providers": [BEDROCK, XAI]})
    assert route.error.startswith("Cannot resolve delegation provider 'anthropic': Anthropic credentials are "
                                  "rate-limited for claude-opus-5-5")
    assert "bedrock/global.anthropic.claude-opus-5-5: No AWS credentials found." in route.error
    assert "xai-oauth/grok-4.7: xAI OAuth token expired." in route.error
    assert _fallback_warnings(caplog) == []


def test_built_child_mid_run_chain_resolve_fallback_excludes_the_new_primary(resolver):
    """The entry that became the child's primary is not also its first mid-run fallback; the rest stay."""
    from tests.tools.test_delegate import _make_mock_parent
    from tools.delegate_tool import _build_child_agent, _route_overrides

    resolver.failing["anthropic"] = _exhausted_anthropic
    (route,) = _routes([{"goal": "g"}], {**PRIMARY, "fallback_providers": [BEDROCK, OPENROUTER]})
    parent = _make_mock_parent(depth=0)
    parent._fallback_chain = [{"provider": "nous", "model": "parent-chain-model"}]
    with patch("tools.delegate_tool._load_config", return_value={}), patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value = MagicMock()
        _build_child_agent(task_index=0, goal="g", context=None, toolsets=None, model=route.creds["model"],
                           max_iterations=5, parent_agent=parent, task_count=1, **_route_overrides(route))
    kwargs = agent_cls.call_args.kwargs
    assert (kwargs["provider"], kwargs["model"]) == ("bedrock", "global.anthropic.claude-opus-5-5")
    assert [(e["provider"], e["model"]) for e in kwargs["fallback_model"]] == [("openrouter", "backup-model")]


def test_last_entry_resolve_fallback_leaves_no_mid_run_chain(resolver):
    resolver.failing["anthropic"] = _exhausted_anthropic
    (route,) = _routes([{"goal": "g"}], {**PRIMARY, "fallback_providers": [BEDROCK]})
    from tools.delegate_tool_config import _resolve_child_fallback_chain
    assert route.error is None and route.creds["provider"] == "bedrock"
    assert route.routing_cfg["fallback_providers"] == []
    assert _resolve_child_fallback_chain(_parent(), route.routing_cfg, pinned=True) is None


def test_healthy_primary_resolve_fallback_changes_nothing(resolver, caplog):
    config = {**PRIMARY, "fallback_providers": [BEDROCK]}
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        (route,) = _routes([{"goal": "g"}], config)
    assert route.error is None and route.creds["provider"] == "anthropic"
    assert route.routing_cfg["fallback_providers"] == [BEDROCK]
    assert resolver.calls == [("anthropic", "claude-opus-5-5")]
    assert _fallback_warnings(caplog) == []


def test_internal_caller_route_resolve_fallback(resolver):
    """``route_for_config`` (``/review`` -> ``auxiliary.review``) walks its block's chain the same way."""
    from tools.delegate_tool_routes import route_for_config
    resolver.failing["anthropic"] = _exhausted_anthropic
    route = route_for_config({**PRIMARY, "fallback_providers": [BEDROCK]}, _parent())
    assert (route.creds["provider"], route.creds["model"]) == ("bedrock", "global.anthropic.claude-opus-5-5")
    assert route.routing_cfg["fallback_providers"] == []


def test_profile_route_resolve_fallback_walks_the_profiles_chain(resolver, monkeypatch, caplog):
    """A profile route resolves inside the profile and walks that profile's own ``fallback_providers``."""
    from hermes_constants import get_hermes_home
    from tools.delegate_tool_config import ResolvedReasoning
    from tools.delegate_tool_routes import _profile_route

    resolver.failing["anthropic"] = _exhausted_anthropic
    cfg = {"model": {"default": "claude-opus-5-5", "provider": "anthropic"},
           "fallback_providers": [BEDROCK, OPENROUTER]}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    # The profile's model block decodes to this request through resolve_startup_model_route (not under test).
    monkeypatch.setattr("tools.delegate_tool_routes._profile_model_request",
                        lambda cfg: {"target_model": "claude-opus-5-5", "requested": "anthropic"})
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        route = _profile_route(get_hermes_home())
    assert (route.creds["provider"], route.creds["model"]) == ("bedrock", "global.anthropic.claude-opus-5-5")
    assert [(e["provider"], e["model"]) for e in route.routing_cfg["fallback_providers"]] == [
        ("openrouter", "backup-model")]
    assert isinstance(route.reasoning_effort, ResolvedReasoning)
    assert len(_fallback_warnings(caplog)) == 1
