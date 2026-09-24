"""Subagent endpoint-trust capabilities (follow-up to #94036/#97292/#105347).

The capability map is endpoint-scoped trust, so it follows the route the child
actually calls: an unpinned child runs the parent's exact route and inherits its
map; a pinned one carries the map ITS OWN route declares and never the parent's.
Dropping a pinned route's map leaves the child on a different wire than its
provider declares (an ``anthropic_oauth_proxy`` child loses the OAuth identity
transform and upstream answers HTTP 429).

``delegation.model`` is such a pin in its own right: the same endpoint may declare
different capabilities per model, so a child pinned only to model B must carry
B's map rather than the parent model's.
"""

from types import SimpleNamespace

import pytest
import yaml

from tools.delegate_tool import _child_route_capabilities
from tools.delegate_tool_config import _resolve_child_runtime, _runtime_provider_credentials

TRUSTED = {"anthropic_oauth_proxy": True}
UNTRUSTED = {"anthropic_oauth_proxy": False}
_PARENT_DEFAULT = object()

RELAY_URL = "https://proxy.example:3443"
PARENT_MODEL = "claude-opus-5"
# On the shared relay entry below: the provider level trusts, this model opts itself out.
DENYING_MODEL = "claude-haiku-4-6"


def _parent(capabilities=_PARENT_DEFAULT, model=PARENT_MODEL, provider="custom"):
    return SimpleNamespace(
        provider=provider,
        requested_provider=provider,
        base_url=RELAY_URL,
        model=model,
        api_mode="anthropic_messages",
        capabilities=TRUSTED if capabilities is _PARENT_DEFAULT else capabilities,
        request_overrides={},
        reasoning_config=None,
        acp_command=None,
        acp_args=[],
    )


@pytest.fixture
def relay_config(tmp_path, monkeypatch):
    """One relay whose provider level trusts and whose ``DENYING_MODEL`` declares itself off."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY_KEY", "opaque-relay-key")
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {"provider": "custom:relay", "default": PARENT_MODEL},
            "providers": {
                "relay": {
                    "api": RELAY_URL,
                    "key_env": "TEST_RELAY_KEY",
                    "transport": "anthropic_messages",
                    "capabilities": {"anthropic_oauth_proxy": True},
                    "models": {DENYING_MODEL: {"anthropic_oauth_proxy": False}},
                },
                "inverse": {
                    "api": RELAY_URL,
                    "key_env": "TEST_RELAY_KEY",
                    "transport": "anthropic_messages",
                    "capabilities": {"anthropic_oauth_proxy": False},
                    "models": {DENYING_MODEL: {"anthropic_oauth_proxy": True}},
                },
            },
        }),
        encoding="utf-8",
    )
    return tmp_path


# ── the parent's route ───────────────────────────────────────────────────────

def test_unpinned_child_inherits_the_parents_map():
    assert _child_route_capabilities(_parent(), None, None, None) == TRUSTED


def test_inherited_map_is_sanitized_to_str_bool():
    parent = _parent({"anthropic_oauth_proxy": True, "bad": "yes", 3: True, "n": 0})
    assert _child_route_capabilities(parent, None, None, None) == TRUSTED


def test_non_dict_parent_capabilities_yield_nothing():
    assert not _child_route_capabilities(_parent(None), None, None, None)


def test_a_child_on_the_parents_own_model_still_inherits():
    """Same model = the parent's exact route, not a pin."""
    assert _child_route_capabilities(
        _parent(), None, None, None, effective_model=PARENT_MODEL,
    ) == TRUSTED


# ── a pinned route ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("provider,base_url", [("openai", None), (None, "https://other.example/v1")])
def test_pin_never_borrows_the_parents_trust(provider, base_url):
    assert not _child_route_capabilities(_parent(), provider, base_url, None)


def test_pinned_route_carries_its_own_declared_map():
    assert _child_route_capabilities(_parent({}), "teamclaude", None, TRUSTED) == TRUSTED


# ── a model-only pin on one endpoint (#105347) ───────────────────────────────

def test_model_only_pin_does_not_inherit_a_denying_models_route(relay_config):
    """provider-level ``true``, model-B ``false``: B must not borrow the parent model's trust."""
    assert _child_route_capabilities(
        _parent(), None, None, None,
        effective_provider="custom:relay", effective_model=DENYING_MODEL,
    ) == UNTRUSTED


def test_model_only_pin_does_not_lose_its_own_declared_trust(relay_config):
    """The inverse: provider-level ``false``, model-B ``true`` — B keeps its own opt-in."""
    parent = _parent(UNTRUSTED, model=PARENT_MODEL)
    assert _child_route_capabilities(
        parent, None, None, None,
        effective_provider="custom:inverse", effective_model=DENYING_MODEL,
    ) == TRUSTED


def test_a_model_only_pin_declaring_nothing_gets_nothing(relay_config):
    """An unknown route cannot fall back to the parent's map — default-deny holds."""
    assert not _child_route_capabilities(
        _parent(), None, None, None,
        effective_provider="custom:absent", effective_model=DENYING_MODEL,
    )


def test_explicitly_declared_child_capabilities_win_over_the_config_lookup(relay_config):
    """A resolved bundle that already carries the child's map is authoritative."""
    assert _child_route_capabilities(
        _parent(), None, None, TRUSTED,
        effective_provider="custom:relay", effective_model=DENYING_MODEL,
    ) == TRUSTED


# ── resolution end to end ────────────────────────────────────────────────────

def test_runtime_provider_credentials_carry_declared_capabilities(monkeypatch):
    """The ``delegation.provider`` branch keeps the provider's capability map."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "custom", "api_mode": "anthropic_messages", "api_key": "sk-test",
            "base_url": RELAY_URL, "model": PARENT_MODEL, "capabilities": TRUSTED,
        },
    )
    creds = _runtime_provider_credentials(
        {"provider": "teamclaude", "model": PARENT_MODEL, "api_mode": None}, None,
    )
    assert creds["capabilities"] == TRUSTED


def test_pinned_route_capabilities_reach_the_child_kwargs():
    rt = _resolve_child_runtime(
        _parent({}), {}, "sk-parent", model=None, override_provider="teamclaude",
        override_base_url=RELAY_URL, override_api_key="sk-child",
        override_api_mode="anthropic_messages", override_acp_command=None, override_acp_args=None,
        override_capabilities=TRUSTED,
    )
    assert rt["capabilities"] == TRUSTED


@pytest.mark.parametrize(
    "provider,parent_capabilities,expected",
    [
        ("custom:relay", TRUSTED, UNTRUSTED),    # true → false
        ("custom:inverse", UNTRUSTED, TRUSTED),  # false → true
    ],
)
def test_model_only_delegation_reaches_the_child_kwargs(
    relay_config, provider, parent_capabilities, expected
):
    """``delegation.model`` alone: the child's kwargs carry ITS model's declared map."""
    rt = _resolve_child_runtime(
        _parent(parent_capabilities, provider=provider), {}, "sk-parent", model=DENYING_MODEL,
        override_provider=None, override_base_url=None, override_api_key=None,
        override_api_mode=None, override_acp_command=None, override_acp_args=None,
        override_capabilities=None,
    )
    assert rt["model"] == DENYING_MODEL
    assert rt["capabilities"] == expected

