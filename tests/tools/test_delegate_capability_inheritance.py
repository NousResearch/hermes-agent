"""Subagent endpoint-trust capabilities (follow-up to #94036/#97292).

The capability map is endpoint-scoped trust, so it follows the route the child
actually calls: an unpinned child runs the parent's exact route and inherits its
map; a pinned one carries the map ITS OWN route declares and never the parent's.
Dropping a pinned route's map leaves the child on a different wire than its
provider declares (an ``anthropic_oauth_proxy`` child loses the OAuth identity
transform and upstream answers HTTP 429).
"""

from types import SimpleNamespace

import pytest

from tools.delegate_tool import _child_route_capabilities
from tools.delegate_tool_config import _resolve_child_runtime, _runtime_provider_credentials

TRUSTED = {"anthropic_oauth_proxy": True}
_PARENT_DEFAULT = object()


def _parent(capabilities=_PARENT_DEFAULT):
    return SimpleNamespace(
        provider="custom",
        base_url="https://proxy.example:3443",
        model="claude-opus-5",
        api_mode="anthropic_messages",
        capabilities=TRUSTED if capabilities is _PARENT_DEFAULT else capabilities,
        request_overrides={},
        reasoning_config=None,
        acp_command=None,
        acp_args=[],
    )


# ── the parent's route ───────────────────────────────────────────────────────

def test_unpinned_child_inherits_the_parents_map():
    assert _child_route_capabilities(_parent(), None, None, None) == TRUSTED


def test_inherited_map_is_sanitized_to_str_bool():
    parent = _parent({"anthropic_oauth_proxy": True, "bad": "yes", 3: True, "n": 0})
    assert _child_route_capabilities(parent, None, None, None) == TRUSTED


def test_non_dict_parent_capabilities_yield_nothing():
    assert not _child_route_capabilities(_parent(None), None, None, None)


# ── a pinned route ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("provider,base_url", [("openai", None), (None, "https://other.example/v1")])
def test_pin_never_borrows_the_parents_trust(provider, base_url):
    assert not _child_route_capabilities(_parent(), provider, base_url, None)


def test_pinned_route_carries_its_own_declared_map():
    assert _child_route_capabilities(_parent({}), "teamclaude", None, TRUSTED) == TRUSTED


# ── resolution end to end ────────────────────────────────────────────────────

def test_runtime_provider_credentials_carry_declared_capabilities(monkeypatch):
    """The ``delegation.provider`` branch keeps the provider's capability map."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {
            "provider": "custom", "api_mode": "anthropic_messages", "api_key": "sk-test",
            "base_url": "https://proxy.example:3443", "model": "claude-opus-5", "capabilities": TRUSTED,
        },
    )
    creds = _runtime_provider_credentials(
        {"provider": "teamclaude", "model": "claude-opus-5", "api_mode": None}, None,
    )
    assert creds["capabilities"] == TRUSTED


def test_pinned_route_capabilities_reach_the_child_kwargs():
    rt = _resolve_child_runtime(
        _parent({}), {}, "sk-parent", model=None, override_provider="teamclaude",
        override_base_url="https://proxy.example:3443", override_api_key="sk-child",
        override_api_mode="anthropic_messages", override_acp_command=None, override_acp_args=None,
        override_capabilities=TRUSTED,
    )
    assert rt["capabilities"] == TRUSTED
