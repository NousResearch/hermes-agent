"""Regression tests for the delegation.request_overrides config key.

PR #90953 (salvage): ``delegation.request_overrides`` is an explicit dict of
per-child request settings that must be honored on EVERY resolution branch of
``_resolve_delegation_credentials``:

1. direct base_url (provider=custom) — the branch the original PR fixed,
2. named provider (delegation.provider set, no base_url),
3. parent-inherit (neither provider nor base_url).

Precedence contract (post-#98237): explicit config values merge OVER
runtime/parent-derived overrides — top-level explicit keys win; the
``extra_body`` sub-dict is deep-merged one level so runtime extra_body keys
survive unless the explicit key redefines them. Nested values are deep-copied
so transport-side mutation cannot leak back into config.
"""

from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _merge_request_overrides,
    _resolve_delegation_credentials,
)


def _cfg(**overrides):
    cfg = {
        "model": "deepseek/deepseek-v4-flash-0731",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key-1234567890",
    }
    cfg.update(overrides)
    return cfg


def _parent(**attrs):
    parent = MagicMock()
    parent._delegate_depth = 0
    # MagicMock attributes are MagicMocks (non-dict) by default; set real
    # values for the ones the resolution path inspects.
    parent.request_overrides = attrs.pop("request_overrides", None)
    for k, v in attrs.items():
        setattr(parent, k, v)
    return parent


# ── Branch 1: direct base_url ──────────────────────────────────────────────


def test_direct_branch_forwards_request_overrides():
    """delegation.request_overrides flows through the direct-endpoint branch."""
    cfg = _cfg(
        request_overrides={
            "extra_body": {"provider": {"sort": "throughput"}},
        }
    )
    creds = _resolve_delegation_credentials(cfg, parent_agent=None)
    assert creds["request_overrides"] == {
        "extra_body": {"provider": {"sort": "throughput"}},
    }
    # Shape parity with the named-provider branch: max_output_tokens present.
    assert "max_output_tokens" in creds


def test_direct_branch_absent_request_overrides_stays_none():
    """No delegation.request_overrides → None, preserving the old contract."""
    creds = _resolve_delegation_credentials(_cfg(), parent_agent=None)
    assert creds["request_overrides"] is None


def test_direct_branch_non_dict_request_overrides_stays_none():
    """Garbage in config (string/list) must not crash or forward junk."""
    for bad in ("throughput", ["extra_body"], 42):
        creds = _resolve_delegation_credentials(
            _cfg(request_overrides=bad), parent_agent=None
        )
        assert creds["request_overrides"] is None


def test_direct_branch_deep_copies_nested_extra_body():
    """Transport-side mutation of the child's overrides must not leak back
    into the config dict (copy.deepcopy, not a shallow dict())."""
    source = {"extra_body": {"provider": {"sort": "throughput"}}}
    cfg = _cfg(request_overrides=source)
    creds = _resolve_delegation_credentials(cfg, parent_agent=None)
    creds["request_overrides"]["extra_body"]["provider"]["sort"] = "mutated"
    creds["request_overrides"]["extra_body"]["injected"] = True
    assert source == {"extra_body": {"provider": {"sort": "throughput"}}}


@patch("hermes_cli.runtime_provider.resolve_runtime_provider")
def test_explicit_merges_over_runtime_on_provider_alongside_base_url(mock_resolve):
    """Precedence on the provider-alongside-base_url path (#98237 interplay):
    explicit delegation.request_overrides merges OVER the named provider's
    runtime overrides — runtime extra_body keys survive unless redefined,
    explicit top-level keys win, and max_output_tokens is preserved."""
    mock_resolve.return_value = {
        "provider": "custom",
        "base_url": "https://provider-default.example/v1",
        "api_key": "provider-key",
        "api_mode": "chat_completions",
        "request_overrides": {
            "service_tier": "default",
            "extra_body": {"thinking": {"type": "disabled"}, "provider": {"sort": "price"}},
        },
        "max_output_tokens": 8192,
    }
    cfg = _cfg(
        provider="mimo",
        request_overrides={
            "service_tier": "flex",
            "extra_body": {"provider": {"sort": "throughput"}},
        },
    )
    creds = _resolve_delegation_credentials(cfg, parent_agent=None)
    assert creds["request_overrides"] == {
        # explicit top-level key wins
        "service_tier": "flex",
        "extra_body": {
            # runtime extra_body key survives (not redefined)
            "thinking": {"type": "disabled"},
            # explicit extra_body key wins over runtime's
            "provider": {"sort": "throughput"},
        },
    }
    assert creds["max_output_tokens"] == 8192


# ── Branch 2: named provider (no base_url) ─────────────────────────────────


@patch("hermes_cli.runtime_provider.resolve_runtime_provider")
def test_named_provider_branch_honors_explicit_key(mock_resolve):
    """The named-provider branch merges the explicit key over the provider's
    runtime overrides — the config key never silently no-ops."""
    mock_resolve.return_value = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "runtime-key",
        "api_mode": "chat_completions",
        "request_overrides": {"extra_body": {"reasoning": {"enabled": True}}},
        "max_output_tokens": 4096,
    }
    cfg = {
        "model": "deepseek/deepseek-v4-flash-0731",
        "provider": "openrouter",
        "request_overrides": {"extra_body": {"provider": {"sort": "throughput"}}},
    }
    creds = _resolve_delegation_credentials(cfg, parent_agent=None)
    assert creds["request_overrides"] == {
        "extra_body": {
            "reasoning": {"enabled": True},
            "provider": {"sort": "throughput"},
        }
    }


@patch("hermes_cli.runtime_provider.resolve_runtime_provider")
def test_named_provider_branch_without_explicit_key_unchanged(mock_resolve):
    """Without the config key the named-provider branch behaves as before."""
    mock_resolve.return_value = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "runtime-key",
        "api_mode": "chat_completions",
        "request_overrides": {"extra_body": {"reasoning": {"enabled": True}}},
        "max_output_tokens": 4096,
    }
    cfg = {"model": "m", "provider": "openrouter"}
    creds = _resolve_delegation_credentials(cfg, parent_agent=None)
    assert creds["request_overrides"] == {"extra_body": {"reasoning": {"enabled": True}}}


# ── Branch 3: parent-inherit (no provider, no base_url) ────────────────────


def test_inherit_branch_honors_explicit_key_over_parent():
    """Pure-inherit setups still apply delegation.request_overrides, merged
    over the parent agent's own request_overrides."""
    parent = _parent(
        request_overrides={
            "service_tier": "default",
            "extra_body": {"thinking": {"type": "disabled"}},
        }
    )
    cfg = {
        "model": "",
        "provider": "",
        "request_overrides": {"extra_body": {"provider": {"sort": "throughput"}}},
    }
    creds = _resolve_delegation_credentials(cfg, parent)
    assert creds["request_overrides"] == {
        "service_tier": "default",
        "extra_body": {
            "thinking": {"type": "disabled"},
            "provider": {"sort": "throughput"},
        },
    }


def test_inherit_branch_without_key_stays_none():
    """No explicit key and no parent overrides → None (old contract: the
    child's construction path falls back to the parent's request_overrides)."""
    parent = _parent(request_overrides=None)
    creds = _resolve_delegation_credentials({"model": "", "provider": ""}, parent)
    assert creds["request_overrides"] is None


def test_inherit_branch_deep_copies_parent_overrides():
    """Parent's nested overrides must be deep-copied on the inherit branch."""
    parent_overrides = {"extra_body": {"thinking": {"type": "disabled"}}}
    parent = _parent(request_overrides=parent_overrides)
    cfg = {
        "model": "",
        "provider": "",
        "request_overrides": {"extra_body": {"provider": {"sort": "throughput"}}},
    }
    creds = _resolve_delegation_credentials(cfg, parent)
    creds["request_overrides"]["extra_body"]["thinking"]["type"] = "mutated"
    assert parent_overrides == {"extra_body": {"thinking": {"type": "disabled"}}}


# ── Merge helper unit tests ────────────────────────────────────────────────


def test_merge_helper_both_none():
    assert _merge_request_overrides(None, None) is None
    assert _merge_request_overrides({}, {}) is None
    assert _merge_request_overrides("junk", 42) is None


def test_merge_helper_explicit_only():
    assert _merge_request_overrides(None, {"a": 1}) == {"a": 1}


def test_merge_helper_runtime_only():
    assert _merge_request_overrides({"a": 1}, None) == {"a": 1}


def test_merge_helper_explicit_top_level_wins():
    assert _merge_request_overrides({"a": 1, "b": 2}, {"a": 9}) == {"a": 9, "b": 2}


def test_merge_helper_extra_body_one_level_merge():
    merged = _merge_request_overrides(
        {"extra_body": {"keep": 1, "clash": "runtime"}},
        {"extra_body": {"clash": "explicit", "new": 2}},
    )
    assert merged == {"extra_body": {"keep": 1, "clash": "explicit", "new": 2}}


def test_merge_helper_non_dict_runtime_extra_body_replaced():
    merged = _merge_request_overrides(
        {"extra_body": "junk"}, {"extra_body": {"a": 1}}
    )
    assert merged == {"extra_body": {"a": 1}}
