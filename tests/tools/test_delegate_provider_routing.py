"""Child OpenRouter routing must follow the child's model, not the parent's pin.

When ``delegate_task`` spawns a child on a different OpenRouter model without
an explicit ``override_provider``, inheriting the parent's resolved
``providers_order`` / ``only`` lets sticky_order pin the child to a slug that
has no endpoints for the child's model (OpenRouter 404). The child must
re-resolve ``provider_routing`` from the parent's raw snapshot for *its*
model, and a programmatic parent with no snapshot must not leak constructor
filters across models.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import _provider_preferences_for_agent
from agent.sticky_provider_order import bind_sticky_order
from hermes_constants import (
    provider_routing_model_ids_match,
    resolve_provider_routing_for_model,
)
from tools.delegate_tool import (
    _build_child_agent,
    _is_same_delegation_provider,
    _parent_provider_routing_snapshot,
    _resolve_delegation_credentials,
    _stamp_child_provider_routing,
)


_PARENT_MODEL = "z-ai/glm-4.5"
_CHILD_MODEL = "google/gemini-3.7-flash"
_THIRD_MODEL = "qwen/qwen3.8-27b"
_PARENT_SLUG = "z-ai/fp8"
_CHILD_ONLY = "google-ai-studio"
_FREE_ONLY = "google-ai-studio-free"
_THIRD_ORDER = "reka/fp8"


def _routing(**extra):
    payload = {
        "order": [_PARENT_SLUG, "novita/fp8"],
        "sticky_order": {"enabled": True},
        "models": {
            _CHILD_MODEL: {"only": [_CHILD_ONLY]},
            _THIRD_MODEL: {"order": [_THIRD_ORDER]},
        },
    }
    payload.update(extra)
    return payload


def _parent(**attrs):
    parent = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        provider="openrouter",
        api_mode="chat_completions",
        model=_PARENT_MODEL,
        platform="cli",
        providers_allowed=None,
        providers_ignored=None,
        providers_order=[_PARENT_SLUG, "novita/fp8"],
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection="",
        openrouter_min_coding_score=None,
        enabled_toolsets=["terminal", "file"],
        disabled_toolsets=None,
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
        capabilities={},
        request_overrides={},
        reasoning_config=None,
        max_tokens=None,
        prefill_messages=None,
        _fallback_chain=None,
        _client_kwargs={
            "api_key": "sk-test",
            "base_url": "https://openrouter.ai/api/v1",
        },
        client=None,
        session_id="parent-sid",
        _current_turn_id="",
        _subagent_id=None,
        _provider_routing_config=None,
        acp_command=None,
        acp_args=[],
    )
    for key, value in attrs.items():
        setattr(parent, key, value)
    return parent


def _build_child(parent, **kwargs):
    captured = {}

    class FakeAgent:
        def __init__(self, **kw):
            captured.update(kw)
            for key, value in kw.items():
                setattr(self, key, value)
            self.session_id = "child-session"
            self._session_init_model_config = None

    defaults = {
        "task_index": 0,
        "goal": "child work",
        "context": None,
        "toolsets": None,
        "model": None,
        "max_iterations": 10,
        "task_count": 1,
        "parent_agent": parent,
    }
    defaults.update(kwargs)
    with (
        patch("run_agent.AIAgent", FakeAgent),
        patch("tools.delegate_tool._load_config", return_value={}),
    ):
        child = _build_child_agent(**defaults)
    return child, captured


def _pinned_parent_slug(prefs: dict) -> bool:
    """True when prefs are the failing sticky shape: parent slug, no fallbacks."""
    return prefs.get("allow_fallbacks") is False and prefs.get("order") == [_PARENT_SLUG]


# ── Identity / overlay lookup ──────────────────────────────────────────────


def test_routing_identity_variant_request_matches_base_key():
    assert provider_routing_model_ids_match(
        "google/gemini-3.7-flash", "google/gemini-3.7-flash:free"
    )


def test_routing_identity_variant_key_does_not_match_base_request():
    assert not provider_routing_model_ids_match(
        "google/gemini-3.7-flash:nitro", "google/gemini-3.7-flash"
    )


def test_routing_identity_rejects_cross_variant_skus():
    assert not provider_routing_model_ids_match(
        "openai/gpt-5.2:floor", "openai/gpt-5.2:nitro"
    )


def test_routing_identity_rejects_other_vendor():
    assert not provider_routing_model_ids_match(
        "google/gemini-flash", "openai/gemini-flash"
    )


def test_routing_identity_accepts_version_spelling():
    assert provider_routing_model_ids_match(
        "anthropic/claude-opus-4.5", "anthropic/claude-opus-4-5"
    )


def test_resolve_overlay_matches_variant_suffix():
    resolved = resolve_provider_routing_for_model(
        _routing(), "google/gemini-3.7-flash:free"
    )
    assert resolved["only"] == [_CHILD_ONLY]
    assert "models" not in resolved


def test_resolve_exact_variant_key_beats_base():
    routing = _routing()
    routing["models"][f"{_CHILD_MODEL}:free"] = {"only": [_FREE_ONLY]}
    resolved = resolve_provider_routing_for_model(routing, f"{_CHILD_MODEL}:free")
    assert resolved["only"] == [_FREE_ONLY]


def test_resolve_floor_key_does_not_apply_to_nitro_request():
    routing = {
        "only": ["foo"],
        "models": {
            "openai/gpt-5.2:floor": {"only": ["floor-only"]},
        },
    }
    resolved = resolve_provider_routing_for_model(routing, "openai/gpt-5.2:nitro")
    assert resolved["only"] == ["foo"]
    assert "models" not in resolved


def test_resolve_free_key_does_not_apply_to_base_request():
    routing = {
        "only": ["foo"],
        "models": {
            f"{_CHILD_MODEL}:free": {"only": [_FREE_ONLY]},
        },
    }
    resolved = resolve_provider_routing_for_model(routing, _CHILD_MODEL)
    assert resolved["only"] == ["foo"]


def test_snapshot_distinguishes_none_from_empty_dict():
    assert _parent_provider_routing_snapshot(_parent()) is None
    assert _parent_provider_routing_snapshot(
        _parent(_provider_routing_config={})
    ) == {}
    missing = _parent()
    del missing._provider_routing_config
    assert _parent_provider_routing_snapshot(missing) is None


def test_non_dict_snapshot_warns_and_is_missing(caplog):
    import logging

    parent = _parent(_provider_routing_config="not-a-dict")
    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        snapshot = _parent_provider_routing_snapshot(parent)
    assert snapshot is None
    assert "_provider_routing_config" in caplog.text
    assert "not a dict" in caplog.text


# ── Different model + sticky + per-model only ──────────────────────────────


def test_different_model_gets_per_model_only_not_parent_sticky_pin():
    routing = _routing()
    parent = _parent(_provider_routing_config=routing)
    bind_sticky_order(parent, routing)
    parent._sticky_provider_order.active_index = 1

    child, captured = _build_child(parent, model=_CHILD_MODEL)

    assert captured["providers_allowed"] == [_CHILD_ONLY]
    assert captured["model"] == _CHILD_MODEL
    prefs = _provider_preferences_for_agent(child)
    assert prefs.get("only") == [_CHILD_ONLY]
    assert not _pinned_parent_slug(prefs)
    if prefs.get("allow_fallbacks") is False:
        assert _PARENT_SLUG not in (prefs.get("order") or [])
    assert child._sticky_provider_order is not parent._sticky_provider_order
    assert child._sticky_provider_order.active_index == 0


def test_delegation_model_inherit_branch_uses_same_re_resolve():
    """``delegation.model`` without ``delegation.provider`` is the same path."""
    routing = _routing()
    parent = _parent(_provider_routing_config=routing)
    creds = _resolve_delegation_credentials(
        {"model": _CHILD_MODEL, "provider": ""},
        parent,
    )
    assert creds["provider"] is None
    assert creds["model"] == _CHILD_MODEL

    child, captured = _build_child(
        parent,
        model=creds["model"],
        override_provider=creds["provider"],
        override_base_url=creds["base_url"],
        override_api_key=creds["api_key"],
        override_api_mode=creds["api_mode"],
        override_request_overrides=creds.get("request_overrides"),
    )
    assert captured["providers_allowed"] == [_CHILD_ONLY]
    assert not _pinned_parent_slug(_provider_preferences_for_agent(child))


# ── Same model keeps inherited filters ─────────────────────────────────────


def test_same_model_keeps_inherited_filters():
    routing = _routing()
    parent = _parent(
        providers_order=[_PARENT_SLUG],
        providers_allowed=[_PARENT_SLUG],
        _provider_routing_config=routing,
    )
    child, captured = _build_child(parent, model=_PARENT_MODEL)

    assert captured["providers_order"] == [_PARENT_SLUG]
    assert captured["providers_allowed"] == [_PARENT_SLUG]
    prefs = _provider_preferences_for_agent(child)
    assert prefs["order"] == [_PARENT_SLUG]
    assert prefs.get("allow_fallbacks") is False


def test_same_model_via_none_uses_parent_model():
    parent = _parent(
        providers_order=[_PARENT_SLUG],
        _provider_routing_config=_routing(),
    )
    _child, captured = _build_child(parent, model=None)
    assert captured["model"] == _PARENT_MODEL
    assert captured["providers_order"] == [_PARENT_SLUG]


# ── Programmatic parent (no snapshot) ──────────────────────────────────────


def test_programmatic_parent_different_model_drops_order():
    parent = _parent(
        providers_order=[_PARENT_SLUG, "novita/fp8"],
        providers_allowed=[_PARENT_SLUG],
        providers_ignored=["other"],
        _provider_routing_config=None,
    )
    _child, captured = _build_child(parent, model=_CHILD_MODEL)
    assert captured["providers_order"] is None
    assert captured["providers_allowed"] is None
    assert captured["providers_ignored"] is None


def test_programmatic_parent_missing_attr_different_model_drops_order():
    parent = _parent(providers_order=[_PARENT_SLUG])
    del parent._provider_routing_config
    _child, captured = _build_child(parent, model=_CHILD_MODEL)
    assert captured["providers_order"] is None
    assert captured["providers_allowed"] is None


def test_programmatic_parent_same_model_keeps_constructor_order():
    parent = _parent(
        providers_order=[_PARENT_SLUG],
        _provider_routing_config=None,
    )
    _child, captured = _build_child(parent, model=_PARENT_MODEL)
    assert captured["providers_order"] == [_PARENT_SLUG]


def test_empty_dict_snapshot_is_explicit_empty_not_missing():
    parent = _parent(
        providers_order=[_PARENT_SLUG],
        _provider_routing_config={},
    )
    child, captured = _build_child(parent, model=_CHILD_MODEL)
    assert captured["providers_order"] is None
    assert captured["providers_allowed"] is None
    assert child._provider_routing_config == {}


# ── Raw snapshot on the child (nested delegation) ──────────────────────────


def test_raw_routing_snapshot_is_available_on_child():
    routing = _routing()
    parent = _parent(_provider_routing_config=routing)
    child, _captured = _build_child(parent, model=_CHILD_MODEL)
    raw = child._provider_routing_config
    assert isinstance(raw, dict)
    assert raw["models"][_CHILD_MODEL]["only"] == [_CHILD_ONLY]
    assert raw is not routing
    raw["models"][_THIRD_MODEL] = {"only": ["mutated"]}
    assert routing["models"][_THIRD_MODEL] == {"order": [_THIRD_ORDER]}
    raw["models"][_CHILD_MODEL]["only"].append("leaked")
    assert routing["models"][_CHILD_MODEL]["only"] == [_CHILD_ONLY]


def test_nested_delegation_resolves_third_model_from_child_snapshot():
    routing = _routing()
    parent = _parent(_provider_routing_config=routing)
    mid, _ = _build_child(parent, model=_CHILD_MODEL)
    grandchild, captured = _build_child(mid, model=_THIRD_MODEL)
    assert captured["providers_order"] == [_THIRD_ORDER]
    assert grandchild._provider_routing_config["models"][_THIRD_MODEL]["order"] == [
        _THIRD_ORDER
    ]


# ── Canonicalization on the spawn path ─────────────────────────────────────


def test_variant_suffix_without_own_key_uses_base_overlay():
    """``:free`` vs base is not inherit; the base ``models.`` overlay applies."""
    parent = _parent(
        model=_CHILD_MODEL,
        providers_order=[_CHILD_ONLY],
        providers_allowed=[_CHILD_ONLY],
        _provider_routing_config=_routing(),
    )
    _child, captured = _build_child(parent, model=f"{_CHILD_MODEL}:free")
    assert captured["providers_allowed"] == [_CHILD_ONLY]
    # Re-resolved from the snapshot, not the parent's already-applied filters.
    assert captured["providers_order"] == [_PARENT_SLUG, "novita/fp8"]


def test_variant_suffix_own_key_wins_over_base_overlay():
    routing = _routing()
    routing["models"][f"{_CHILD_MODEL}:free"] = {"only": [_FREE_ONLY]}
    parent = _parent(
        model=_CHILD_MODEL,
        providers_order=[_CHILD_ONLY],
        providers_allowed=[_CHILD_ONLY],
        _provider_routing_config=routing,
    )
    _child, captured = _build_child(parent, model=f"{_CHILD_MODEL}:free")
    assert captured["providers_allowed"] == [_FREE_ONLY]


def test_variant_suffix_child_resolves_base_overlay_key():
    parent = _parent(_provider_routing_config=_routing())
    _child, captured = _build_child(parent, model=f"{_CHILD_MODEL}:free")
    assert captured["providers_allowed"] == [_CHILD_ONLY]


def test_stamp_logs_setattr_failure(caplog):
    import logging

    class _Frozen:
        def __setattr__(self, name, value):
            raise RuntimeError("frozen")

    with caplog.at_level(logging.WARNING, logger="tools.delegate_tool"):
        _stamp_child_provider_routing(_Frozen(), {"order": ["x"]}, "m")
    assert "Could not stamp child _provider_routing_config" in caplog.text
    assert "Could not bind sticky provider order on child" in caplog.text


# ── Fallback / restore resync (config-managed routing) ─────────────────────


def test_stamped_child_is_config_managed_and_fallback_reresolves():
    """A snapshot-stamped child must resync routing when the model changes."""
    from agent.agent_runtime_helpers import resync_per_model_routing_and_tier

    parent = _parent(_provider_routing_config=_routing())
    child, _captured = _build_child(parent, model=_CHILD_MODEL)
    assert child._config_managed_routing_tier is True
    assert child.providers_allowed == [_CHILD_ONLY]

    child.model = _THIRD_MODEL
    resync_per_model_routing_and_tier(child)
    prefs = _provider_preferences_for_agent(child)
    assert prefs.get("only") != [_CHILD_ONLY]
    assert _CHILD_ONLY not in (prefs.get("only") or [])
    assert child.providers_order == [_THIRD_ORDER]

    child.model = _CHILD_MODEL
    resync_per_model_routing_and_tier(child)
    assert child.providers_allowed == [_CHILD_ONLY]


def test_stamped_child_restore_primary_reresolves_original_overlay():
    """restore_primary_runtime uses the same resync; original only returns."""
    from agent.agent_runtime_helpers import restore_primary_runtime
    from agent.chat_completion_helpers import try_activate_fallback
    from run_agent import AIAgent

    routing = _routing()
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        child = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model=_CHILD_MODEL,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            providers_allowed=[_CHILD_ONLY],
            fallback_model=[
                {
                    "provider": "openrouter",
                    "model": _THIRD_MODEL,
                    "base_url": "https://openrouter.ai/api/v1",
                }
            ],
        )
    _stamp_child_provider_routing(child, routing, _CHILD_MODEL)
    assert child._config_managed_routing_tier is True
    child.client = MagicMock()
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "fb-key"
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(mock_client, _THIRD_MODEL),
    ):
        assert try_activate_fallback(child) is True
    assert child.model == _THIRD_MODEL
    prefs = _provider_preferences_for_agent(child)
    assert _CHILD_ONLY not in (prefs.get("only") or [])
    assert child.providers_order == [_THIRD_ORDER]

    assert restore_primary_runtime(child) is True
    assert child.model == _CHILD_MODEL
    assert child.providers_allowed == [_CHILD_ONLY]


def test_programmatic_child_stays_unmanaged_on_fallback():
    """No snapshot → constructor filters win; resync is a no-op."""
    from agent.agent_runtime_helpers import resync_per_model_routing_and_tier

    parent = _parent(_provider_routing_config=None)
    child, _captured = _build_child(parent, model=_CHILD_MODEL)
    assert getattr(child, "_config_managed_routing_tier", False) is not True
    child.providers_allowed = ["constructor-only"]
    child.model = _THIRD_MODEL
    resync_per_model_routing_and_tier(child)
    assert child.providers_allowed == ["constructor-only"]


# ── Same-provider explicit pin is not an override ──────────────────────────


def test_same_delegation_provider_comparison():
    assert _is_same_delegation_provider("openrouter", "openrouter") is True
    assert _is_same_delegation_provider("OpenRouter", "openrouter") is True
    assert _is_same_delegation_provider("z-ai", "zai") is True
    assert _is_same_delegation_provider("minimax", "openrouter") is False
    assert _is_same_delegation_provider("openrouter", None) is False
    assert _is_same_delegation_provider("openrouter", "") is False
    assert _is_same_delegation_provider(None, "openrouter") is False
    # * Bare custom is a family label, not an endpoint
    assert _is_same_delegation_provider("custom", "custom") is False
    assert _is_same_delegation_provider("custom:foo", "custom") is False
    assert _is_same_delegation_provider("custom:foo", "custom:foo") is True
    assert _is_same_delegation_provider("custom:Foo", "custom:foo") is True
    assert _is_same_delegation_provider("custom:foo", "custom:bar") is False


def test_explicit_same_provider_different_model_gets_per_model_overlay():
    """``delegation.provider: openrouter`` on an OpenRouter parent is a pin,
    not a provider switch — child's models.<id>.only must apply.
    """
    routing = _routing()
    parent = _parent(_provider_routing_config=routing)
    child, captured = _build_child(
        parent,
        model=_CHILD_MODEL,
        override_provider="openrouter",
    )
    assert captured["providers_allowed"] == [_CHILD_ONLY]
    assert child._provider_routing_config["models"][_CHILD_MODEL]["only"] == [
        _CHILD_ONLY
    ]
    assert child._config_managed_routing_tier is True
    prefs = _provider_preferences_for_agent(child)
    assert prefs.get("only") == [_CHILD_ONLY]
    assert not _pinned_parent_slug(prefs)


def test_explicit_same_provider_same_model_inherits_filters():
    routing = _routing()
    parent = _parent(
        providers_order=[_PARENT_SLUG],
        providers_allowed=[_PARENT_SLUG],
        _provider_routing_config=routing,
    )
    child, captured = _build_child(
        parent,
        model=_PARENT_MODEL,
        override_provider="openrouter",
    )
    assert captured["providers_order"] == [_PARENT_SLUG]
    assert captured["providers_allowed"] == [_PARENT_SLUG]
    assert child._config_managed_routing_tier is True
    prefs = _provider_preferences_for_agent(child)
    assert prefs["order"] == [_PARENT_SLUG]


def test_explicit_same_provider_none_override_unchanged():
    """``delegation.provider`` unset (None) keeps the inherit/re-resolve path."""
    routing = _routing()
    parent = _parent(_provider_routing_config=routing)
    child, captured = _build_child(
        parent,
        model=_CHILD_MODEL,
        override_provider=None,
    )
    assert captured["providers_allowed"] == [_CHILD_ONLY]
    assert child._config_managed_routing_tier is True


def test_uncomparable_parent_provider_clears_filters():
    """Empty parent provider is incomparable — keep the old clear."""
    parent = _parent(
        provider="",
        providers_order=[_PARENT_SLUG],
        providers_allowed=[_PARENT_SLUG],
        _provider_routing_config=_routing(),
    )
    child, captured = _build_child(
        parent,
        model=_CHILD_MODEL,
        override_provider="openrouter",
    )
    assert captured["providers_order"] is None
    assert captured["providers_allowed"] is None
    assert not hasattr(child, "_provider_routing_config")
    assert getattr(child, "_config_managed_routing_tier", False) is not True


# ── override_provider path unchanged ───────────────────────────────────────


def test_override_provider_still_clears_filters():
    parent = _parent(
        providers_order=[_PARENT_SLUG],
        providers_allowed=[_PARENT_SLUG],
        _provider_routing_config=_routing(),
    )
    child, captured = _build_child(
        parent,
        model="minimax/m2",
        override_provider="minimax",
        override_base_url="https://api.minimax.example/v1",
        override_api_key="sk-mm-x",
    )
    assert captured["providers_order"] is None
    assert captured["providers_allowed"] is None
    # * override_provider must not stamp the parent's OpenRouter snapshot.
    assert not hasattr(child, "_provider_routing_config")
