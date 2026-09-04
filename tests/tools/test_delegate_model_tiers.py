"""Tests for per-task model selection via ``delegation.model_tiers``.

The model picks a TIER NAME (constrained by a schema enum built from the
user's own config), not a raw model id — so a hallucinated model string can
never reach a provider. Tier specs are partial delegation-config blocks
layered over the global delegation config and resolved through the same
``_resolve_delegation_credentials`` path as the global pin.

Contract covered here:

1. ``_load_model_tiers`` normalisation (mapping, string shorthand, junk).
2. ``_resolve_tier_credentials`` layering, unknown-tier refusal, and the
   no-tier passthrough that keeps existing calls byte-for-byte identical.
3. Schema shaping: the ``model_tier`` property appears with the configured
   enum ONLY when tiers exist, and is dropped entirely otherwise.
"""

from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _build_dynamic_schema_overrides,
    _load_model_tiers,
    _resolve_tier_credentials,
)


BASE_CREDS = {
    "model": "default-model",
    "provider": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
    "request_overrides": None,
    "max_output_tokens": None,
}


def _parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.request_overrides = None
    return parent


# ── _load_model_tiers normalisation ────────────────────────────────────────


def test_load_tiers_mapping_form():
    cfg = {
        "model_tiers": {
            "fast": {"provider": "openrouter", "model": "cheap-model"},
            "deep": {"model": "expensive-model"},
        }
    }
    assert _load_model_tiers(cfg) == {
        "fast": {"provider": "openrouter", "model": "cheap-model"},
        "deep": {"model": "expensive-model"},
    }


def test_load_tiers_string_shorthand_expands_to_model():
    """A bare string is shorthand for {model: <string>}."""
    assert _load_model_tiers({"model_tiers": {"fast": "cheap-model"}}) == {
        "fast": {"model": "cheap-model"}
    }


def test_load_tiers_names_are_lowercased():
    assert set(_load_model_tiers({"model_tiers": {"Fast": "m", "DEEP": "n"}})) == {
        "fast",
        "deep",
    }


def test_load_tiers_absent_or_malformed_returns_empty():
    for cfg in ({}, {"model_tiers": None}, {"model_tiers": "fast"}, {"model_tiers": []}):
        assert _load_model_tiers(cfg) == {}


def test_load_tiers_drops_junk_entries_without_raising():
    """Garbage values are skipped, valid siblings survive — this runs inside
    get_definitions() and must never break tool listing."""
    cfg = {"model_tiers": {"fast": "cheap", "bad": 42, "": "x", "none": None}}
    assert _load_model_tiers(cfg) == {"fast": {"model": "cheap"}}


def test_load_tiers_does_not_alias_config_dicts():
    """Mutating a returned tier spec must not write back into config."""
    source = {"model": "cheap-model"}
    cfg = {"model_tiers": {"fast": source}}
    _load_model_tiers(cfg)["fast"]["model"] = "mutated"
    assert source == {"model": "cheap-model"}


# ── _resolve_tier_credentials ──────────────────────────────────────────────


def test_no_tier_returns_base_creds_identity():
    """Absent tier → the exact same object, so untiered calls are unchanged."""
    for tier in (None, "", "   "):
        creds, err = _resolve_tier_credentials(tier, {}, _parent(), BASE_CREDS)
        assert err is None
        assert creds is BASE_CREDS


def test_tier_overrides_model_and_inherits_global_provider():
    """A tier setting only `model` keeps the global provider/base_url."""
    cfg = {
        "base_url": "https://example.test/v1",
        "api_key": "global-key-1234567890",
        "model": "default-model",
        "model_tiers": {"fast": {"model": "cheap-model"}},
    }
    creds, err = _resolve_tier_credentials("fast", cfg, _parent(), BASE_CREDS)
    assert err is None
    assert creds["model"] == "cheap-model"
    assert creds["base_url"] == "https://example.test/v1"


def test_tier_name_is_case_insensitive():
    cfg = {"model": "default-model", "model_tiers": {"fast": {"model": "cheap"}}}
    creds, err = _resolve_tier_credentials("FAST", cfg, _parent(), BASE_CREDS)
    assert err is None
    assert creds["model"] == "cheap"


def test_unknown_tier_refuses_and_names_valid_tiers():
    """A silent downgrade to the default model is exactly the failure mode
    this feature must avoid — refuse loudly and list the real tiers."""
    cfg = {"model_tiers": {"fast": "a", "deep": "b"}}
    creds, err = _resolve_tier_credentials("turbo", cfg, _parent(), BASE_CREDS)
    assert creds is BASE_CREDS
    assert err is not None
    assert "turbo" in err and "deep" in err and "fast" in err


def test_tier_requested_with_no_tiers_configured_errors():
    creds, err = _resolve_tier_credentials("fast", {}, _parent(), BASE_CREDS)
    assert creds is BASE_CREDS
    assert err is not None
    assert "model_tiers" in err


def test_model_tiers_key_never_reaches_credential_resolver():
    """The nested mapping is not a credential key; it must be stripped before
    _resolve_delegation_credentials sees the merged config-shaped dict."""
    cfg = {"model": "default-model", "model_tiers": {"fast": {"model": "cheap"}}}
    with patch("tools.delegate_tool._resolve_delegation_credentials") as mock_resolve:
        mock_resolve.return_value = dict(BASE_CREDS)
        _resolve_tier_credentials("fast", cfg, _parent(), BASE_CREDS)
    merged = mock_resolve.call_args[0][0]
    assert "model_tiers" not in merged
    assert merged["model"] == "cheap"


def test_tier_resolution_failure_surfaces_as_error_not_exception():
    """A tier naming an unresolvable provider reports an error string; the
    ValueError must not escape into the dispatch path."""
    cfg = {"model_tiers": {"fast": {"provider": "nonexistent-provider"}}}
    with patch("tools.delegate_tool._resolve_delegation_credentials") as mock_resolve:
        mock_resolve.side_effect = ValueError("no such provider")
        creds, err = _resolve_tier_credentials("fast", cfg, _parent(), BASE_CREDS)
    assert creds is BASE_CREDS
    assert err is not None and "no such provider" in err


# ── Schema shaping ─────────────────────────────────────────────────────────


def _task_item_props(overrides):
    return overrides["parameters"]["properties"]["tasks"]["items"]["properties"]


def test_schema_omits_model_tier_when_unconfigured():
    """A default install must not advertise a knob that cannot work."""
    with patch("tools.delegate_tool._load_model_tiers", return_value={}):
        assert "model_tier" not in _task_item_props(_build_dynamic_schema_overrides())


def test_schema_advertises_configured_tiers_as_enum():
    tiers = {"fast": {"model": "cheap-model"}, "deep": {"model": "expensive-model"}}
    with patch("tools.delegate_tool._load_model_tiers", return_value=tiers):
        prop = _task_item_props(_build_dynamic_schema_overrides())["model_tier"]
    assert prop["enum"] == ["deep", "fast"]
    # The model names are surfaced so the model can judge cost/capability.
    assert "cheap-model" in prop["description"]


def test_schema_shaping_does_not_mutate_the_static_schema():
    """_build_dynamic_schema_overrides runs on every get_definitions() call;
    leaking model_tier into the module-level dict would make an unconfigured
    session inherit a previous session's tiers."""
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    tiers = {"fast": {"model": "cheap-model"}}
    with patch("tools.delegate_tool._load_model_tiers", return_value=tiers):
        _build_dynamic_schema_overrides()

    static_props = (
        DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    )
    assert static_props["model_tier"]["description"] == (
        "(injected at get_definitions() time)"
    )
    assert "enum" not in static_props["model_tier"]

    with patch("tools.delegate_tool._load_model_tiers", return_value={}):
        assert "model_tier" not in _task_item_props(_build_dynamic_schema_overrides())
