"""Phase03 integration tests for the delegate_tool per-task credential path.

These exercise ``_resolve_task_credentials`` / ``_parse_task_model_override``
— the glue between the deterministic model router and ``_build_child_agent`` —
with every external resolver monkeypatched, so no live provider APIs are hit.

Contract coverage:
  * router OFF (default) → inherited creds returned unchanged, no fallback
  * explicit tasks[].model / tasks[].provider → bypass router, resolve override
  * forced delegation.provider → bypass router
  * router ON → dynamic selection + provider-diverse fallback chain
  * router ON but resolution fails → safe fallback to inherited creds
"""

import pytest

import tools.delegate_tool as dt


DEFAULT_CREDS = {
    "model": None,
    "provider": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
    "request_overrides": None,
    "max_output_tokens": None,
}


def _bundle(provider, model):
    return {
        "model": model,
        "provider": provider,
        "base_url": f"https://{provider}.example/v1",
        "api_key": "[REDACTED]",
        "api_mode": "chat_completions",
        "request_overrides": {},
        "max_output_tokens": None,
        "command": None,
        "args": [],
    }


# --- _parse_task_model_override -------------------------------------------

def test_parse_override_string_model():
    assert dt._parse_task_model_override({"model": "acme/x"}) == (None, "acme/x")


def test_parse_override_dict_model():
    assert dt._parse_task_model_override(
        {"model": {"provider": "acme", "model": "x"}}
    ) == ("acme", "x")


def test_parse_override_provider_only():
    assert dt._parse_task_model_override({"provider": "acme"}) == ("acme", None)


def test_parse_override_none():
    assert dt._parse_task_model_override({"goal": "hi"}) == (None, None)


# --- router OFF (default) --------------------------------------------------

def test_router_off_returns_inherited_creds():
    cfg = {"model_router": {"enabled": False}}
    creds, chain = dt._resolve_task_credentials({"goal": "write code"}, cfg, DEFAULT_CREDS)
    assert creds is DEFAULT_CREDS
    assert chain is None


def test_no_router_key_returns_inherited_creds():
    creds, chain = dt._resolve_task_credentials({"goal": "write code"}, {}, DEFAULT_CREDS)
    assert creds is DEFAULT_CREDS
    assert chain is None


# --- explicit override bypasses router ------------------------------------

def test_explicit_provider_model_resolves(monkeypatch):
    calls = {}

    def fake_bundle(provider, model):
        calls["args"] = (provider, model)
        return _bundle(provider, model)

    monkeypatch.setattr(dt, "_resolve_provider_model_bundle", fake_bundle)
    cfg = {"model_router": {"enabled": True}}
    creds, chain = dt._resolve_task_credentials(
        {"goal": "x", "model": {"provider": "acme", "model": "x"}}, cfg, DEFAULT_CREDS
    )
    assert calls["args"] == ("acme", "x")
    assert creds["provider"] == "acme"
    assert creds["model"] == "x"
    assert chain is None  # explicit override doesn't build a router chain


def test_explicit_model_string_without_provider_keeps_inherited_auth(monkeypatch):
    # model string alone, no provider → inherit auth, just swap the model id
    monkeypatch.setattr(
        dt, "_resolve_provider_model_bundle",
        lambda p, m: (_ for _ in ()).throw(AssertionError("should not resolve")),
    )
    cfg = {"model_router": {"enabled": True}}
    creds, chain = dt._resolve_task_credentials(
        {"goal": "x", "model": "just-a-model"}, cfg, DEFAULT_CREDS
    )
    assert creds["model"] == "just-a-model"
    assert creds["provider"] is None
    assert chain is None


def test_explicit_override_resolution_failure_falls_back(monkeypatch):
    monkeypatch.setattr(dt, "_resolve_provider_model_bundle", lambda p, m: None)
    cfg = {"model_router": {"enabled": True}}
    creds, chain = dt._resolve_task_credentials(
        {"goal": "x", "model": {"provider": "acme", "model": "x"}}, cfg, DEFAULT_CREDS
    )
    # resolution failed → inherited creds, model id still applied
    assert creds["model"] == "x"
    assert chain is None


# --- forced delegation provider bypasses router ---------------------------

def test_forced_delegation_provider_bypasses_router(monkeypatch):
    forced = dict(DEFAULT_CREDS, provider="forced", model="forced-model")

    def _boom(*a, **k):
        raise AssertionError("router must not run when provider is forced")

    monkeypatch.setattr(dt, "_resolve_provider_model_bundle", _boom)
    cfg = {"model_router": {"enabled": True}}
    creds, chain = dt._resolve_task_credentials({"goal": "x"}, cfg, forced)
    assert creds is forced
    assert chain is None


# --- router ON: dynamic selection -----------------------------------------

def _install_fake_router(monkeypatch, selection):
    """Patch the lazy imports inside _resolve_task_credentials."""
    import agent.model_router as mr
    import hermes_cli.models as models

    monkeypatch.setattr(mr, "router_enabled", lambda cfg: bool(cfg.get("enabled")))
    monkeypatch.setattr(mr, "select_delegation_model", lambda *a, **k: selection)
    monkeypatch.setattr(models, "list_available_providers",
                        lambda: [{"id": "acme", "authenticated": True}])
    monkeypatch.setattr(models, "curated_models_for_provider",
                        lambda p, **k: [("x", "desc")])
    monkeypatch.setattr(models, "get_pricing_for_provider", lambda p, **k: {})


def test_router_on_selects_and_builds_fallback_chain(monkeypatch):
    selection = {
        "selected": {"provider": "acme", "model": "x"},
        "candidates": [],
        "fallback_chain": [
            {"provider": "beta", "model": "y"},
            {"provider": "gamma", "model": "z"},
        ],
        "route": "coding",
    }
    _install_fake_router(monkeypatch, selection)
    monkeypatch.setattr(dt, "_resolve_provider_model_bundle",
                        lambda p, m: _bundle(p, m))

    cfg = {"model_router": {"enabled": True}}
    creds, chain = dt._resolve_task_credentials(
        {"goal": "Implement and debug the parser"}, cfg, DEFAULT_CREDS
    )
    assert creds["provider"] == "acme"
    assert creds["model"] == "x"
    assert chain == [
        {"provider": "beta", "model": "y"},
        {"provider": "gamma", "model": "z"},
    ]


def test_router_on_no_selection_falls_back(monkeypatch):
    _install_fake_router(monkeypatch, None)
    cfg = {"model_router": {"enabled": True}}
    creds, chain = dt._resolve_task_credentials({"goal": "x"}, cfg, DEFAULT_CREDS)
    assert creds is DEFAULT_CREDS
    assert chain is None


def test_router_on_bundle_failure_falls_back(monkeypatch):
    selection = {
        "selected": {"provider": "acme", "model": "x"},
        "candidates": [],
        "fallback_chain": [],
        "route": "coding",
    }
    _install_fake_router(monkeypatch, selection)
    monkeypatch.setattr(dt, "_resolve_provider_model_bundle", lambda p, m: None)
    cfg = {"model_router": {"enabled": True}}
    creds, chain = dt._resolve_task_credentials({"goal": "x"}, cfg, DEFAULT_CREDS)
    assert creds is DEFAULT_CREDS
    assert chain is None


def test_router_end_to_end_unmocked_select(monkeypatch):
    """Exercise the REAL ``agent.model_router.select_delegation_model`` (no mock).

    Only the boundary dependencies are faked — provider inventory, model
    catalog, pricing, credential availability, and the credential-bundle
    resolver — so the router's own route inference, scoring, and
    provider-diverse fallback-chain construction all run for real. This is the
    one test that would catch a wiring/signature drift between the glue and
    the router, which the mocked variants above cannot.
    """
    import agent.model_router as mr  # noqa: F401  (real, NOT monkeypatched)
    import hermes_cli.models as models

    monkeypatch.setattr(
        dt, "_load_config",
        lambda: {"model_router": {"enabled": True, "provider_priority": ["beta"]}},
    )
    monkeypatch.setattr(
        models, "list_available_providers",
        lambda: [
            {"id": "acme", "authenticated": True},
            {"id": "beta", "authenticated": True},
            {"id": "freeco", "authenticated": True},
        ],
    )
    monkeypatch.setattr(
        models, "curated_models_for_provider",
        lambda p, **k: {
            "acme": [("acme/coder-pro", "d"), ("acme/writer", "d")],
            "beta": [("beta/coder", "d")],
            "freeco": [("freeco/oss-coder", "d")],
        }.get(p, []),
    )
    monkeypatch.setattr(
        models, "get_pricing_for_provider",
        lambda p, **k: {
            "acme": {"acme/coder-pro": {"prompt": "1", "completion": "2"}},
            "beta": {"beta/coder": {"prompt": "0.5", "completion": "1"}},
            "freeco": {"freeco/oss-coder": {"prompt": "0.0", "completion": "0.0"}},
        }.get(p, {}),
    )
    monkeypatch.setattr(dt, "_provider_has_available_credentials", lambda p: True)
    monkeypatch.setattr(dt, "_resolve_provider_model_bundle", lambda p, m: _bundle(p, m))

    cfg = {"model_router": {"enabled": True, "provider_priority": ["beta"]}}
    task = {"goal": "Implement and debug the parser module in python, write pytest"}
    creds, chain = dt._resolve_task_credentials(task, cfg, DEFAULT_CREDS)

    # Real router must produce a selection (provider_priority pins beta first).
    assert creds["provider"] == "beta"
    assert creds["model"] == "beta/coder"
    # Fallback chain excludes the selected pair.
    assert chain is not None and len(chain) >= 1
    chain_pairs = {(c["provider"], c["model"]) for c in chain}
    assert ("beta", "beta/coder") not in chain_pairs


# --- model-facing schema ---------------------------------------------------

def test_delegate_schema_exposes_per_task_model_and_provider():
    """Regression: internal override support must be visible to the model."""
    props = dt.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "model" in props
    assert "provider" in props
    assert "explicit model override" in props["model"]["description"]
    assert "explicit provider override" in props["provider"]["description"]

    dynamic = dt._build_dynamic_schema_overrides()
    dynamic_props = dynamic["parameters"]["properties"]["tasks"]["items"]["properties"]
    assert "model" in dynamic_props
    assert "provider" in dynamic_props
    assert "tasks[].model" in dynamic["description"]
    assert "tasks[].provider" in dynamic["description"]
    assert "Subagent model is NOT selectable per call" not in dynamic["description"]
