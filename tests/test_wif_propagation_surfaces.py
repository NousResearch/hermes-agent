from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wif_does_not_add_a_parallel_bearer_mode_flag():
    """A callable api_key is the complete WIF credential and transport contract."""
    forbidden = "anthropic_force_" + "bearer_auth"
    offenders = []
    for path in ROOT.rglob("*.py"):
        if path == Path(__file__):
            continue
        if forbidden in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_wif_callable_contract_is_used_by_runtime_and_client_builder():
    runtime = (ROOT / "hermes_cli/runtime_provider.py").read_text(encoding="utf-8")
    adapter = (ROOT / "agent/anthropic_adapter.py").read_text(encoding="utf-8")

    assert '"api_key": token' in runtime
    assert "build_anthropic_wif_token_provider" in runtime
    assert "if callable(api_key) and not isinstance(api_key, str):" in adapter
    assert "_build_anthropic_client_with_bearer_hook(" in adapter


def test_fallback_re_resolves_anthropic_runtime_callable():
    helper = (ROOT / "agent/chat_completion_helpers.py").read_text(encoding="utf-8")
    assert 'resolve_runtime_provider(\n                        requested="anthropic"' in helper
    assert 'resolved_key = fb_runtime.get("api_key")' in helper
    assert "build_anthropic_client(\n                effective_key" in helper
