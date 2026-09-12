"""Matrix policy uses canonical configuration overlays and expansion."""
import pytest
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_cli import config
from plugins.platforms.matrix import tool_policy

@pytest.fixture
def home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def test_managed_deny_overrides_user_and_legacy_env(home, monkeypatch):
    (home / "config.yaml").write_text("matrix:\n  tools:\n    allow_redaction: true\n")
    monkeypatch.setattr(config.managed_scope, "load_managed_config", lambda: {
        "matrix": {"tools": {"allow_redaction": False}}})
    monkeypatch.setattr(tool_policy, "get_secret", lambda _: "true")
    assert not tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")


def test_policy_values_use_canonical_env_expansion(home, monkeypatch):
    (home / "config.yaml").write_text("matrix:\n  tools:\n    allow_redaction: ${MATRIX_TEST_GATE}\n")
    monkeypatch.setenv("MATRIX_TEST_GATE", "true")
    monkeypatch.setattr(config.managed_scope, "load_managed_config", lambda: {})
    assert tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")


@pytest.mark.parametrize("policy,legacy,expected", [
    ({}, None, False),
    ({}, "true", True),
    ({"allow_redaction": False, "allowed_rooms": []}, "true", False),
])
def test_explicit_policy_overrides_legacy_but_defaults_do_not(home, monkeypatch, policy, legacy, expected):
    import yaml
    from tools.matrix_tool import _allowed_rooms

    (home / "config.yaml").write_text(yaml.safe_dump({"matrix": {"tools": policy}}))
    monkeypatch.setattr(config.managed_scope, "load_managed_config", lambda: {})
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    monkeypatch.delenv("MATRIX_TOOLS_ALLOW_REDACTION", raising=False)
    if legacy is not None:
        monkeypatch.setenv("MATRIX_TOOLS_ALLOW_REDACTION", legacy)
    monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", "!allowed:example.org")
    assert tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION") is expected
    assert _allowed_rooms() == (set() if "allowed_rooms" in policy else {"!allowed:example.org"})


@pytest.mark.parametrize("prior", ["cold", "last-known-good", "cached-fallback"])
def test_normalization_failure_cannot_discard_explicit_deny(home, monkeypatch, prior):
    path = home / "config.yaml"
    monkeypatch.setattr(config.managed_scope, "load_managed_config", lambda: {})
    monkeypatch.setattr(tool_policy, "get_secret", lambda _: "true")
    if prior != "cold":
        path.write_text("matrix:\n  tools:\n    allow_redaction: true\n")
        config.load_config_readonly()
    text = "max_turns: 5\nagent: invalid\nmatrix:\n  tools:\n    allow_redaction: false\n"
    path.write_text(text)
    if prior == "cached-fallback":
        config.load_config_readonly()
    with pytest.raises(ValueError):
        tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")
    assert path.read_text() == text


@pytest.mark.parametrize("text", ["matrix: [broken", "- invalid-root"])
def test_malformed_policy_never_falls_back_to_allowing_env(home, monkeypatch, text):
    (home / "config.yaml").write_text(text)
    monkeypatch.setattr(tool_policy, "get_secret", lambda _: "true")
    with pytest.raises(ValueError):
        tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")
