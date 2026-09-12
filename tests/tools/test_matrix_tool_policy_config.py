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
    managed = home / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("matrix:\n  tools:\n    allow_redaction: false\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(tool_policy, "get_secret", lambda _: "true")
    assert not tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")


def test_policy_values_use_canonical_env_expansion(home, monkeypatch):
    (home / "config.yaml").write_text("matrix:\n  tools:\n    allow_redaction: ${MATRIX_TEST_GATE}\n")
    monkeypatch.setenv("MATRIX_TEST_GATE", "true")
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


@pytest.mark.parametrize("text", ["matrix: [broken", "- invalid-root"])
def test_invalid_managed_policy_cannot_reveal_user_allow(home, monkeypatch, text):
    managed = home / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (home / "config.yaml").write_text("matrix:\n  tools:\n    allow_redaction: true\n")
    path = managed / "config.yaml"
    path.write_text("matrix:\n  tools:\n    allow_redaction: false\n")
    assert not tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")
    path.write_text(text)
    with pytest.raises(ValueError):
        tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")


def test_multiplex_policy_requires_scope_before_yaml_expansion(home, monkeypatch):
    from agent import secret_scope

    (home / "config.yaml").write_text("matrix:\n  tools:\n    allow_redaction: ${MATRIX_TEST_GATE}\n")
    monkeypatch.setenv("MATRIX_TEST_GATE", "true")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    token = secret_scope.set_secret_scope(None)
    try:
        with pytest.raises(ValueError):
            tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")
        scoped_token = secret_scope.set_secret_scope({"MATRIX_TEST_GATE": "false"})
        try:
            assert not tool_policy.gate("allow_redaction", "MATRIX_TOOLS_ALLOW_REDACTION")
        finally:
            secret_scope.reset_secret_scope(scoped_token)
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("rooms", [False, 0, {}, None, [None]])
def test_invalid_room_allowlist_cannot_remove_target_restriction(home, monkeypatch, rooms):
    import yaml
    from tools import matrix_tool

    monkeypatch.setattr(matrix_tool, "get_session_env", lambda key, default="": "!room:example.org")
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump({"matrix": {"tools": {"allowed_rooms": rooms}}}))
    with pytest.raises(ValueError):
        matrix_tool._authorize_room_id()
    path.write_text(yaml.safe_dump({"matrix": {"tools": {"allowed_rooms": ["!room:example.org"]}}}))
    assert matrix_tool._authorize_room_id() == ("!room:example.org", "")
