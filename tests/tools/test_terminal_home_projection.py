"""Profile terminal projection retains home policy and refuses corrupt YAML."""
import pytest


@pytest.mark.parametrize("setting,expected", [(None, "auto"), ("real", "real"), ("profile", "profile")])
def test_profile_projection_home_mode(tmp_path, monkeypatch, setting, expected):
    from tools.terminal_scope import build_profile_terminal_scope
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    value = "{}\n" if setting is None else f"terminal:\n  home_mode: {setting}\n"
    (tmp_path / "config.yaml").write_text(value)
    result = build_profile_terminal_scope(tmp_path)
    assert result["TERMINAL_HOME_MODE"] == expected


def test_profile_projection_refuses_invalid_yaml(tmp_path, monkeypatch):
    from tools.terminal_scope import TerminalPolicyUnavailable, build_profile_terminal_scope
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    (tmp_path / "config.yaml").write_text("terminal: [unterminated\n")
    with pytest.raises(TerminalPolicyUnavailable):
        build_profile_terminal_scope(tmp_path)
