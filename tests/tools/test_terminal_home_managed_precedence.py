"""Managed dotenv enters terminal scope before profile YAML, managed YAML last."""
import os

import pytest


@pytest.mark.parametrize("profile_yaml,managed_yaml,expected", [
    (None, None, "real"),
    ("profile", None, "profile"),
    ("profile", "real", "real"),
    ("real", "profile", "profile"),
])
def test_managed_dotenv_yaml_precedence(tmp_path, monkeypatch, profile_yaml, managed_yaml, expected):
    from hermes_cli import managed_scope
    from tools.terminal_scope import build_profile_terminal_scope, set_terminal_scope, reset_terminal_scope
    from tools.environments.local import build_subprocess_env

    owner, managed, real = tmp_path / "owner", tmp_path / "managed", tmp_path / "real"
    (owner / "home").mkdir(parents=True)
    managed.mkdir()
    real.mkdir()
    (owner / ".env").write_text("TERMINAL_HOME_MODE=profile\n")
    (owner / "config.yaml").write_text(f"terminal: {{home_mode: {profile_yaml}}}\n" if profile_yaml else "{}\n")
    (managed / ".env").write_text("TERMINAL_HOME_MODE=real\n")
    if managed_yaml:
        (managed / "config.yaml").write_text(f"terminal: {{home_mode: {managed_yaml}}}\n")
    for k, v in {"HERMES_HOME": str(owner), "HOME": str(real), "HERMES_REAL_HOME": str(real),
                 "HERMES_MANAGED_DIR": str(managed), "TERMINAL_HOME_MODE": "profile"}.items():
        monkeypatch.setenv(k, v)
    before = dict(os.environ)
    managed_scope.invalidate_managed_cache()
    try:
        scope = build_profile_terminal_scope(owner, env_overlay={"TERMINAL_HOME_MODE": "profile"})
        assert scope["TERMINAL_HOME_MODE"] == expected
        token = set_terminal_scope(scope)
        try:
            child = build_subprocess_env()
            assert child["HOME"] == str(real if expected == "real" else owner / "home")
            assert child["HERMES_REAL_HOME"] == str(real)
        finally:
            reset_terminal_scope(token)
    finally:
        managed_scope.invalidate_managed_cache()
    assert dict(os.environ) == before
