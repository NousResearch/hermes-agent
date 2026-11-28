"""Managed terminal dotenv is authoritative between profile dotenv and YAML."""
import pytest


@pytest.mark.parametrize("profile_mode,managed_mode,expected", [
    (None, None, "real"),
    ("profile", None, "profile"),
    ("profile", "real", "real"),
])
def test_managed_dotenv_precedes_profile_yaml_but_not_managed_yaml(
        tmp_path, monkeypatch, profile_mode, managed_mode, expected):
    from hermes_cli import managed_scope
    from tools.terminal_scope import build_profile_terminal_scope

    profile = tmp_path / "profile"
    managed = tmp_path / "managed"
    profile.mkdir()
    managed.mkdir()
    (profile / ".env").write_text("TERMINAL_HOME_MODE=profile\n")
    (profile / "config.yaml").write_text(
        f"terminal: {{home_mode: {profile_mode}}}\n" if profile_mode else "{}\n")
    (managed / ".env").write_text("TERMINAL_HOME_MODE=real\nTERMINAL_DOCKER_IMAGE=managed-image\n")
    if managed_mode:
        (managed / "config.yaml").write_text(f"terminal: {{home_mode: {managed_mode}}}\n")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    try:
        scope = build_profile_terminal_scope(profile, env_overlay={"TERMINAL_HOME_MODE": "profile"})
        assert scope["TERMINAL_HOME_MODE"] == expected
        assert scope["TERMINAL_DOCKER_IMAGE"] == "managed-image"
        assert scope["TERMINAL_DOCKER_IMAGE_PINNED"] == "1"
    finally:
        managed_scope.invalidate_managed_cache()
