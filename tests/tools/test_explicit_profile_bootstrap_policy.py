"""Explicit target owners also own configured bootstrap-secret deny policy."""
from pathlib import Path

import pytest

from agent import secret_scope
from hermes_cli import env_loader
from tools.environments.local import build_subprocess_env
from tools.environments.remote_common import resolve_passthrough_env


@pytest.mark.parametrize("surface", ["child", "remote"])
def test_explicit_target_bootstrap_secret_cannot_be_forwarded(monkeypatch, tmp_path, surface):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(source))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    name = "CUSTOM_VAULT_BOOTSTRAP"
    (target / ".env").write_text(f"{name}=synthetic-target-bootstrap\n", encoding="utf-8")
    (target / "config.yaml").write_text(
        f"secrets:\n  bitwarden:\n    enabled: false\n    access_token_env: {name}\n"
        f"terminal:\n  env_passthrough: [{name}]\n", encoding="utf-8")
    token = secret_scope.set_secret_scope(None)
    env_loader.reset_secret_source_cache()
    try:
        if surface == "child":
            child = build_subprocess_env(base={}, profile_home=target,
                                         source_profile_home=source, enforce_profile_boundary=True)
            assert name not in child
        else:
            boundary = secret_scope.build_profile_env_boundary(source, target)
            values, _ = resolve_passthrough_env([name], profile_boundary=boundary)
            assert name not in values
    finally:
        secret_scope.reset_secret_scope(token)
        env_loader.reset_secret_source_cache()
