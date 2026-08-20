"""Tests for the profiles.create RPC's credential-mirroring wiring.

The mirroring logic itself (copy launch .env/auth.json, inherit model,
mirror voice sections) is exercised thoroughly via its REST twin,
POST /api/profiles (tests/hermes_cli/test_web_server.py::TestNewEndpoints).
These tests exist to prove profiles.create is correctly WIRED to the same
shared _mirror_profile_credentials() helper — not to re-verify every mirroring
branch a second time.
"""

from __future__ import annotations

import tui_gateway.server as srv


def _call(method: str, params: dict) -> dict:
    envelope = srv._methods[method](1, params)
    return envelope["result"]


def test_profiles_create_mirrors_launch_credentials_by_default(monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config, save_config
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

    root = get_hermes_home()
    (root / ".env").write_text("LAUNCH_API_KEY=launch-secret\n", encoding="utf-8")
    cfg = load_config()
    cfg["model"] = {"provider": "anthropic", "default": "claude-rpc-mirror-test"}
    save_config(cfg)

    result = _call("profiles.create", {"name": "rpc-mirror-default"})

    assert result["ok"] is True
    assert result["mirrored"]["env"] is True
    assert result["mirrored"]["model_inherited"] is True
    assert result["model_set"] is True

    profile_dir = root / "profiles" / "rpc-mirror-default"
    assert (profile_dir / ".env").read_text(encoding="utf-8") == (
        "LAUNCH_API_KEY=launch-secret\n"
    )


def test_profiles_create_mirror_credentials_false_skips_mirroring(monkeypatch):
    from hermes_constants import get_hermes_home
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

    root = get_hermes_home()
    (root / ".env").write_text("LAUNCH_API_KEY=launch-secret\n", encoding="utf-8")

    result = _call(
        "profiles.create",
        {"name": "rpc-mirror-off", "mirror_credentials": False},
    )

    assert result["ok"] is True
    assert result["mirrored"] == {
        "env": False, "auth": False, "model_inherited": False, "voice": False,
    }
    profile_dir = root / "profiles" / "rpc-mirror-off"
    assert "launch-secret" not in (profile_dir / ".env").read_text(encoding="utf-8")


def test_profiles_create_explicit_model_pin_skips_inheritance(monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config, save_config
    import hermes_cli.profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "create_wrapper_script", lambda name: None)

    cfg = load_config()
    cfg["model"] = {"provider": "anthropic", "default": "claude-rpc-mirror-test"}
    save_config(cfg)

    result = _call(
        "profiles.create",
        {"name": "rpc-mirror-pin", "provider": "openai", "model": "gpt-5"},
    )

    assert result["ok"] is True
    assert result["model_set"] is True
    assert result["mirrored"]["model_inherited"] is False

    import yaml

    root = get_hermes_home()
    profile_cfg = yaml.safe_load(
        (root / "profiles" / "rpc-mirror-pin" / "config.yaml").read_text(encoding="utf-8")
    )
    assert profile_cfg["model"]["provider"] == "openai"
    assert profile_cfg["model"]["default"] == "gpt-5"
