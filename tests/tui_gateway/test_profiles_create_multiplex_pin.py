"""profiles.create (Desktop New Agent RPC) pins multiplex secondary API server."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest
import yaml

from tui_gateway.methods_profiles import _registry
from utils import is_truthy_value


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    monkeypatch.delenv("API_SERVER_ENABLED", raising=False)
    return default_home


def _bind_profiles_create():
    fn = next(f for n, f in _registry._pending if n == "profiles.create")

    def _ok(rid, result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def _err(rid, code, msg, data=None):
        error = {"code": code, "message": msg}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": rid, "error": error}

    g = dict(fn.__globals__)
    g.update({"_ok": _ok, "_err": _err, "is_truthy_value": is_truthy_value, "os": os})
    return types.FunctionType(
        fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__
    )


def _write_launch(home: Path) -> None:
    (home / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n"
        "platforms:\n  api_server:\n    enabled: true\n",
        encoding="utf-8",
    )
    (home / ".env").write_text("API_SERVER_KEY=repro-api-server-key-16+\n", encoding="utf-8")


def _api_enabled(home: Path):
    cfg = home / "config.yaml"
    if not cfg.is_file():
        return None
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    return ((data.get("platforms") or {}).get("api_server") or {}).get("enabled")


def _port_binding(home: Path) -> list[str]:
    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )
    from gateway.config import load_gateway_config, platform_binds_port
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    secret = set_secret_scope(build_profile_secret_scope(home))
    try:
        cfg = load_gateway_config()
    finally:
        reset_secret_scope(secret)
        reset_hermes_home_override(token)
    return sorted(
        platform.value
        for platform, pcfg in cfg.platforms.items()
        if pcfg.enabled and platform_binds_port(platform.value, pcfg.extra)
    )


def test_fresh_create_mirrors_key_but_pins_disable(profile_env):
    _write_launch(profile_env)
    handler = _bind_profiles_create()
    result = handler(1, {"name": "freshbot"})["result"]
    assert result["ok"] is True
    home = Path(result["path"])
    assert result["mirrored"]["env"] is True
    assert result["multiplex"]["multiplex"] is True
    assert result["multiplex"]["served"] is True
    assert (home / ".env").read_text(encoding="utf-8").startswith("API_SERVER_KEY=")
    assert _api_enabled(home) is False
    assert _port_binding(home) == []


def test_clone_from_default_does_not_keep_listener_intent(profile_env):
    _write_launch(profile_env)
    handler = _bind_profiles_create()
    result = handler(2, {"name": "clonebot", "clone_from": "default"})["result"]
    home = Path(result["path"])
    assert _api_enabled(home) is False
    assert _port_binding(home) == []
    assert "API_SERVER_KEY=" in (home / ".env").read_text(encoding="utf-8")
    assert result["multiplex"]["served"] is True


def test_no_pin_when_launch_is_not_multiplexed(profile_env):
    (profile_env / "config.yaml").write_text(
        "platforms:\n  api_server:\n    enabled: true\n",
        encoding="utf-8",
    )
    (profile_env / ".env").write_text(
        "API_SERVER_KEY=repro-api-server-key-16+\n", encoding="utf-8"
    )
    handler = _bind_profiles_create()
    result = handler(3, {"name": "lonely", "clone_from": "default"})["result"]
    home = Path(result["path"])
    assert result["multiplex"]["multiplex"] is False
    assert _api_enabled(home) is True
