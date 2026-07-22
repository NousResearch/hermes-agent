"""Tests for ``_warm_configured_platform_sdks`` (#50209).

The helper must resolve exactly the *connected* platforms through the
platform registry (triggering their deferred platform-module imports, which
is what pulls heavy SDKs like lark_oapi into sys.modules before the ready
announcement) and must never propagate failures — the lazy import path in
the request handlers is the fallback.
"""

import types

import gateway.config as gw_config
import gateway.platform_registry as gw_registry

from hermes_cli import web_server


class _FakePlatform:
    def __init__(self, value):
        self.value = value


def _patch_gateway_config(monkeypatch, connected):
    fake_config = types.SimpleNamespace(
        get_connected_platforms=lambda: connected
    )
    monkeypatch.setattr(gw_config, "load_gateway_config", lambda: fake_config)


def test_warmup_resolves_each_connected_platform(monkeypatch):
    _patch_gateway_config(
        monkeypatch, [_FakePlatform("feishu"), _FakePlatform("telegram")]
    )
    calls = []
    monkeypatch.setattr(
        gw_registry, "platform_registry", types.SimpleNamespace(get=calls.append)
    )

    web_server._warm_configured_platform_sdks()

    assert calls == ["feishu", "telegram"]


def test_warmup_with_no_connected_platforms_is_noop(monkeypatch):
    _patch_gateway_config(monkeypatch, [])
    calls = []
    monkeypatch.setattr(
        gw_registry, "platform_registry", types.SimpleNamespace(get=calls.append)
    )

    web_server._warm_configured_platform_sdks()

    assert calls == []


def test_warmup_swallows_config_failures(monkeypatch):
    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(gw_config, "load_gateway_config", _boom)

    # Must not raise — a broken config must not block server startup.
    web_server._warm_configured_platform_sdks()


def test_warmup_swallows_registry_failures(monkeypatch):
    _patch_gateway_config(monkeypatch, [_FakePlatform("feishu")])

    def _boom(name):
        raise RuntimeError(f"cannot import {name}")

    monkeypatch.setattr(
        gw_registry, "platform_registry", types.SimpleNamespace(get=_boom)
    )

    # Must not raise — the request-time lazy import remains as fallback.
    web_server._warm_configured_platform_sdks()
