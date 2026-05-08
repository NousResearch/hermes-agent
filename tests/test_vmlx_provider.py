"""Tests for the vMLX model-provider plugin.

Validates the v0.13.0 ``providers/`` contract: two ``ProviderProfile``
instances are registered at module import (vmlx + vmlx-janitor), and the
plugin's ``platform.system()`` guard fails the import cleanly on non-Darwin.

The plugin lives at ``plugins/model-providers/vmlx/`` (hyphen — not a valid
Python identifier) so we load it via ``importlib.util.spec_from_file_location``
the same way ``providers/__init__.py._discover_providers()`` does.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

PLUGIN_INIT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "model-providers"
    / "vmlx"
    / "__init__.py"
)


def _load_plugin(register_provider_mock: Any) -> Any:
    sys.modules.pop("vmlx_plugin_under_test", None)
    spec = importlib.util.spec_from_file_location(
        "vmlx_plugin_under_test", PLUGIN_INIT
    )
    assert spec is not None and spec.loader is not None, (
        f"plugin not found at {PLUGIN_INIT}"
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch("providers.register_provider", register_provider_mock):
        spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(sys.platform != "darwin", reason="vMLX is macOS-only")
def test_plugin_registers_two_profiles() -> None:
    register_provider = mock.MagicMock()
    _load_plugin(register_provider)

    assert register_provider.call_count == 2
    profiles = [c.args[0] for c in register_provider.call_args_list]
    names = {p.name for p in profiles}
    assert names == {"vmlx", "vmlx-janitor"}


@pytest.mark.skipif(sys.platform != "darwin", reason="vMLX is macOS-only")
def test_primary_profile_has_correct_defaults() -> None:
    register_provider = mock.MagicMock()
    _load_plugin(register_provider)

    primary = next(
        c.args[0]
        for c in register_provider.call_args_list
        if c.args[0].name == "vmlx"
    )
    assert primary.base_url == "http://localhost:8000/v1"
    assert primary.env_vars == ()
    assert primary.fallback_models == ()
    assert "mlx" in primary.aliases


@pytest.mark.skipif(sys.platform != "darwin", reason="vMLX is macOS-only")
def test_janitor_profile_on_separate_port() -> None:
    register_provider = mock.MagicMock()
    _load_plugin(register_provider)

    janitor = next(
        c.args[0]
        for c in register_provider.call_args_list
        if c.args[0].name == "vmlx-janitor"
    )
    assert janitor.base_url == "http://localhost:8001/v1"
    assert janitor.env_vars == ()
    assert janitor.fallback_models == ()


def test_plugin_raises_importerror_on_non_darwin() -> None:
    register_provider = mock.MagicMock()
    with mock.patch("platform.system", return_value="Linux"):
        with pytest.raises(ImportError, match="macOS"):
            _load_plugin(register_provider)
    register_provider.assert_not_called()
