"""Unit tests for hermes_constants.get_bundled_whatsapp_bridge_dir().

Shipping the bridge in the wheel/sdist (test_packaging_metadata.py) is only half
the fix: the runtime must also *find* it. In a wheel install the gateway adapter
module lives at ``<site-packages>/gateway/platforms/whatsapp.py``, so the old
``parents[2]/scripts/whatsapp-bridge`` resolved to ``<site-packages>/scripts/...``
— a location setuptools data-files never populate (they land under the interpreter
``data`` scheme, i.e. ``sys.prefix``). These tests pin the resolution order so the
packaged-install path can't silently regress to the source-only behavior.
"""

from pathlib import Path

import hermes_constants
from hermes_constants import get_bundled_whatsapp_bridge_dir


def test_env_var_override_wins(tmp_path, monkeypatch):
    custom = tmp_path / "custom-bridge"
    custom.mkdir()
    monkeypatch.setenv("HERMES_WHATSAPP_BRIDGE_DIR", str(custom))
    assert get_bundled_whatsapp_bridge_dir() == custom


def test_env_var_empty_string_ignored(tmp_path, monkeypatch):
    """An empty override must fall through to the normal resolution chain."""
    monkeypatch.setenv("HERMES_WHATSAPP_BRIDGE_DIR", "")
    # No packaged dir on a source checkout, so we land on the source path.
    result = get_bundled_whatsapp_bridge_dir()
    assert result.name == "whatsapp-bridge"
    assert result.parent.name == "scripts"


def test_packaged_data_dir_used_when_present(tmp_path, monkeypatch):
    """Wheel install: data-files land under a sysconfig scheme, not site-packages.

    Simulate that by pointing the ``data`` scheme at a tmp prefix that contains
    ``scripts/whatsapp-bridge`` and asserting the resolver finds it there rather
    than the source-tree fallback.
    """
    monkeypatch.delenv("HERMES_WHATSAPP_BRIDGE_DIR", raising=False)
    prefix = tmp_path / "prefix"
    packaged = prefix / "scripts" / "whatsapp-bridge"
    packaged.mkdir(parents=True)

    real_get_path = hermes_constants.sysconfig.get_path

    def fake_get_path(scheme, *args, **kwargs):
        if scheme == "data":
            return str(prefix)
        return real_get_path(scheme, *args, **kwargs)

    monkeypatch.setattr(hermes_constants.sysconfig, "get_path", fake_get_path)
    assert get_bundled_whatsapp_bridge_dir() == packaged


def test_env_var_wins_over_packaged(tmp_path, monkeypatch):
    """Explicit override beats an installed data-files dir."""
    prefix = tmp_path / "prefix"
    (prefix / "scripts" / "whatsapp-bridge").mkdir(parents=True)
    override = tmp_path / "override"
    override.mkdir()
    monkeypatch.setenv("HERMES_WHATSAPP_BRIDGE_DIR", str(override))
    monkeypatch.setattr(
        hermes_constants.sysconfig, "get_path",
        lambda scheme, *a, **k: str(prefix) if scheme == "data" else None,
    )
    assert get_bundled_whatsapp_bridge_dir() == override


def test_source_fallback_points_at_repo_scripts(monkeypatch):
    """Source/editable install: resolve next to this top-level module (repo root)."""
    monkeypatch.delenv("HERMES_WHATSAPP_BRIDGE_DIR", raising=False)
    # Force the packaged lookup to find nothing so we exercise the fallback.
    monkeypatch.setattr(hermes_constants, "_get_packaged_data_dir", lambda name: None)
    result = get_bundled_whatsapp_bridge_dir()
    expected = Path(hermes_constants.__file__).resolve().parent / "scripts" / "whatsapp-bridge"
    assert result == expected
    # In this repo checkout the bridge actually exists at the resolved path.
    assert (result / "bridge.js").is_file()


def test_caller_default_used_when_nothing_else(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_WHATSAPP_BRIDGE_DIR", raising=False)
    monkeypatch.setattr(hermes_constants, "_get_packaged_data_dir", lambda name: None)
    fallback = tmp_path / "explicit-default"
    assert get_bundled_whatsapp_bridge_dir(default=fallback) == fallback
