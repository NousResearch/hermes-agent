"""All YAML write paths use indented block sequences (#31999)."""

import pytest

import hermes_yaml as yaml
from utils import atomic_roundtrip_yaml_update, atomic_yaml_write


def test_atomic_write_then_key_update_keeps_layout_and_values(tmp_path):
    data = {"custom_providers": [{"name": "Tëst 🦀", "base_url": "https://example.com"}]}
    path = tmp_path / "config.yaml"
    atomic_yaml_write(path, data)
    initial = path.read_text(encoding="utf-8")
    atomic_roundtrip_yaml_update(path, "approvals.mode", "off")
    content = path.read_text(encoding="utf-8")
    assert "Tëst 🦀" in content
    assert not list(tmp_path.glob(".config_*.tmp"))
    assert yaml.roundtrip_yaml().load(content) == {**data, "approvals": {"mode": "off"}}
    assert content.startswith(initial)
    assert "\n  - " in content
    assert yaml.safe_load(content) == {**data, "approvals": {"mode": "off"}}


def test_failed_atomic_yaml_write_keeps_original(tmp_path):
    path = tmp_path / "config.yaml"
    original = "# keep original\nkey: value\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        atomic_yaml_write(path, {"object": object()})
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".config_*.tmp"))
