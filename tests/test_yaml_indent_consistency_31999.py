"""All YAML write paths use indented block sequences (#31999)."""

import io

import pytest

import hermes_yaml as yaml
from utils import atomic_roundtrip_yaml_update, atomic_yaml_write


def test_safe_dump_produces_indented_lists():
    data = {"custom_providers": [{"name": "NVIDIA", "base_url": "https://api.nvidia.com"}]}
    out = yaml.safe_dump(data)
    assert "\n  - " in out
    assert yaml.safe_load(out) == data


def test_safe_and_roundtrip_writers_use_the_same_layout():
    data = {"items": [{"key": "value1"}, {"key": "value2"}]}
    stream = io.StringIO()
    yaml.roundtrip_yaml().dump(data, stream)
    assert yaml.safe_dump(data, sort_keys=False) == stream.getvalue()


def test_atomic_write_then_key_update_keeps_layout_and_values(tmp_path):
    data = {"custom_providers": [{"name": "Test", "base_url": "https://example.com"}]}
    path = tmp_path / "config.yaml"
    atomic_yaml_write(path, data)
    initial = path.read_text(encoding="utf-8")
    atomic_roundtrip_yaml_update(path, "approvals.mode", "off")
    content = path.read_text(encoding="utf-8")
    assert content.startswith(initial)
    assert "\n  - " in content
    assert yaml.safe_load(content) == {**data, "approvals": {"mode": "off"}}


def test_atomic_yaml_write_preserves_unicode(tmp_path):
    path = tmp_path / "config.yaml"
    atomic_yaml_write(path, {"name": "Tëst Näme 🦀"})
    assert "Tëst Näme 🦀" in path.read_text(encoding="utf-8")


def test_atomic_yaml_write_is_atomic(tmp_path):
    path = tmp_path / "config.yaml"
    atomic_yaml_write(path, {"key": "value"})
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"key": "value"}
    assert not list(tmp_path.glob(".config_*.tmp"))


def test_failed_atomic_yaml_write_keeps_original(tmp_path):
    path = tmp_path / "config.yaml"
    original = "# keep original\nkey: value\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        atomic_yaml_write(path, {"object": object()})
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".config_*.tmp"))


def test_atomic_yaml_write_loads_in_roundtrip_editor(tmp_path):
    data = {
        "custom_providers": [
            {"name": "Provider A", "base_url": "https://a.example.com"},
            {"name": "Provider B", "base_url": "https://b.example.com"},
        ],
        "fallback_providers": ["backup1", "backup2"],
    }
    path = tmp_path / "config.yaml"
    atomic_yaml_write(path, data)
    assert yaml.roundtrip_yaml().load(path.read_text(encoding="utf-8")) == data
