"""Regression tests for issue #31999.

All YAML config write paths must produce 2-space-indented list items
(matching ruamel.yaml's layout).  Mixing 0-indent (default PyYAML) and
2-indent (ruamel.yaml) in the same config.yaml produces a file that
stricter parsers like js-yaml reject with "bad indentation of a mapping
entry", silently dropping custom_providers and breaking model switching.
"""

import yaml
from utils import IndentDumper, atomic_yaml_write


class TestIndentDumperShape:
    """IndentDumper emits 2-space-indented list items under mapping keys."""

    def test_indent_dumper_produces_2_indent_lists(self):
        """List items under a mapping key must start at column 2, not 0."""
        data = {
            "custom_providers": [
                {"name": "NVIDIA", "base_url": "https://api.nvidia.com"},
            ],
        }
        out = yaml.dump(data, Dumper=IndentDumper, default_flow_style=False)
        # The list item should be indented 2 spaces under the key
        assert "  - " in out, f"Expected 2-indent list, got:\n{out}"

    def test_default_pyyaml_produces_0_indent_lists(self):
        """Default PyYAML (the buggy baseline) emits 0-indent lists."""
        data = {
            "custom_providers": [
                {"name": "NVIDIA", "base_url": "https://api.nvidia.com"},
            ],
        }
        out = yaml.dump(data, default_flow_style=False)
        # The list item should be at column 0 (no leading spaces)
        lines = out.strip().split("\n")
        list_lines = [l for l in lines if l.lstrip().startswith("- ")]
        assert all(not l.startswith("  - ") for l in list_lines), \
            f"Expected 0-indent list (buggy baseline), got:\n{out}"

    def test_indent_dumper_matches_ruamel_layout(self):
        """IndentDumper output should match ruamel.yaml's list-under-mapping layout."""
        data = {
            "items": [
                {"key": "value1"},
                {"key": "value2"},
            ],
        }
        pyyaml_out = yaml.dump(data, Dumper=IndentDumper, default_flow_style=False)
        # ruamel.yaml with indent(mapping=2, sequence=4, offset=2) produces:
        #   items:
        #     - key: value1
        #     - key: value2
        # The key check: list items are NOT at column 0
        lines = pyyaml_out.strip().split("\n")
        list_lines = [l for l in lines if l.lstrip().startswith("- ")]
        assert all(l.startswith("  - ") for l in list_lines), \
            f"List items not 2-indent:\n{pyyaml_out}"


class TestAtomicYamlWriteUsesIndentDumper:
    """atomic_yaml_write must produce 2-indent lists via IndentDumper."""

    def test_atomic_yaml_write_produces_2_indent_lists(self, tmp_path):
        """The file written by atomic_yaml_write must have 2-indent list items."""
        data = {
            "custom_providers": [
                {"name": "Test", "base_url": "https://example.com"},
            ],
        }
        path = tmp_path / "config.yaml"
        atomic_yaml_write(path, data)

        content = path.read_text(encoding="utf-8")
        assert "  - " in content, \
            f"Expected 2-indent list in file, got:\n{content}"

    def test_atomic_yaml_write_preserves_unicode(self, tmp_path):
        """allow_unicode=True should write real UTF-8, not escape sequences."""
        data = {"name": "Tëst Näme"}
        path = tmp_path / "config.yaml"
        atomic_yaml_write(path, data)

        content = path.read_text(encoding="utf-8")
        assert "Tëst Näme" in content

    def test_atomic_yaml_write_is_atomic(self, tmp_path):
        """atomic_yaml_write should create the file and clean up temp files."""
        data = {"key": "value"}
        path = tmp_path / "config.yaml"
        atomic_yaml_write(path, data)

        assert path.exists()
        assert path.read_text(encoding="utf-8").strip().endswith("value")
        # No leftover temp files
        temp_files = list(tmp_path.glob(".config_*.tmp"))
        assert len(temp_files) == 0


class TestRoundtripConsistency:
    """Output of atomic_yaml_write should round-trip through ruamel.yaml."""

    def test_pyyaml_output_loads_in_ruamel(self, tmp_path):
        """File written by atomic_yaml_write should load in ruamel.yaml without errors."""
        data = {
            "custom_providers": [
                {"name": "Provider A", "base_url": "https://a.example.com"},
                {"name": "Provider B", "base_url": "https://b.example.com"},
            ],
            "fallback_providers": ["backup1", "backup2"],
        }
        path = tmp_path / "config.yaml"
        atomic_yaml_write(path, data)

        from ruamel.yaml import YAML
        yaml_rt = YAML(typ="rt")
        loaded = yaml_rt.load(path.read_text(encoding="utf-8"))
        assert loaded["custom_providers"][0]["name"] == "Provider A"
        assert loaded["fallback_providers"] == ["backup1", "backup2"]
