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
