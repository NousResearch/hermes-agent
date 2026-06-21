import pytest
from unittest.mock import patch


def _load_config():
    """Re-import-safe accessor for tests."""
    from tools.delegate_tool import _load_config as _lc
    return _lc()


class TestLoadConfigMerge:
    """Tests for _load_config() merge behavior (fixes #50199)."""

    def test_in_mem_complete_wins_over_disk(self):
        """Fully-populated in-memory block should be used as-is."""
        in_mem = {"base_url": "http://mem:8090/v1", "model": "mem-model"}
        on_disk = {"base_url": "http://disk:8090/v1", "model": "disk-model"}
        with patch("cli.CLI_CONFIG", {"delegation": in_mem}), \
             patch("hermes_cli.config.load_config", return_value={"delegation": on_disk}):
            cfg = _load_config()
        assert cfg["base_url"] == "http://mem:8090/v1"
        assert cfg["model"] == "mem-model"

    def test_in_mem_empty_base_url_filled_from_disk(self):
        """The #50199 scenario: in-mem has empty base_url, disk has the real one."""
        in_mem = {"base_url": "", "model": "mem-model"}
        on_disk = {"base_url": "http://worker:8090/v1", "model": "disk-model"}
        with patch("cli.CLI_CONFIG", {"delegation": in_mem}), \
             patch("hermes_cli.config.load_config", return_value={"delegation": on_disk}):
            cfg = _load_config()
        assert cfg["base_url"] == "http://worker:8090/v1"  # filled from disk
        assert cfg["model"] == "mem-model"                 # in-mem wins

    def test_in_mem_missing_base_url_filled_from_disk(self):
        """In-mem block omits base_url entirely; disk supplies it."""
        in_mem = {"model": "mem-model"}
        on_disk = {"base_url": "http://worker:8090/v1", "model": "disk-model"}
        with patch("cli.CLI_CONFIG", {"delegation": in_mem}), \
             patch("hermes_cli.config.load_config", return_value={"delegation": on_disk}):
            cfg = _load_config()
        assert cfg["base_url"] == "http://worker:8090/v1"
        assert cfg["model"] == "mem-model"

    def test_whitespace_base_url_treated_as_empty(self):
        """Whitespace-only base_url should also fall back to disk."""
        in_mem = {"base_url": "   ", "model": "mem-model"}
        on_disk = {"base_url": "http://worker:8090/v1"}
        with patch("cli.CLI_CONFIG", {"delegation": in_mem}), \
             patch("hermes_cli.config.load_config", return_value={"delegation": on_disk}):
            cfg = _load_config()
        assert cfg["base_url"] == "http://worker:8090/v1"

    def test_in_mem_empty_dict_falls_back_to_disk(self):
        """Legacy behavior: empty in-mem block reads disk entirely."""
        on_disk = {"base_url": "http://worker:8090/v1", "model": "disk-model"}
        with patch("cli.CLI_CONFIG", {"delegation": {}}), \
             patch("hermes_cli.config.load_config", return_value={"delegation": on_disk}):
            cfg = _load_config()
        assert cfg == on_disk

    def test_both_empty_returns_empty_dict(self):
        """No config anywhere -> empty dict (not None)."""
        with patch("cli.CLI_CONFIG", {}), \
             patch("hermes_cli.config.load_config", return_value={}):
            cfg = _load_config()
        assert cfg == {}

    def test_cli_config_import_failure_falls_back_to_disk(self):
        """If `from cli import CLI_CONFIG` raises, disk path still works."""
        on_disk = {"base_url": "http://worker:8090/v1"}
        import builtins
        real_import = builtins.__import__

        def _fail_cli(name, *args, **kwargs):
            if name == "cli":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fail_cli), \
             patch("hermes_cli.config.load_config", return_value={"delegation": on_disk}):
            cfg = _load_config()
        assert cfg["base_url"] == "http://worker:8090/v1"

    def test_non_dict_delegation_block_treated_as_empty(self):
        """Defensive: if CLI_CONFIG['delegation'] is a string/list, ignore it."""
        on_disk = {"base_url": "http://worker:8090/v1"}
        with patch("cli.CLI_CONFIG", {"delegation": "not-a-dict"}), \
             patch("hermes_cli.config.load_config", return_value={"delegation": on_disk}):
            cfg = _load_config()
        assert cfg == on_disk
