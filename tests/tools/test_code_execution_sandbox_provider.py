"""execute_code sandbox config merge."""

from unittest.mock import patch

from tools.code_execution_tool import get_merged_sandbox_config


def test_get_merged_sandbox_config_prefers_code_execution():
    minimal_default = {
        "sandbox": {
            "type": "local",
            "image": "base:1",
            "profiles": {"default": {"cpu_quota": 2.0}},
        }
    }
    with patch("hermes_cli.config.DEFAULT_CONFIG", minimal_default):
        with patch("cli.CLI_CONFIG", {"sandbox": {"type": "docker"}}):
            with patch("tools.code_execution_tool._load_config", return_value={"sandbox": {"image": "ce:1"}}):
                m = get_merged_sandbox_config()
                assert m["type"] == "docker"
                assert m["image"] == "ce:1"
                assert m["profiles"]["default"]["cpu_quota"] == 2.0
