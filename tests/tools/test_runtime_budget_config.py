from unittest.mock import patch

from tools.budget_config import BudgetConfig, get_runtime_budget_config


def test_runtime_budget_uses_existing_defaults_when_unconfigured():
    with patch("hermes_cli.config.load_config", return_value={}):
        cfg = get_runtime_budget_config()
    assert cfg == BudgetConfig()


def test_runtime_budget_reads_user_config():
    user_config = {
        "tool_output": {
            "result_budget_chars": 4_000,
            "turn_budget_chars": 12_000,
            "preview_chars": 1_000,
            "tool_overrides": {"terminal": 8_000},
        }
    }
    with patch("hermes_cli.config.load_config", return_value=user_config):
        cfg = get_runtime_budget_config()
    assert cfg.default_result_size == 4_000
    assert cfg.turn_budget == 12_000
    assert cfg.preview_size == 1_000
    assert cfg.tool_overrides == {"terminal": 8_000}


def test_runtime_budget_rejects_invalid_values():
    user_config = {
        "tool_output": {
            "result_budget_chars": -1,
            "turn_budget_chars": "bad",
            "preview_chars": 0,
            "tool_overrides": {"terminal": -2, "web_search": "5000"},
        }
    }
    with patch("hermes_cli.config.load_config", return_value=user_config):
        cfg = get_runtime_budget_config()
    assert cfg == BudgetConfig(tool_overrides={"web_search": 5_000})
