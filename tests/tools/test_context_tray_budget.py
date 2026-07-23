from pathlib import Path

import yaml

from tools import budget_config


def test_context_tray_defaults_bound_first_send(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    assert hasattr(budget_config, "load_budget_config")
    budget = budget_config.load_budget_config(context_length=1_000_000)

    assert budget_config.DEFAULT_RESULT_SIZE_CHARS == 32_768
    assert budget_config.DEFAULT_TURN_BUDGET_CHARS == 65_536
    assert budget.default_result_size == 32_768
    assert budget.turn_budget == 65_536


def test_context_tray_reads_config_yaml_without_env_toggles(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "tool_results": {
                    "threshold_chars": 24_000,
                    "turn_budget_chars": 48_000,
                    "preview_chars": 2_000,
                    "tool_overrides": {"web_extract": 20_000},
                }
            }
        ),
        encoding="utf-8",
    )

    assert hasattr(budget_config, "load_budget_config")
    budget = budget_config.load_budget_config(context_length=1_000_000)

    assert budget.default_result_size == 24_000
    assert budget.turn_budget == 48_000
    assert budget.preview_size == 2_000
    assert budget.tool_overrides["web_extract"] == 20_000
    assert budget.resolve_threshold("read_file") == float("inf")