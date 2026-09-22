from dataclasses import replace

import pytest

from tools.quarantine_dev_context import (
    DevHumanApprovalContext,
    validate_dev_approval_context,
)


def safe_context() -> DevHumanApprovalContext:
    return DevHumanApprovalContext(
        environment="LAB",
        simulation_only=True,
        background=False,
        production_enforce=False,
        provider_execution_enabled=False,
        browser_execution_enabled=False,
        financial_execution_enabled=False,
        destructive_host_actions_enabled=False,
        active_production_config=False,
        active_skill_store_target=False,
    )


def test_default_context_fails_closed():
    result = validate_dev_approval_context(DevHumanApprovalContext())
    assert result.valid is False


def test_explicit_lab_simulation_context_is_valid():
    result = validate_dev_approval_context(safe_context())
    assert result.valid is True
    assert result.reason == "dev_context_valid"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("environment", "PROD", "dev_context_environment_invalid"),
        ("simulation_only", False, "dev_context_not_simulation_only"),
        ("background", True, "dev_context_background_forbidden"),
        ("production_enforce", True, "dev_context_production_enforce_forbidden"),
        ("provider_execution_enabled", True, "dev_context_provider_execution_forbidden"),
        ("browser_execution_enabled", True, "dev_context_browser_execution_forbidden"),
        ("financial_execution_enabled", True, "dev_context_financial_execution_forbidden"),
        ("destructive_host_actions_enabled", True, "dev_context_destructive_host_action_forbidden"),
        ("active_production_config", True, "dev_context_active_production_config_forbidden"),
        ("active_skill_store_target", True, "dev_context_active_skill_store_target_forbidden"),
    ],
)
def test_each_production_or_execution_tripwire_rejects(field, value, reason):
    result = validate_dev_approval_context(replace(safe_context(), **{field: value}))
    assert result.valid is False
    assert result.reason == reason


def test_test_environment_is_allowed_only_with_all_simulation_tripwires_safe():
    result = validate_dev_approval_context(replace(safe_context(), environment="TEST"))
    assert result.valid is True
