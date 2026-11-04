"""Fail-closed context guard for temporary LAB human confirmation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DevHumanApprovalContext:
    environment: str = ""
    simulation_only: bool = False
    background: bool = True
    production_enforce: bool = True
    provider_execution_enabled: bool = True
    browser_execution_enabled: bool = True
    financial_execution_enabled: bool = True
    destructive_host_actions_enabled: bool = True
    active_production_config: bool = True
    active_skill_store_target: bool = True


@dataclass(frozen=True)
class ContextValidation:
    valid: bool
    reason: str


def validate_dev_approval_context(context: DevHumanApprovalContext) -> ContextValidation:
    if context.environment not in {"LAB", "TEST"}:
        return ContextValidation(False, "dev_context_environment_invalid")
    if not context.simulation_only:
        return ContextValidation(False, "dev_context_not_simulation_only")
    if context.background:
        return ContextValidation(False, "dev_context_background_forbidden")
    if context.production_enforce:
        return ContextValidation(False, "dev_context_production_enforce_forbidden")
    if context.provider_execution_enabled:
        return ContextValidation(False, "dev_context_provider_execution_forbidden")
    if context.browser_execution_enabled:
        return ContextValidation(False, "dev_context_browser_execution_forbidden")
    if context.financial_execution_enabled:
        return ContextValidation(False, "dev_context_financial_execution_forbidden")
    if context.destructive_host_actions_enabled:
        return ContextValidation(False, "dev_context_destructive_host_action_forbidden")
    if context.active_production_config:
        return ContextValidation(False, "dev_context_active_production_config_forbidden")
    if context.active_skill_store_target:
        return ContextValidation(False, "dev_context_active_skill_store_target_forbidden")
    return ContextValidation(True, "dev_context_valid")
