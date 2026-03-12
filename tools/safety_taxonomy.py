from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyClassification:
    risk_class: str
    action_type: str
    approval_required: bool
    rollback_expected: bool


_READ_ONLY = {
    "read_file", "search_files", "web_search", "web_extract", "browser_snapshot", "session_search", "list_cronjobs"
}
_REVERSIBLE = {
    "patch", "write_file", "remove_cronjob", "memory", "todo"
}
_IRREVERSIBLE = {
    "terminal", "schedule_cronjob", "send_message", "mcp_call_tool", "skill_manage"
}


def classify_tool_action(tool_name: str) -> SafetyClassification:
    if tool_name in _READ_ONLY:
        return SafetyClassification("low", "read_only", False, False)
    if tool_name in _REVERSIBLE:
        return SafetyClassification("medium", "reversible_side_effect", True, True)
    if tool_name in _IRREVERSIBLE:
        return SafetyClassification("high", "irreversible_side_effect", True, False)
    if tool_name.startswith("browser_"):
        return SafetyClassification("medium", "external_interaction", False, False)
    return SafetyClassification("low", "unknown", False, False)
