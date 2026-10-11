"""Presentation of the session.status RPC response."""
from hermes_cli.status_report import status_lines


def session_status_lines(fields: dict, info: dict, project: dict | None) -> list[str]:
    return [
        "Hermes TUI Status", "", *status_lines(fields, "session_id", "path"),
        *([f"Project: {project['name']}"] if project else []),
        *status_lines(fields, "title", "model"),
        f"Reasoning: {info.get('reasoning_effort') or 'default'}",
        f"Fast: {'Yes' if info.get('fast') else 'No'}",
        *status_lines(fields, "created", "last_activity", "tokens", "agent_running"),
    ]
