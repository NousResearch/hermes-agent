"""Session status presentation payload; profile and live-agent resolution stay in the RPC."""

from hermes_cli.status_report import status_lines


def session_status_result(fields: dict, usage: dict, project: dict | None) -> dict:
    lines = [
        "Hermes TUI Status", "", *status_lines(fields, "session_id", "path"),
        *([f"Project: {project['name']}"] if project else []),
        *status_lines(fields, "title", "model", "created", "last_activity", "tokens", "agent_running"),
    ]
    return {
        "output": "\n".join(lines),
        "details": {**fields, "tokens": int(usage.get("total") or 0),
                    "project": project["name"] if project else ""},
    }
