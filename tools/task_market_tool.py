"""TaskMarket tools — browse, claim, and complete tasks for USDC on Base Network."""

import json
import os
import subprocess
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)

# Default CLI path, overridable via env
TASKMARKET_CLI = os.environ.get(
    "TASKMARKET_CLI_PATH",
    os.path.expanduser("~/.hermes/node/bin/taskmarket")
)


def check_taskmarket_requirements() -> bool:
    """TaskMarket available if CLI is installed and keystore exists."""
    keystore = os.path.expanduser("~/.taskmarket/keystore.json")
    return os.path.isfile(TASKMARKET_CLI) and os.path.isfile(keystore)


def _run_cli(*args, timeout: int = 30) -> dict:
    """Run a taskmarket CLI command and return parsed JSON or error."""
    try:
        result = subprocess.run(
            [TASKMARKET_CLI] + list(args),
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error_text = result.stderr.strip() or output
            return {"error": error_text, "exit_code": result.returncode}
        # Try to parse as JSON
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"output": output}
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def task_search(status: str = "open", tags: str = "", reward_min: float = 0,
                limit: int = 10, task_id: str = None, **kwargs) -> str:
    """Search TaskMarket for available tasks."""
    args = ["task", "list", "--limit", str(limit)]
    if status:
        args.extend(["--status", status])
    if tags:
        args.extend(["--tags", tags])
    if reward_min > 0:
        args.extend(["--reward-min", str(reward_min)])
    return json.dumps(_run_cli(*args))


def task_get(task_id: str, **kwargs) -> str:
    """Get full details of a specific task."""
    return json.dumps(_run_cli("task", "get", task_id))


def task_claim(task_id: str, **kwargs) -> str:
    """Claim a task to work on it."""
    return json.dumps(_run_cli("task", "claim", task_id))


def task_pitch(task_id: str, text: str, duration_hours: int = 24, **kwargs) -> str:
    """Submit a pitch for a pitch-mode task."""
    return json.dumps(_run_cli("task", "pitch", task_id,
                               "--text", text, "--duration", str(duration_hours)))


def task_submit(task_id: str, file_path: str, **kwargs) -> str:
    """Submit completed work for a task."""
    expanded = os.path.expanduser(file_path)
    if not os.path.isfile(expanded):
        return json.dumps({"error": f"File not found: {file_path}"})
    return json.dumps(_run_cli("task", "submit", task_id, "--file", expanded, timeout=60))


def task_stats(**kwargs) -> str:
    """Get your agent stats — completed tasks, earnings, rating, skills."""
    return json.dumps(_run_cli("stats"))


def task_inbox(**kwargs) -> str:
    """Show tasks you created and tasks you are working on."""
    return json.dumps(_run_cli("inbox"))


# --- Schema definitions ---

TASK_SEARCH_SCHEMA = {
    "name": "task_search",
    "description": "Search TaskMarket for available tasks. Returns open tasks with rewards, descriptions, required skills, and deadlines. Use to find work opportunities.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["open", "claimed", "pending_approval", "completed", "expired"],
                "description": "Filter by task status (default: open)"
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated skill tags to filter by (e.g. 'python,research')"
            },
            "reward_min": {
                "type": "number",
                "description": "Minimum reward in USDC"
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default: 10)"
            },
        }
    }
}

TASK_GET_SCHEMA = {
    "name": "task_get",
    "description": "Get full details of a specific TaskMarket task including description, reward, deadline, status, and pending actions.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task ID to look up"},
        },
        "required": ["task_id"]
    }
}

TASK_CLAIM_SCHEMA = {
    "name": "task_claim",
    "description": "Claim a TaskMarket task to work on it. Only works for claim-mode tasks that are still open.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task ID to claim"},
        },
        "required": ["task_id"]
    }
}

TASK_PITCH_SCHEMA = {
    "name": "task_pitch",
    "description": "Submit a pitch for a pitch-mode TaskMarket task. Describe your approach and estimated duration.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task ID to pitch for"},
            "text": {"type": "string", "description": "Your pitch — describe your approach and qualifications"},
            "duration_hours": {"type": "integer", "description": "Estimated hours to complete (default: 24)"},
        },
        "required": ["task_id", "text"]
    }
}

TASK_SUBMIT_SCHEMA = {
    "name": "task_submit",
    "description": "Submit completed work for a TaskMarket task. Provide the path to your output file.",
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task ID"},
            "file_path": {"type": "string", "description": "Absolute path to the output file to submit"},
        },
        "required": ["task_id", "file_path"]
    }
}

TASK_STATS_SCHEMA = {
    "name": "task_stats",
    "description": "Get your TaskMarket agent stats — completed tasks, total earnings, average rating, and earned skill tags.",
    "parameters": {"type": "object", "properties": {}}
}

TASK_INBOX_SCHEMA = {
    "name": "task_inbox",
    "description": "Show your TaskMarket inbox — tasks you created and tasks you are currently working on.",
    "parameters": {"type": "object", "properties": {}}
}


# --- Register all tools ---

_ALL_TOOLS = [
    ("task_search", TASK_SEARCH_SCHEMA, lambda args, **kw: task_search(**args, **kw)),
    ("task_get", TASK_GET_SCHEMA, lambda args, **kw: task_get(**args, **kw)),
    ("task_claim", TASK_CLAIM_SCHEMA, lambda args, **kw: task_claim(**args, **kw)),
    ("task_pitch", TASK_PITCH_SCHEMA, lambda args, **kw: task_pitch(**args, **kw)),
    ("task_submit", TASK_SUBMIT_SCHEMA, lambda args, **kw: task_submit(**args, **kw)),
    ("task_stats", TASK_STATS_SCHEMA, lambda args, **kw: task_stats(**args, **kw)),
    ("task_inbox", TASK_INBOX_SCHEMA, lambda args, **kw: task_inbox(**args, **kw)),
]

for tool_name, schema, handler in _ALL_TOOLS:
    registry.register(
        name=tool_name,
        toolset="task_market",
        schema=schema,
        handler=handler,
        check_fn=check_taskmarket_requirements,
    )
