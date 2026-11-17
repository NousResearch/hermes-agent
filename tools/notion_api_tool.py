"""notion_api — a thin, structured proxy over the `ntn` Notion CLI.

The model calls ``notion_api(path, method, body)`` with structured arguments and
gets parsed JSON back. It never writes shell, never writes python, never parses
JSON itself — which is what keeps quiet Slack channels silent (no interpreter
trips the security-scan approval prompt). Internals run a FIXED argv
(``ntn api <path> -X <METHOD>``) with the JSON body piped to the process's
stdin — ntn 0.15.x reads the request body from stdin with no flag — and
``shell=False``, so nothing the model supplies can become a shell token. ``ntn``
is wrapped (rather than calling api.notion.com directly) to reuse its
Markdown->Notion-blocks conversion (the ``markdown`` body field and the
``v1/pages/<id>/markdown`` endpoint).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = frozenset({"GET", "POST", "PATCH", "DELETE"})
_NTN_TIMEOUT_SECONDS = 30


NOTION_API_SCHEMA = {
    "name": "notion_api",
    "description": (
        "Call the Notion API and get parsed JSON back. Use this for ALL Notion "
        "work — never shell out, never use python/jq/curl for Notion. Pass the "
        "API path (e.g. 'v1/pages', 'v1/data_sources/<id>', "
        "'v1/data_sources/<id>/query', 'v1/pages/<id>', 'v1/pages/<id>/markdown'), "
        "an HTTP method, and an optional JSON body object. Markdown bodies are "
        "supported: include a 'markdown' field on a create body, or PATCH "
        "'v1/pages/<id>/markdown' with {'markdown': '...'} to append to a page body."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Notion API path beginning with 'v1/', e.g. 'v1/pages'.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PATCH", "DELETE"],
                "description": "HTTP method. Defaults to GET.",
                "default": "GET",
            },
            "body": {
                "type": "object",
                "description": "JSON body object for POST/PATCH (properties, filter, markdown, …). Omit for GET.",
            },
        },
        "required": ["path"],
    },
}


def _ntn_env() -> dict:
    """Build the subprocess env, encapsulating the per-shell Notion setup.

    ``ntn`` reads NOTION_API_TOKEN; the stored secret is NOTION_API_KEY. Bridge
    the two and disable the OS keyring so ``ntn`` runs non-interactively.
    Self-heal for delegation sub-agents / workers that never loaded
    ~/.hermes/.env (mirrors slack_react's token self-heal).
    """
    from agent.secret_scope import is_multiplex_active

    if is_multiplex_active():
        # The ntn child must carry THIS profile's token and none of the
        # default profile's secrets: os.environ minus the default's .env,
        # overlaid with the profile scope. No self-heal (agent.fork_secrets).
        from agent.fork_secrets import read_secret, scoped_subprocess_env

        env = scoped_subprocess_env()
        env.pop("NOTION_API_TOKEN", None)
        env.pop("NOTION_API_KEY", None)
        token = read_secret("NOTION_API_TOKEN", tool="notion_api") or read_secret(
            "NOTION_API_KEY", tool="notion_api"
        )
        if token:
            env["NOTION_API_TOKEN"] = token
        env.setdefault("NOTION_KEYRING", "0")
        return env

    env = dict(os.environ)
    if not env.get("NOTION_API_TOKEN") and not env.get("NOTION_API_KEY"):
        try:
            from hermes_cli.env_loader import load_hermes_dotenv
            load_hermes_dotenv()
            env = dict(os.environ)
        except Exception:
            pass
    if not env.get("NOTION_API_TOKEN") and env.get("NOTION_API_KEY"):
        env["NOTION_API_TOKEN"] = env["NOTION_API_KEY"]
    env.setdefault("NOTION_KEYRING", "0")
    return env


def _notion_api_handler(args: dict, **_kw) -> str:
    path = (args.get("path") or "").strip()
    method = (args.get("method") or "GET").strip().upper()
    body = args.get("body")

    if method not in _ALLOWED_METHODS:
        return tool_error(
            f"Unsupported method '{method}'. Use one of GET, POST, PATCH, DELETE."
        )
    if not path.startswith("v1/"):
        return tool_error(
            "path must be a Notion API path beginning with 'v1/' "
            "(e.g. 'v1/pages', 'v1/data_sources/<id>/query')."
        )
    if any(ch.isspace() for ch in path):
        return tool_error("path must not contain whitespace.")

    if not shutil.which("ntn"):
        return tool_error("Notion CLI 'ntn' is not installed on this host.")

    if body is not None:
        try:
            json.dumps(body)
        except (TypeError, ValueError) as e:
            return tool_error(f"body is not JSON-serializable: {e}")

    # Fork boundary (education design 5a): a profile with notion.write_guard
    # gets every non-read request checked against its rules before ntn runs.
    # Mirrored in the model_tools execution backstop. Inert when unguarded.
    from tools.notion_write_guard import guard_denial
    denial = guard_denial(method, path, body)
    if denial:
        return tool_error(denial)

    ok, data = _run_ntn(path, method, body)
    if not ok:
        return tool_error(str(data))
    if data is None:
        return tool_result({"ok": True, "data": None})
    if isinstance(data, str):
        # Non-JSON payload (e.g. a raw markdown body) — return it as text.
        return tool_result({"ok": True, "text": data})
    return tool_result({"ok": True, "data": data})


def _run_ntn(path: str, method: str, body) -> tuple[bool, object]:
    """Run ``ntn api <path> -X <method>`` (body JSON on stdin).

    Returns ``(True, parsed_json | raw_text | None)`` or ``(False, error)``.
    Shared with the write guard's target lookups (tools/notion_write_guard.py)
    so both use the same credentials path.
    """
    env = _ntn_env()
    if not env.get("NOTION_API_TOKEN"):
        return False, "Notion token not configured (set NOTION_API_KEY or NOTION_API_TOKEN)."

    argv = ["ntn", "api", path, "-X", method]
    stdin_data = None
    if body is not None:
        try:
            # ntn 0.15.x takes the request body from stdin JSON (no flag);
            # subprocess feeds this via input= below.
            stdin_data = json.dumps(body)
        except (TypeError, ValueError) as e:
            return False, f"body is not JSON-serializable: {e}"

    try:
        proc = subprocess.run(
            argv,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=_NTN_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"ntn timed out after {_NTN_TIMEOUT_SECONDS}s ({method} {path})."
    except Exception as e:  # pragma: no cover - defensive
        return False, f"ntn invocation failed: {e}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        detail = stderr or stdout or f"exit code {proc.returncode}"
        logger.warning("[notion_api] ntn failed (%s %s): %s", method, path, detail)
        return False, f"Notion API error: {detail}"

    if not stdout:
        return True, None
    try:
        return True, json.loads(stdout)
    except ValueError:
        return True, stdout


def _check_notion_api() -> bool:
    """Available whenever the ``ntn`` CLI is on PATH."""
    return bool(shutil.which("ntn"))


registry.register(
    name="notion_api",
    toolset="notion",
    schema=NOTION_API_SCHEMA,
    handler=_notion_api_handler,
    check_fn=_check_notion_api,
    requires_env=[],
    is_async=False,
    emoji="🗒️",
    max_result_size_chars=16000,
)
