"""Native Plannotator integration for interactive review and annotation.

Use this tool when a user wants browser-based feedback on a code diff or a
markdown/text artifact. It launches a Plannotator session, shares the live URL,
and can wait for browser-submitted feedback so Hermes can continue with the
review result in the same workflow.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tools.exposure_helpers import run_command_template
from tools.registry import registry

logger = logging.getLogger(__name__)

_ENV_TEMPLATE_BY_ACTION = {
    "prepare": "HERMES_PLANNOTATOR_PREPARE_TEMPLATE",
    "review": "HERMES_PLANNOTATOR_REVIEW_TEMPLATE",
    "annotate": "HERMES_PLANNOTATOR_ANNOTATE_TEMPLATE",
    "last": "HERMES_PLANNOTATOR_LAST_TEMPLATE",
}

_INLINE_ACTION_MAP = {
    "inline_review": "review",
    "inline_annotate": "annotate",
    "inline_last": "last",
}

_DEFAULT_LAUNCH_TIMEOUT_SECONDS = 120
_DEFAULT_COMPLETION_TIMEOUT_SECONDS = 3600
_DEFAULT_POLL_INTERVAL_SECONDS = 2.0
_MAX_LOG_BYTES = 128_000
_PLANNOTATOR_HOST_ENV = "PLANNOTATOR_HOST"

_PLANNOTATOR_SCHEMA = {
    "name": "plannotator_session",
    "description": (
        "Open an interactive Plannotator review or annotation session for code changes or markdown files. "
        "Use this when the user should inspect a live browser UI, add comments or replacements, and send feedback back to Hermes. "
        "Inline actions can post the review URL immediately in the active conversation, then wait for the submitted feedback before returning."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["prepare", "review", "annotate", "inline_review", "inline_annotate", "last", "inline_last"],
                "description": "prepare: reserve/generate a host+URL before launching. review: start a review session. annotate: start a markdown annotation session. inline_review / inline_annotate / inline_last: post the URL into the active conversation, then wait for feedback. last: last-message flow if the launcher supports it, with Hermes fallback to the latest assistant message transcript when needed."
            },
            "review_target": {
                "type": "string",
                "description": "Optional PR/MR URL or other review target for review/inline_review. Omit to review the current local diff if the configured launcher supports it."
            },
            "artifact_path": {
                "type": "string",
                "description": "Absolute path to a markdown artifact when action='annotate' or 'inline_annotate'."
            },
            "repo_path": {
                "type": "string",
                "description": "Optional working directory for review launches against the current local diff."
            },
            "fixed_host": {
                "type": "string",
                "description": "Optional fixed host to use for prepare/review/annotate/inline actions, e.g. 'review-abc123.example.com'. Inline actions prepare and launch using the same host."
            },
            "exposure_strategy": {
                "type": "string",
                "enum": ["auto", "localhost", "reverse-proxy", "tailscale-serve", "tailscale-funnel"],
                "description": "Hint passed through to the launcher template so one Plannotator launcher can support multiple exposure backends."
            },
            "command_template": {
                "type": "string",
                "description": "Optional one-off shell template override. Available placeholders: {artifact_path}, {review_target}, {review_target_arg}, {exposure_strategy}, {repo_path}."
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Launcher timeout in seconds for the initial bridge start. Default 120."
            },
            "wait_for_completion": {
                "type": "boolean",
                "description": "If true (default for review/annotate and always true for inline actions), wait synchronously for the launched Plannotator session to finish before returning."
            },
            "completion_timeout_seconds": {
                "type": "integer",
                "description": "How long to wait for the Plannotator session to complete when wait_for_completion=true. Default 3600 (60 minutes)."
            },
            "poll_interval_seconds": {
                "type": "number",
                "description": "Polling interval while waiting for session completion. Default 2 seconds."
            }
        },
        "required": ["action"]
    }
}


def plannotator_session_tool(args: dict[str, Any], **_kw) -> str:
    action = (args.get("action") or "").strip().lower()
    if action in _INLINE_ACTION_MAP:
        return _execute_inline_flow(args)
    if action not in _ENV_TEMPLATE_BY_ACTION:
        return json.dumps({"error": f"Unsupported action: {action}"})

    normalized = _normalize_args(args)
    validation_error = _validate_args(normalized)
    if validation_error:
        return json.dumps({"error": validation_error})

    result = _launch_plannotator(normalized)
    return json.dumps(result)


def _execute_inline_flow(args: dict[str, Any]) -> str:
    normalized = _normalize_args(args)
    requested_action = normalized["action"]
    base_action = _INLINE_ACTION_MAP[requested_action]
    normalized["action"] = base_action
    normalized["wait_for_completion"] = True

    validation_error = _validate_args(normalized)
    if validation_error:
        return json.dumps({"error": validation_error})

    if base_action == "last":
        fallback = _build_last_message_fallback_args(normalized)
        if fallback:
            normalized = fallback
            normalized["wait_for_completion"] = True

    prepared = _launch_plannotator(
        {
            **normalized,
            "action": "prepare",
            "wait_for_completion": False,
        }
    )
    if not prepared.get("success"):
        return json.dumps(prepared)

    send_result = _send_inline_url_message(prepared["suggested_message"])
    if send_result.get("error"):
        return json.dumps(
            {
                "error": f"Failed to send prepared Plannotator URL message: {send_result['error']}",
                "prepared_url": prepared.get("url"),
                "prepared_host": prepared.get("host"),
                "send_message_result": send_result,
            }
        )

    launched = _launch_plannotator(
        {
            **normalized,
            "fixed_host": prepared.get("host") or normalized.get("fixed_host"),
            "wait_for_completion": True,
        }
    )
    if not launched.get("success"):
        launched["prepared_url"] = prepared.get("url")
        launched["prepared_host"] = prepared.get("host")
        launched["send_message_result"] = send_result
        return json.dumps(launched)

    launched["inline_message_sent"] = True
    launched["prepared_url"] = prepared.get("url")
    launched["prepared_host"] = prepared.get("host")
    launched["send_message_result"] = send_result
    return json.dumps(launched)


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    action = (args.get("action") or "").strip().lower()
    normalized = {
        "action": action,
        "artifact_path": (args.get("artifact_path") or "").strip(),
        "review_target": (args.get("review_target") or "").strip(),
        "repo_path": (args.get("repo_path") or "").strip() or None,
        "fixed_host": (args.get("fixed_host") or "").strip() or None,
        "exposure_strategy": (args.get("exposure_strategy") or "auto").strip().lower() or "auto",
        "command_template": args.get("command_template"),
        "launch_timeout": int(args.get("timeout_seconds") or _DEFAULT_LAUNCH_TIMEOUT_SECONDS),
        "wait_for_completion": args.get("wait_for_completion"),
        "completion_timeout": int(args.get("completion_timeout_seconds") or _DEFAULT_COMPLETION_TIMEOUT_SECONDS),
        "poll_interval": max(0.25, float(args.get("poll_interval_seconds") or _DEFAULT_POLL_INTERVAL_SECONDS)),
    }
    if normalized["wait_for_completion"] is None:
        normalized["wait_for_completion"] = normalized["action"] in {"review", "annotate"}
    return normalized


def _validate_args(args: dict[str, Any]) -> str | None:
    action = args["action"]
    artifact_path = args["artifact_path"]
    if action == "annotate" and not artifact_path:
        return "'artifact_path' is required when action='annotate'"
    if action == "annotate" and not os.path.isabs(artifact_path):
        return "'artifact_path' must be an absolute path"
    return None


def _build_last_message_fallback_args(args: dict[str, Any]) -> dict[str, Any] | None:
    last_message = _get_last_assistant_message_from_gateway_session()
    if not last_message:
        return None

    artifact_path = _write_last_message_markdown(last_message)
    fallback = dict(args)
    fallback["action"] = "annotate"
    fallback["artifact_path"] = artifact_path
    fallback["command_template"] = os.getenv(_ENV_TEMPLATE_BY_ACTION["annotate"], "").strip() or None
    return fallback


def _get_last_assistant_message_from_gateway_session() -> str | None:
    session_id = _resolve_current_gateway_session_id()
    if not session_id:
        return None

    sessions_dir = _get_gateway_sessions_dir()
    candidates = [
        sessions_dir / f"session_{session_id}.json",
        sessions_dir / f"{session_id}.jsonl",
    ]
    for path in candidates:
        last_assistant = _read_last_assistant_message(path)
        if last_assistant:
            return last_assistant
    return None


def _read_last_assistant_message(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.suffix == ".jsonl":
        return _read_last_assistant_message_from_jsonl(path)
    if path.suffix == ".json":
        return _read_last_assistant_message_from_session_json(path)
    return None


def _read_last_assistant_message_from_jsonl(path: Path) -> str | None:
    last_assistant = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    message = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                content = _message_text(message)
                if message.get("role") == "assistant" and content:
                    last_assistant = content
    except OSError:
        return None
    return last_assistant


def _read_last_assistant_message_from_session_json(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None

    last_assistant = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = _message_text(message)
        if message.get("role") == "assistant" and content:
            last_assistant = content
    return last_assistant


def _message_text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
    return None


def _resolve_current_gateway_session_id() -> str | None:
    sessions_file = _get_gateway_sessions_dir() / "sessions.json"
    if not sessions_file.exists():
        return None

    session_key = _build_current_gateway_session_key()
    if not session_key:
        return None

    try:
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    entry = data.get(session_key)
    if isinstance(entry, dict):
        return entry.get("session_id")
    return None


def _build_current_gateway_session_key() -> str | None:
    platform_name = (os.getenv("HERMES_SESSION_PLATFORM") or "").strip().lower()
    chat_id = (os.getenv("HERMES_SESSION_CHAT_ID") or "").strip()
    thread_id = (os.getenv("HERMES_SESSION_THREAD_ID") or "").strip() or None
    if not platform_name or not chat_id:
        return None

    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key

    source = SessionSource(platform=Platform(platform_name), chat_id=chat_id, chat_type="dm", thread_id=thread_id)
    return build_session_key(source)


def _get_gateway_sessions_dir() -> Path:
    hermes_home = Path(os.getenv("HERMES_HOME") or (Path.home() / ".hermes"))
    return hermes_home / "sessions"


def _write_last_message_markdown(content: str) -> str:
    platform_name = (os.getenv("HERMES_SESSION_PLATFORM") or "session").strip().lower() or "session"
    chat_id = (os.getenv("HERMES_SESSION_CHAT_ID") or "chat").strip() or "chat"
    thread_id = (os.getenv("HERMES_SESSION_THREAD_ID") or "thread").strip() or "thread"
    key = f"{platform_name}-{chat_id}-{thread_id}"
    safe_key = ''.join(ch if ch.isalnum() else '-' for ch in key)[:80]
    path = Path(tempfile.gettempdir()) / f"plannotator-last-{safe_key}.md"
    path.write_text(f"# Last assistant message\n\n{content}\n", encoding="utf-8")
    return str(path)


def _launch_plannotator(args: dict[str, Any]) -> dict[str, Any]:
    action = args["action"]
    template = _resolve_template(action, args.get("command_template"))
    if not template:
        env_name = _ENV_TEMPLATE_BY_ACTION[action]
        return {
            "error": (
                f"No Plannotator launcher template configured for action '{action}'. "
                f"Set {env_name} or pass command_template directly."
            )
        }

    review_target_arg = f" {args['review_target']}" if args["review_target"] else ""
    variables = {
        "artifact_path": args["artifact_path"],
        "review_target": args["review_target"],
        "review_target_arg": review_target_arg,
        "exposure_strategy": args["exposure_strategy"],
        "repo_path": args["repo_path"] or "",
    }
    child_env = {"PLANNOTATOR_EXPOSURE_STRATEGY": args["exposure_strategy"]}
    if args.get("fixed_host"):
        child_env[_PLANNOTATOR_HOST_ENV] = args["fixed_host"]

    try:
        execution = run_command_template(
            template,
            variables=variables,
            cwd=args["repo_path"],
            timeout=args["launch_timeout"],
            env=child_env,
        )
    except KeyError as exc:
        return {"error": f"Command template references unknown placeholder: {exc}"}
    except Exception as exc:
        logger.exception("plannotator_session failed")
        return {"error": f"Failed to launch Plannotator: {type(exc).__name__}: {exc}"}

    if execution["exit_code"] != 0:
        return {
            "error": f"Plannotator launcher failed with exit code {execution['exit_code']}",
            "command": execution["command"],
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
        }

    if not execution["url"]:
        return {
            "error": "Plannotator launcher succeeded but did not report a URL.",
            "command": execution["command"],
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
        }

    host = execution["parsed"].get("HOST") or _host_from_url(execution["url"])
    result = {
        "success": True,
        "action": action,
        "host": host,
        "url": execution["url"],
        "pid": execution["pid"],
        "log": execution["log"],
        "command": execution["command"],
        "stdout": execution["stdout"],
        "stderr": execution["stderr"],
        "exposure_strategy": args["exposure_strategy"],
        "suggested_message": _build_suggested_message(execution["url"]),
        "waited_for_completion": False,
    }

    if not args["wait_for_completion"]:
        return result

    wait_result = _wait_for_plannotator_completion(
        pid=execution.get("pid"),
        log_path=execution.get("log"),
        timeout_seconds=args["completion_timeout"],
        poll_interval_seconds=args["poll_interval"],
    )
    result.update(wait_result)
    result["waited_for_completion"] = True
    return result


def _send_inline_url_message(message: str) -> dict[str, Any]:
    try:
        from tools.send_message_tool import send_message_tool

        raw = send_message_tool({
            "action": "send",
            "target": "origin",
            "message": f"{message}\n\nI’ll wait here for your annotations and report back once they arrive.",
            "reply_to_current": True,
        })
    except Exception as exc:
        logger.exception("plannotator inline send_message failed")
        return {"error": f"send_message raised {type(exc).__name__}: {exc}"}

    try:
        parsed = json.loads(raw)
    except Exception:
        return {"error": f"send_message returned non-JSON output: {raw!r}"}
    return parsed


def _resolve_template(action: str, template_override: str | None) -> str:
    override = (template_override or "").strip()
    if override:
        return override
    env_name = _ENV_TEMPLATE_BY_ACTION[action]
    return os.getenv(env_name, "").strip()


def _host_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc or None


def _build_suggested_message(url: str) -> str:
    return (
        f"Temporary review URL:\n{url}\n\n"
        "What to do\n"
        "- open the link\n"
        "- add comments / replacements\n"
        "- press Send Annotations when done"
    )


def _wait_for_plannotator_completion(
    *,
    pid: str | int | None,
    log_path: str | None,
    timeout_seconds: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    pid_int = _coerce_pid(pid)
    started = time.monotonic()
    last_log_content = _read_log_excerpt(log_path)

    if pid_int is None and not log_path:
        return {
            "completed": False,
            "timed_out": False,
            "status": "not_waitable",
            "final_log": None,
            "message": "Plannotator session could not be waited on because the launcher did not report a PID or log path.",
        }

    while True:
        elapsed = time.monotonic() - started
        process_alive = _pid_is_running(pid_int) if pid_int is not None else None
        current_log = _read_log_excerpt(log_path)
        if current_log is not None:
            last_log_content = current_log

        if process_alive is False:
            final_log = _read_log_excerpt(log_path)
            if final_log is not None:
                last_log_content = final_log
            feedback_fields = _extract_plannotator_feedback_fields(last_log_content)
            return {
                "completed": True,
                "timed_out": False,
                "status": "completed",
                "final_log": last_log_content,
                "message": "Plannotator session completed.",
                "elapsed_seconds": round(elapsed, 2),
                **feedback_fields,
            }

        if elapsed >= timeout_seconds:
            feedback_fields = _extract_plannotator_feedback_fields(last_log_content)
            return {
                "completed": False,
                "timed_out": True,
                "status": "timeout",
                "final_log": last_log_content,
                "message": f"Timed out waiting for Plannotator session after {timeout_seconds} seconds.",
                "elapsed_seconds": round(elapsed, 2),
                "session_still_running": bool(process_alive),
                **feedback_fields,
            }

        time.sleep(poll_interval_seconds)


def _extract_plannotator_feedback_fields(log_text: str | None) -> dict[str, Any]:
    feedback_block = _extract_feedback_block(log_text)
    has_feedback = bool(feedback_block)
    fields: dict[str, Any] = {
        "feedback_detected": has_feedback,
        "feedback_markdown": feedback_block,
    }
    if has_feedback:
        fields["next_step_instruction"] = (
            "Treat feedback_markdown as the user's latest feedback. Incorporate it into the work and continue the task; do not stop at a summary unless the feedback explicitly asks only for a summary."
        )
    return fields


def _extract_feedback_block(log_text: str | None) -> str | None:
    if not log_text:
        return None
    marker = "# File Feedback"
    start = log_text.find(marker)
    if start == -1:
        return None
    feedback = log_text[start:].strip()
    bridge_marker = "\n[bridge]"
    end = feedback.find(bridge_marker)
    if end != -1:
        feedback = feedback[:end].rstrip()
    return feedback or None


def _coerce_pid(pid: str | int | None) -> int | None:
    if pid in (None, ""):
        return None
    try:
        return int(pid)
    except (TypeError, ValueError):
        return None


def _pid_is_running(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_log_excerpt(log_path: str | None) -> str | None:
    if not log_path:
        return None
    path = Path(log_path).expanduser()
    try:
        if not path.exists():
            return None
        data = path.read_bytes()
    except OSError:
        return None

    if len(data) > _MAX_LOG_BYTES:
        data = data[-_MAX_LOG_BYTES:]

    return data.decode("utf-8", errors="replace")


registry.register(
    name="plannotator_session",
    toolset="plannotator",
    schema=_PLANNOTATOR_SCHEMA,
    handler=plannotator_session_tool,
    emoji="📝",
)
