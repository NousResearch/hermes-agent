"""Mandatory, fail-closed admission immediately before an MCP transport call.

This is separate from optional ``pre_tool_call`` and middleware. It is reached by
registry tools, deferred tool-search calls, concurrent workers, and plugin direct
MCP calls because all of those end at the native MCP handler. The callback may
canonicalize arguments, but only a current, admitted gateway origin (or a
separately allowlisted scheduler occurrence) can satisfy a required policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
from typing import Any

logger = logging.getLogger("tools.mcp_tool")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CALL_IDS = ("task_id", "session_id", "tool_call_id", "turn_id", "api_request_id")
_DEFAULT_TIMEOUT_SECONDS = 2.0
_MAX_ARGS_BYTES = 1024 * 1024
_ORIGIN_ARG_KEYS = frozenset({
    "slack", "actor_slack_id", "slack_user_id", "slack_channel_id", "slack_workspace_id",
    "slack_message_ts", "slack_thread_ts", "channel_id", "workspace_id", "message_ts", "thread_ts",
})


def _json_object(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("MCP arguments must be a JSON object")
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
    if len(encoded) > _MAX_ARGS_BYTES:
        raise ValueError("MCP arguments exceed guard limit")
    return json.loads(encoded)


def _has_origin_fields(value: Any) -> bool:
    """A scheduler call cannot present Slack-like identity in its arguments."""
    if isinstance(value, dict):
        return any(
            key.lower().replace("-", "_") in _ORIGIN_ARG_KEYS or _has_origin_fields(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_origin_fields(item) for item in value)
    return False


def _policy(server: str, profile: str) -> dict | None:
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly()
    mcp = cfg.get("mcp") if isinstance(cfg, dict) else None
    rows = mcp.get("required_dispatch_guards", []) if isinstance(mcp, dict) else []
    if not isinstance(rows, list):
        raise ValueError("mcp.required_dispatch_guards must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("server") == server
               and row.get("profile") == profile]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("duplicate required MCP dispatch policy")
    row = matches[0]
    if (not isinstance(row.get("plugin"), str) or not row["plugin"]
            or not isinstance(row.get("platform"), str) or not row["platform"]):
        raise ValueError("invalid required MCP dispatch policy")
    cron = row.get("cron", {})
    if not isinstance(cron, dict):
        raise ValueError("invalid required MCP cron policy")
    hashes, names = cron.get("job_id_sha256", []), cron.get("read_only_tools", [])
    if (not isinstance(hashes, list) or not isinstance(names, list)
            or not all(isinstance(h, str) and _SHA256.fullmatch(h) for h in hashes)
            or not all(isinstance(n, str) and n for n in names)):
        raise ValueError("invalid required MCP cron allowlist")
    return row


def _scheduler_allowed(policy: dict, profile: str, tool_name: str) -> bool:
    from cron.execution_identity import current_cron_execution
    from tools.mcp_tool_handlers import _tool_is_read_only

    cron = current_cron_execution()
    allowed = policy.get("cron", {})
    if (cron is None or cron.profile != profile or cron.source != "builtin"
            or not cron.scheduled_instant or not cron.execution_id
            or tool_name not in allowed.get("read_only_tools", ())):
        return False
    digest = hashlib.sha256(cron.job_id.encode("utf-8")).hexdigest()
    return digest in allowed.get("job_id_sha256", ()) and _tool_is_read_only(policy["server"], tool_name)


def _run_callback(callback, *, server: str, tool_name: str, args: dict,
                  call_ids: dict, timeout: float) -> dict:
    results: queue.Queue = queue.Queue(maxsize=1)

    def invoke():
        try:
            result = callback(server=server, tool_name=tool_name, args=args, **call_ids)
            results.put_nowait((True, result))
        except Exception as exc:
            logger.exception("Required MCP dispatch callback failed")
            results.put_nowait((False, exc))

    from agent.memory_provider import spawn_context_thread
    worker = spawn_context_thread(invoke, name="required-mcp-dispatch-guard")
    worker.start()
    try:
        ok, result = results.get(timeout=timeout)
    except queue.Empty as exc:
        raise PermissionError("required MCP dispatch guard timed out") from exc
    if not ok:
        raise PermissionError("required MCP dispatch guard failed") from result
    return _json_object(result)


def admit_mcp_dispatch(server: str, tool_name: str, args: dict,
                       **call_kwargs: Any) -> tuple[dict | None, str | None]:
    """Return ``(canonical_args, None)`` or ``(None, safe_denial_message)``.

    A required policy is independent of plugin discovery. Missing/disabled
    plugins, missing origin, callback errors/timeouts and invalid results deny.
    Unconfigured servers retain their original dispatch behavior.
    """
    from hermes_cli.profiles import current_profile_name

    profile = current_profile_name("default") or "default"
    try:
        policy = _policy(server, profile)
    except Exception:
        logger.error("Required MCP dispatch policy could not be read for %s/%s", profile, server,
                     exc_info=True)
        return None, "Required MCP dispatch policy is invalid; no call was sent."
    if policy is None:
        # A registered *required* guard with a missing profile policy is
        # configuration drift, not an opt-out. This also protects a running
        # gateway if config.yaml is edited after plugin discovery.
        from hermes_cli.plugins import get_plugin_manager
        if server in get_plugin_manager()._required_mcp_dispatch_guards:
            return None, "Required MCP dispatch policy is missing; no call was sent."
        return args, None
    try:
        canonical = _json_object(args)
        from gateway.trusted_turn_origin import current_origin
        origin = current_origin()
        if origin is None and not _has_origin_fields(canonical) and _scheduler_allowed(policy, profile, tool_name):
            return canonical, None
        if (origin is None or origin.kind != "gateway" or origin.platform != policy["platform"]
                or origin.profile != profile or origin.transport_profile != profile
                or origin.internal or not origin.admitted):
            raise PermissionError("missing or mismatched current gateway origin")

        from hermes_cli.plugins import get_plugin_manager
        manager = get_plugin_manager()
        plugin_id = policy["plugin"]
        loaded = manager._plugins.get(plugin_id)
        entry = manager._required_mcp_dispatch_guards.get(server)
        if loaded is None or not loaded.enabled or entry is None or entry[0] != plugin_id:
            raise PermissionError("required plugin or callback unavailable")
        raw_timeout = policy.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
        if (isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float))
                or not 0 < raw_timeout <= 5):
            raise ValueError("invalid required MCP guard timeout")
        ids = {key: call_kwargs[key] for key in _CALL_IDS
               if isinstance(call_kwargs.get(key), str)}
        return _run_callback(entry[1], server=server, tool_name=tool_name, args=canonical,
                             call_ids=ids, timeout=float(raw_timeout)), None
    except Exception:
        logger.warning("Required MCP dispatch guard denied %s/%s on profile %s", server, tool_name, profile,
                       exc_info=True)
        return None, "Required MCP dispatch guard denied this call; no call was sent."
