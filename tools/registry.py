"""Central registry for all hermes-agent tools.

Each tool file calls ``registry.register()`` at module level to declare its
schema, handler, toolset membership, and availability check.  ``model_tools.py``
queries the registry instead of maintaining its own parallel data structures.

Import chain (circular-import safe):
    tools/registry.py  (no imports from model_tools or tool files)
           ^
    tools/*.py  (import from tools.registry at module level)
           ^
    model_tools.py  (imports tools.registry + all tool modules)
           ^
    run_agent.py, cli.py, batch_runner.py, etc.
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_TOOL_RESULT_KEYS = ("success", "error", "error_type", "retryable", "data", "metrics")
_TOOL_RESULT_KEY_SET = set(_TOOL_RESULT_KEYS)
_TOOL_RESULT_SIGNAL_KEYS = {"success", "error", "error_type", "retryable"}


def make_tool_result(
    *,
    success: bool,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    retryable: bool = False,
    data: Any = None,
    metrics: Optional[dict] = None,
) -> str:
    """Return the canonical JSON tool result envelope."""
    envelope = {
        "success": bool(success),
        "error": error if error is None else str(error),
        "error_type": error_type if error_type is None else str(error_type),
        "retryable": bool(retryable),
        "data": data,
        "metrics": metrics if isinstance(metrics, dict) else {},
    }
    return json.dumps(envelope, ensure_ascii=False, default=str)


class ToolEntry:
    """Metadata for a single registered tool."""

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        self._toolset_checks: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable = None,
        requires_env: list = None,
        is_async: bool = False,
        description: str = "",
    ):
        """Register a tool.  Called at module-import time by each tool file."""
        self._tools[name] = ToolEntry(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            requires_env=requires_env or [],
            is_async=is_async,
            description=description or schema.get("description", ""),
        )
        if check_fn and toolset not in self._toolset_checks:
            self._toolset_checks[toolset] = check_fn

    # ------------------------------------------------------------------
    # Schema retrieval
    # ------------------------------------------------------------------

    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """Return OpenAI-format tool schemas for the requested tool names.

        Only tools whose ``check_fn()`` returns True (or have no check_fn)
        are included.
        """
        result = []
        for name in sorted(tool_names):
            entry = self._tools.get(name)
            if not entry:
                continue
            if entry.check_fn:
                try:
                    if not entry.check_fn():
                        if not quiet:
                            logger.debug("Tool %s unavailable (check failed)", name)
                        continue
                except Exception:
                    if not quiet:
                        logger.debug("Tool %s check raised; skipping", name)
                    continue
            result.append({"type": "function", "function": entry.schema})
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_parse_json(value: Any) -> Any:
        """Best-effort JSON decode for handler outputs."""
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return value
        return value

    @staticmethod
    def _stringify_field(value: Any) -> Optional[str]:
        """Normalize error-like fields into strings or None."""
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return str(value)

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        """Normalize bool-like values without relying on raw truthiness alone."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", ""}:
                return False
            return default
        return bool(value)

    @staticmethod
    def _sanitize_metrics(metrics: Any, tool_name: str, duration_ms: int) -> dict:
        """Force metrics into a dict and stamp canonical dispatch metadata."""
        normalized = {}
        if isinstance(metrics, dict):
            normalized = {str(key): value for key, value in metrics.items()}
        normalized["tool_name"] = tool_name
        normalized["duration_ms"] = max(0, int(duration_ms))
        return normalized

    @staticmethod
    def _looks_like_envelope(payload: Any) -> bool:
        """Detect existing success/error-style tool payloads."""
        if not isinstance(payload, dict):
            return False
        signal_count = len(_TOOL_RESULT_SIGNAL_KEYS.intersection(payload))
        return "success" in payload or signal_count >= 2

    @staticmethod
    def _merge_data_with_extras(data: Any, extras: dict) -> Any:
        """Preserve extra payload keys alongside canonical ``data``."""
        if not extras:
            return data
        if isinstance(data, dict):
            merged = dict(data)
            for key, value in extras.items():
                merged.setdefault(key, value)
            return merged

        wrapped = {"value": data}
        for key, value in extras.items():
            if key not in wrapped:
                wrapped[key] = value
                continue
            suffix = 2
            conflict_key = f"extra_{key}"
            while conflict_key in wrapped:
                conflict_key = f"extra_{key}_{suffix}"
                suffix += 1
            wrapped[conflict_key] = value
        return wrapped

    def validate_tool_result_envelope(self, payload: Any, *, tool_name: str, duration_ms: int) -> dict:
        """Sanitize arbitrary handler output into the canonical tool envelope."""
        parsed = self._maybe_parse_json(payload)
        metrics = self._sanitize_metrics(None, tool_name, duration_ms)

        if not self._looks_like_envelope(parsed):
            return {
                "success": True,
                "error": None,
                "error_type": None,
                "retryable": False,
                "data": parsed,
                "metrics": metrics,
            }

        extras = {key: value for key, value in parsed.items() if key not in _TOOL_RESULT_KEY_SET}
        error = self._stringify_field(parsed.get("error"))
        error_type = self._stringify_field(parsed.get("error_type"))

        raw_success = parsed.get("success")
        success_default = error is None
        if raw_success is None:
            success = success_default
        else:
            success = self._coerce_bool(raw_success, default=success_default)

        data = parsed.get("data")
        if "data" not in parsed and extras:
            data = extras
        elif "data" in parsed and extras:
            data = self._merge_data_with_extras(data, extras)

        retryable = self._coerce_bool(parsed.get("retryable"), default=False)
        metrics = self._sanitize_metrics(parsed.get("metrics"), tool_name, duration_ms)

        if success:
            error = None
            error_type = None
            retryable = False
        else:
            error = error or "Tool reported failure."

        return {
            "success": success,
            "error": error,
            "error_type": error_type,
            "retryable": retryable,
            "data": data,
            "metrics": metrics,
        }

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """Execute a tool handler by name.

        * Async handlers are bridged automatically via ``_run_async()``.
        * All outputs are normalized into the canonical tool result envelope.
        * All exceptions are caught and returned as a JSON string for
          backward-compatible callers.
        """
        entry = self._tools.get(name)
        if not entry:
            return make_tool_result(
                success=False,
                error=f"Unknown tool: {name}",
                error_type="UnknownToolError",
                retryable=False,
                data=None,
                metrics=self._sanitize_metrics(None, name, 0),
            )

        started_at = time.perf_counter()
        try:
            if entry.is_async:
                from model_tools import _run_async

                raw_result = _run_async(entry.handler(args, **kwargs))
            else:
                raw_result = entry.handler(args, **kwargs)

            duration_ms = round((time.perf_counter() - started_at) * 1000)
            normalized = self.validate_tool_result_envelope(
                raw_result,
                tool_name=name,
                duration_ms=duration_ms,
            )
            return make_tool_result(**normalized)
        except Exception as e:
            logger.exception("Tool %s dispatch error: %s", name, e)
            duration_ms = round((time.perf_counter() - started_at) * 1000)
            return make_tool_result(
                success=False,
                error=f"Tool execution failed: {type(e).__name__}: {e}",
                error_type=type(e).__name__,
                retryable=False,
                data=None,
                metrics=self._sanitize_metrics(None, name, duration_ms),
            )

    # ------------------------------------------------------------------
    # Query helpers  (replace redundant dicts in model_tools.py)
    # ------------------------------------------------------------------

    def get_all_tool_names(self) -> List[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def get_toolset_for_tool(self, name: str) -> Optional[str]:
        """Return the toolset a tool belongs to, or None."""
        entry = self._tools.get(name)
        return entry.toolset if entry else None

    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        """Return ``{tool_name: toolset_name}`` for every registered tool."""
        return {name: e.toolset for name, e in self._tools.items()}

    def is_toolset_available(self, toolset: str) -> bool:
        """Check if a toolset's requirements are met.

        Returns False (rather than crashing) when the check function raises
        an unexpected exception (e.g. network error, missing import, bad config).
        """
        check = self._toolset_checks.get(toolset)
        if not check:
            return True
        try:
            return bool(check())
        except Exception:
            logger.debug("Toolset %s check raised; marking unavailable", toolset)
            return False

    def check_toolset_requirements(self) -> Dict[str, bool]:
        """Return ``{toolset: available_bool}`` for every toolset."""
        toolsets = set(e.toolset for e in self._tools.values())
        return {ts: self.is_toolset_available(ts) for ts in sorted(toolsets)}

    def get_available_toolsets(self) -> Dict[str, dict]:
        """Return toolset metadata for UI display."""
        toolsets: Dict[str, dict] = {}
        for entry in self._tools.values():
            ts = entry.toolset
            if ts not in toolsets:
                toolsets[ts] = {
                    "available": self.is_toolset_available(ts),
                    "tools": [],
                    "description": "",
                    "requirements": [],
                }
            toolsets[ts]["tools"].append(entry.name)
            if entry.requires_env:
                for env in entry.requires_env:
                    if env not in toolsets[ts]["requirements"]:
                        toolsets[ts]["requirements"].append(env)
        return toolsets

    def get_toolset_requirements(self) -> Dict[str, dict]:
        """Build a TOOLSET_REQUIREMENTS-compatible dict for backward compat."""
        result: Dict[str, dict] = {}
        for entry in self._tools.values():
            ts = entry.toolset
            if ts not in result:
                result[ts] = {
                    "name": ts,
                    "env_vars": [],
                    "check_fn": self._toolset_checks.get(ts),
                    "setup_url": None,
                    "tools": [],
                }
            if entry.name not in result[ts]["tools"]:
                result[ts]["tools"].append(entry.name)
            for env in entry.requires_env:
                if env not in result[ts]["env_vars"]:
                    result[ts]["env_vars"].append(env)
        return result

    def check_tool_availability(self, quiet: bool = False):
        """Return (available_toolsets, unavailable_info) like the old function."""
        available = []
        unavailable = []
        seen = set()
        for entry in self._tools.values():
            ts = entry.toolset
            if ts in seen:
                continue
            seen.add(ts)
            if self.is_toolset_available(ts):
                available.append(ts)
            else:
                unavailable.append({
                    "name": ts,
                    "env_vars": entry.requires_env,
                    "tools": [e.name for e in self._tools.values() if e.toolset == ts],
                })
        return available, unavailable


# Module-level singleton
registry = ToolRegistry()
