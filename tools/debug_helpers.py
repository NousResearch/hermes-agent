"""Shared debug session infrastructure for Hermes tools.

Replaces the identical DEBUG_MODE / _log_debug_call / _save_debug_log /
get_debug_session_info boilerplate previously duplicated across web_tools,
vision_tools, mixture_of_agents_tool, and image_generation_tool.

Usage in a tool module:

    from tools.debug_helpers import DebugSession

    _debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")

    # Log a call (no-op when debug mode is off)
    _debug.log_call("web_search", {"query": q, "results": len(r)})

    # Save the debug log (no-op when debug mode is off)
    _debug.save()

    # Expose debug info to external callers
    def get_debug_session_info():
        return _debug.get_session_info()
"""

import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DebugSession:
    """Per-tool debug session that records tool calls to a JSON log file.

    Activated by a tool-specific environment variable (e.g. WEB_TOOLS_DEBUG=true).
    When disabled, all methods are cheap no-ops.
    """

    def __init__(self, tool_name: str, *, env_var: str) -> None:
        self.tool_name = tool_name
        self.enabled = os.getenv(env_var, "false").lower() == "true"
        self.session_id = str(uuid.uuid4()) if self.enabled else ""
        self.log_dir = Path("./logs")
        self._calls: list[Dict[str, Any]] = []
        self._start_time = datetime.datetime.now().isoformat() if self.enabled else ""

        if self.enabled:
            self.log_dir.mkdir(exist_ok=True)
            logger.debug("%s debug mode enabled - Session ID: %s",
                         tool_name, self.session_id)

    @property
    def active(self) -> bool:
        return self.enabled

    def log_call(self, call_name: str, call_data: Dict[str, Any]) -> None:
        """Append a tool-call entry to the in-memory log."""
        if not self.enabled:
            return
        self._calls.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "tool_name": call_name,
            **call_data,
        })

    def save(self) -> None:
        """Flush the in-memory log to a JSON file in the logs directory."""
        if not self.enabled:
            return
        try:
            filename = f"{self.tool_name}_debug_{self.session_id}.json"
            filepath = self.log_dir / filename
            payload = {
                "session_id": self.session_id,
                "start_time": self._start_time,
                "end_time": datetime.datetime.now().isoformat(),
                "debug_enabled": True,
                "total_calls": len(self._calls),
                "tool_calls": self._calls,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.debug("%s debug log saved: %s", self.tool_name, filepath)
        except Exception as e:
            logger.error("Error saving %s debug log: %s", self.tool_name, e)

    def get_session_info(self) -> Dict[str, Any]:
        """Return a summary dict suitable for returning from get_debug_session_info()."""
        if not self.enabled:
            return {
                "enabled": False,
                "session_id": None,
                "log_path": None,
                "total_calls": 0,
            }
        return {
            "enabled": True,
            "session_id": self.session_id,
            "log_path": str(self.log_dir / f"{self.tool_name}_debug_{self.session_id}.json"),
            "total_calls": len(self._calls),
        }
