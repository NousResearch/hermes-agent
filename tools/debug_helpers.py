"""Shared per-tool debug session: records tool calls to a JSON log when a
tool-specific env var (e.g. WEB_TOOLS_DEBUG=true) is set; no-ops otherwise."""

import datetime
import json
import logging
import os
import uuid
from typing import Any, Dict

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.datetime.now().isoformat()


class DebugSession:
    """Per-tool debug session that records tool calls to a JSON log file."""

    def __init__(self, tool_name: str, *, env_var: str) -> None:
        self.tool_name = tool_name
        self.enabled = os.getenv(env_var, "false").lower() == "true"
        self.session_id = str(uuid.uuid4()) if self.enabled else ""
        self.log_dir = get_hermes_home() / "logs"
        self._calls: list[Dict[str, Any]] = []
        self._start_time = _now() if self.enabled else ""
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("%s debug mode enabled - Session ID: %s", tool_name, self.session_id)

    @property
    def active(self) -> bool:
        return self.enabled

    def log_call(self, call_name: str, call_data: Dict[str, Any]) -> None:
        """Append a tool-call entry to the in-memory log."""
        if self.enabled:
            self._calls.append({"timestamp": _now(), "tool_name": call_name, **call_data})

    def save(self) -> None:
        """Flush the in-memory log to a JSON file in the logs directory."""
        if not self.enabled:
            return
        try:
            filepath = self.log_dir / f"{self.tool_name}_debug_{self.session_id}.json"
            payload = {
                "session_id": self.session_id, "start_time": self._start_time, "end_time": _now(),
                "debug_enabled": True, "total_calls": len(self._calls), "tool_calls": self._calls}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.debug("%s debug log saved: %s", self.tool_name, filepath)
        except Exception as e:
            logger.error("Error saving %s debug log: %s", self.tool_name, e)
