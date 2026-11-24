"""Pre-storage output protection shared by process buffers and live sinks."""
from __future__ import annotations

from typing import Any

from agent.secure_input_output import SecureOutputFilter


class SecureProcessOutputMixin:
    _lock: Any
    _output_filter: SecureOutputFilter | None
    exited: bool
    output_buffer: str
    total_output_chars: int
    max_output_chars: int

    def _capture_output(self, text: str, *, unless_exited: bool = False) -> str | None:
        with self._lock:
            if unless_exited and self.exited:
                return None
            safe = self._output_filter.feed(text) if self._output_filter else text
            self._append_locked(safe)
            return safe

    def append_output(self, text: str) -> None:
        self._capture_output(text)

    def append_output_if_running(self, text: str) -> bool:
        return self._capture_output(text, unless_exited=True) is not None

    def _append_locked(self, text: str) -> None:
        self.output_buffer += text
        self.total_output_chars += len(text)
        if len(self.output_buffer) > self.max_output_chars:
            self.output_buffer = self.output_buffer[-self.max_output_chars:]
