"""Explicit advisory side channel for one programmatic tool call.

Only typed advisories emitted by a plugin are collected; result transforms are
never inspected. RPC results themselves retain their original JSON shape.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from threading import Lock


MAX_ADVISORIES = 32
MAX_MESSAGE_CHARS = 2048


@dataclass(frozen=True)
class ToolAdvisory:
    source: str
    code: str
    message: str


class AdvisorySink:
    def __init__(self, limit: int):
        self.items: list[ToolAdvisory] = []
        self.limit = limit
        self.closed = False
        self.lock = Lock()

    def emit(self, advisory: ToolAdvisory) -> None:
        if not isinstance(advisory, ToolAdvisory):
            raise TypeError('Expected ToolAdvisory')
        bounded = ToolAdvisory(advisory.source[:80], advisory.code[:80], advisory.message[:MAX_MESSAGE_CHARS])
        with self.lock:
            if not self.closed and len(self.items) < self.limit and bounded not in self.items:
                self.items.append(bounded)

    def close(self) -> None:
        with self.lock:
            self.closed = True

    def count(self) -> int:
        with self.lock:
            return len(self.items)

    def finish(self) -> list[dict]:
        with self.lock:
            self.closed = True
            items, self.items = self.items, []
        return [asdict(item) for item in items]


_current_sink: ContextVar[AdvisorySink | None] = ContextVar('tool_advisory_sink', default=None)


def current_advisory_sink() -> AdvisorySink | None:
    """Capture the call's sink when rebinding RPC execution to a cell's authority."""
    return _current_sink.get()


@contextmanager
def bind_advisory_sink(sink: AdvisorySink | None):
    token = _current_sink.set(sink)
    try:
        yield
    finally:
        _current_sink.reset(token)


def emit_tool_advisory(advisory: ToolAdvisory) -> None:
    """Emit to the active RPC call only; outside programmatic dispatch this is a no-op."""
    sink = _current_sink.get()
    if sink is not None:
        sink.emit(advisory)


def take_tool_advisories(tool_call_log: list, start: int = 0) -> list[dict]:
    """Drain this execution's metadata, so persistent kernels retain no advisories."""
    items = []
    for index in range(start, len(tool_call_log)):
        entry = tool_call_log[index]
        sink = entry.pop('_advisory_sink', None)
        for advisory in sink.finish() if sink is not None else []:
            if len(items) < MAX_ADVISORIES:
                items.append({'tool': entry['tool'], 'call_index': index - start + 1, **advisory})
    return items
