"""Opt-in, turn-local provider usage observations for a managed TUI turn.

These events are volatile observations, not a durable final-usage or worker-stop receipt.
Only usage from the accounting path that invokes ``record`` is represented.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, cast

_MAX_SAFE_INTEGER = 2**53 - 1
_BUCKETS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "total_tokens")


class ManagedTurnUsageObserver:
    def __init__(self, key: str, user_row_id: int, session_id: str, emit: Callable[[str, dict], None]):
        self.key = key
        self.user_row_id = user_row_id
        self.session_id = session_id
        self.emit = emit
        self.model: str | None = None
        self.observed_calls = 0
        self.usage_complete = True
        self.closed = False
        self.known = {name: 0 for name in _BUCKETS}
        self.known_calls = 0

    def record(self, usage, *, model: str, served_model: str | None, raw_usage_complete: bool) -> None:
        """Observe one accounted response; missing or mismatched metadata poisons completeness."""
        if self.closed:
            return
        self.observed_calls += 1
        if self.model is None:
            self.model = model
        raw = {name: getattr(usage, name, None) for name in _BUCKETS}
        values: dict[str, int] | None = (
            {name: cast(int, raw[name]) for name in _BUCKETS}
            if all(type(value) is int and 0 <= value <= _MAX_SAFE_INTEGER for value in raw.values()) else None)
        valid = (isinstance(model, str) and bool(model) and model == self.model
                 and served_model == model and raw_usage_complete is True and usage is not None
                 and type(getattr(usage, "request_count", None)) is int and usage.request_count == 1
                 and values is not None and values["total_tokens"] > 0
                 and values["total_tokens"] >= (values["input_tokens"] + values["output_tokens"]
                                                 + values["cache_read_tokens"] + values["cache_write_tokens"])
                 and all(self.known[name] <= _MAX_SAFE_INTEGER - values[name] for name in _BUCKETS))
        if valid and values is not None:
            for name in _BUCKETS:
                self.known[name] += values[name]
            self.known_calls += 1
        else:
            self.usage_complete = False
        self.emit(self.session_id, {
            "managed_turn_key": self.key, "user_row_id": self.user_row_id,
            "model": self.model or "", "observed_calls": self.observed_calls,
            "observed_usage_complete": self.usage_complete,
            "coverage": "accounted_responses_only",
            "usage": {"calls": self.known_calls, "input": self.known["input_tokens"],
                      "output": self.known["output_tokens"], "cache_read": self.known["cache_read_tokens"],
                      "cache_write": self.known["cache_write_tokens"],
                      "reasoning": self.known["reasoning_tokens"], "total": self.known["total_tokens"]},
        })

    def close(self) -> None:
        self.closed = True


@contextmanager
def managed_turn_usage_scope(key, row_id, sid, agent, emit):
    """Bind only this admitted turn's usage callback, then retire it before completion."""
    if key is None:
        yield
        return
    if type(row_id) is not int or row_id <= 0:
        raise RuntimeError("managed turn is missing its committed user row")
    if getattr(agent, "api_mode", None) == "codex_app_server":
        raise RuntimeError("managed per-call usage is unavailable for Codex app-server")
    if (getattr(agent, "provider", None) == "moa"
            or callable(getattr(getattr(agent, "client", None), "consume_reference_usage", None))):
        raise RuntimeError("managed per-call usage is unavailable for MoA multi-model routing")
    if getattr(agent, "_managed_turn_usage_callback", None) is not None:
        raise RuntimeError("a managed usage observer is already installed")
    observer = ManagedTurnUsageObserver(key, row_id, sid, emit)
    agent._managed_turn_usage_callback = observer.record
    try:
        yield
    finally:
        observer.close()
        agent._managed_turn_usage_callback = None
