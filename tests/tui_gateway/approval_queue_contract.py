"""Deterministic client contract for Hermes gateway approval events.

This is a test/parity seam, not production gateway state.  It models the
minimum rules a client must enforce while consuming public ``approval.request``
events and sending ``approval.respond`` RPCs:

* the request event's unique, non-empty ``request_id`` is authoritative;
* responses are FIFO and must resolve exactly one queued request;
* an approval request may arrive before ``tool.start``;
* unresolved requests and every event after the turn terminal fail closed.

Only normalized lifecycle facts leave the adapter.  Request ids, session ids,
tool arguments, commands, results, and response text are never rendered.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


EXPECTED_CHOICES = ("once", "deny", "once")
EXPECTED_EXECUTIONS = (1, 0, 1)
TRACE_SCHEMA_VERSION = 1


class ApprovalProtocolViolation(RuntimeError):
    """Raised when a gateway approval exchange cannot be proved safe."""


@dataclass
class GatewayApprovalTrace:
    """Validate one deterministic approval sequence and render a safe trace."""

    session_id: str
    expected_choices: tuple[str, ...] = EXPECTED_CHOICES
    _pending: list[str] = field(default_factory=list, init=False, repr=False)
    _seen: set[str] = field(default_factory=set, init=False, repr=False)
    _responded: set[str] = field(default_factory=set, init=False, repr=False)
    _choices: list[str] = field(default_factory=list, init=False, repr=False)
    _resolved: list[int] = field(default_factory=list, init=False, repr=False)
    _event_order: list[str] = field(default_factory=list, init=False, repr=False)
    _terminal_outcomes: list[str] = field(default_factory=list, init=False, repr=False)

    def observe_event(self, message: Mapping[str, Any]) -> str | None:
        """Observe one public gateway event and return a new approval id."""

        if self._terminal_outcomes:
            raise ApprovalProtocolViolation("event received after turn terminal")
        if message.get("method") != "event":
            raise ApprovalProtocolViolation("expected a public gateway event")
        params = message.get("params")
        if not isinstance(params, Mapping):
            raise ApprovalProtocolViolation("event params must be a mapping")
        event_type = params.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise ApprovalProtocolViolation("event type is required")

        if event_type == "approval.request":
            payload = params.get("payload")
            request_id = payload.get("request_id") if isinstance(payload, Mapping) else None
            if not isinstance(request_id, str) or not request_id.strip():
                raise ApprovalProtocolViolation(
                    "approval.request requires a non-empty request_id"
                )
            if request_id in self._seen:
                raise ApprovalProtocolViolation("duplicate approval request_id")
            self._seen.add(request_id)
            self._pending.append(request_id)
            self._event_order.append("approval.request")
            return request_id

        if event_type == "tool.start":
            if self._pending:
                raise ApprovalProtocolViolation(
                    "tool.start observed before the pending approval resolved"
                )
            self._event_order.append("tool.start")
            return None

        if event_type == "tool.complete":
            if self._pending:
                raise ApprovalProtocolViolation(
                    "tool.complete observed before the pending approval resolved"
                )
            self._event_order.append("tool.complete")
            return None

        if event_type == "message.complete":
            if self._pending:
                raise ApprovalProtocolViolation(
                    "turn completed with an unresolved approval request"
                )
            if len(self._choices) != len(self.expected_choices):
                raise ApprovalProtocolViolation(
                    "turn completed before the approval sequence finished"
                )
            self._terminal_outcomes.append("completed")
            self._event_order.append("message.complete")
            return None

        # Other public lifecycle events do not change approval correlation.
        return None

    def respond(
        self,
        request_id: str,
        rpc: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> str:
        """Resolve the oldest pending approval through the public RPC."""

        if self._terminal_outcomes:
            raise ApprovalProtocolViolation("approval response attempted after terminal")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ApprovalProtocolViolation("approval response requires request_id")
        if request_id in self._responded:
            raise ApprovalProtocolViolation("duplicate approval response")
        if not self._pending:
            raise ApprovalProtocolViolation("unknown approval request_id")
        if self._pending[0] != request_id:
            raise ApprovalProtocolViolation("out-of-order approval request_id")
        if len(self._choices) >= len(self.expected_choices):
            raise ApprovalProtocolViolation("unexpected extra approval request")

        choice = self.expected_choices[len(self._choices)]
        response = rpc(
            {
                "jsonrpc": "2.0",
                "id": f"synthetic-approval-response-{len(self._choices) + 1}",
                "method": "approval.respond",
                "params": {
                    "session_id": self.session_id,
                    "request_id": request_id,
                    "choice": choice,
                },
            }
        )
        result = response.get("result") if isinstance(response, Mapping) else None
        resolved = result.get("resolved") if isinstance(result, Mapping) else None
        if resolved != 1:
            raise ApprovalProtocolViolation(
                "approval.respond did not resolve exactly one request"
            )

        self._pending.pop(0)
        self._responded.add(request_id)
        self._choices.append(choice)
        self._resolved.append(resolved)
        self._event_order.append(f"approval.respond:{choice}")
        return choice

    def normalized_trace(self, execution_counts: tuple[int, ...]) -> dict[str, Any]:
        """Return a parity-runner-safe trace after the complete sequence."""

        if self._pending:
            raise ApprovalProtocolViolation("cannot render with unresolved approvals")
        if tuple(self._choices) != self.expected_choices:
            raise ApprovalProtocolViolation("approval choices are incomplete")
        if tuple(self._resolved) != (1,) * len(self.expected_choices):
            raise ApprovalProtocolViolation("approval resolution cardinality is invalid")
        if execution_counts != EXPECTED_EXECUTIONS:
            raise ApprovalProtocolViolation("fixture execution cardinality is invalid")
        if self._terminal_outcomes != ["completed"]:
            raise ApprovalProtocolViolation("turn must have exactly one terminal outcome")

        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "approval_choices": list(self._choices),
            "approval_resolved": list(self._resolved),
            "event_order": list(self._event_order),
            "fixture_execution_counts": list(execution_counts),
            "approval_request_before_tool_start": (
                self._event_order.index("approval.request")
                < self._event_order.index("tool.start")
            ),
            "terminal_outcomes": list(self._terminal_outcomes),
            "provider_calls": 0,
        }


def render_normalized_trace(trace: Mapping[str, Any]) -> str:
    """Render stable JSON without any opaque request or session identity."""

    return json.dumps(trace, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ApprovalProtocolViolation",
    "EXPECTED_CHOICES",
    "EXPECTED_EXECUTIONS",
    "GatewayApprovalTrace",
    "TRACE_SCHEMA_VERSION",
    "render_normalized_trace",
]
