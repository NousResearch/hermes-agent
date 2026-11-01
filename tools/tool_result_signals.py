"""Structured tool-result control signals.

Tool handlers can embed an explicit control signal in their JSON result so the
runtime can react to known terminal states without brittle substring matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from tools.registry import tool_error
from utils import safe_json_loads


_SIGNAL_KEY = "_hermes_tool_signal"
_SIGNAL_VERSION = 1
_HALT_KIND = "halt"
_BILLING_BLOCKER_CODE = "billing_blocker"


@dataclass(frozen=True)
class ToolResultSignal:
    """Typed runtime control signal emitted by a tool wrapper."""

    kind: str
    code: str
    message: str
    provider: str | None = None
    status_code: int | None = None
    retryable: bool = True


class ProviderBillingBlocker(RuntimeError):
    """An explicit provider HTTP 402 that cannot make progress without user action."""

    def __init__(self, *, provider: str, message: str, status_code: int = 402):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


def provider_billing_blocker_from_exception(
    exc: BaseException,
    *,
    provider: str,
) -> ProviderBillingBlocker | None:
    """Preserve typed blockers and classify only explicit HTTP status metadata."""

    if isinstance(exc, ProviderBillingBlocker):
        return exc
    response = getattr(exc, "response", None)
    for value in (getattr(exc, "status_code", None), getattr(response, "status_code", None)):
        if isinstance(value, int) and not isinstance(value, bool) and value == 402:
            return ProviderBillingBlocker(
                provider=provider,
                status_code=value,
                message=str(exc) or f"{provider} returned HTTP 402 Payment Required",
            )
    return None


def billing_blocker_tool_error(
    message: str,
    *,
    provider: str,
    status_code: int = 402,
) -> str:
    """Return a structured non-retryable billing blocker tool error."""

    return tool_error(
        message,
        success=False,
        retryable=False,
        provider=provider,
        status_code=status_code,
        **{
            _SIGNAL_KEY: {
                "version": _SIGNAL_VERSION,
                "kind": _HALT_KIND,
                "code": _BILLING_BLOCKER_CODE,
                "message": str(message),
                "provider": provider,
                "status_code": status_code,
                "retryable": False,
            }
        },
    )


def parse_tool_result_signal(result: str | Mapping[str, Any] | None) -> ToolResultSignal | None:
    """Parse a structured control signal from a tool result string or envelope."""

    payload = result if isinstance(result, Mapping) else safe_json_loads(result or "")
    if not isinstance(payload, dict):
        return None

    signal = payload.get(_SIGNAL_KEY)
    if not isinstance(signal, dict):
        return None
    if signal.get("version") != _SIGNAL_VERSION:
        return None

    kind = signal.get("kind")
    code = signal.get("code")
    message = signal.get("message")
    if not isinstance(kind, str) or not isinstance(code, str) or not isinstance(message, str):
        return None

    provider = signal.get("provider")
    if provider is not None and not isinstance(provider, str):
        provider = None

    status_code_raw = signal.get("status_code")
    status_code = status_code_raw if isinstance(status_code_raw, int) else None

    retryable = signal.get("retryable")
    if not isinstance(retryable, bool):
        retryable = True

    return ToolResultSignal(
        kind=kind,
        code=code,
        message=message,
        provider=provider,
        status_code=status_code,
        retryable=retryable,
    )


def is_billing_blocker_signal(signal: ToolResultSignal | None) -> bool:
    """Return True when the signal means the tool hit a terminal billing wall."""

    return bool(
        signal is not None
        and signal.kind == _HALT_KIND
        and signal.code == _BILLING_BLOCKER_CODE
        and signal.retryable is False
        and signal.status_code == 402
    )
