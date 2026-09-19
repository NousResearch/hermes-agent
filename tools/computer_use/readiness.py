"""Bounded readiness checks via the cua-driver ``verify_state`` tool (RFC #112639).

A readiness check asks the driver whether a concrete postcondition holds
(window still exists / element matches a selector) with a hard deadline and
no screenshot — far cheaper than a capture + model round trip for "did the
last action land?".

Fails closed: a driver that does not advertise ``verify_state`` yields
``unknown`` and no driver call is made. Bounds are clamped to the driver's
contract; transport failures are reported, never retried.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Driver contract bounds for verify_state arguments.
_TIMEOUT_MS_MIN, _TIMEOUT_MS_MAX = 0, 10000
_STABLE_MIN, _STABLE_MAX = 1, 5
_EXPECT_MIN, _EXPECT_MAX = 1, 8

_DRIVER_STATUSES = frozenset({"satisfied", "unsatisfied", "unknown"})


@dataclass
class ReadinessResult:
    """Outcome of one bounded readiness check."""

    status: str  # satisfied | unsatisfied | unknown | error
    detail: str = ""
    duration_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def verify_readiness(
    backend: Any,
    *,
    pid: int,
    window_id: int,
    expect: List[Dict[str, Any]],
    timeout_ms: int = 2000,
    stable_samples: int = 1,
    include_screenshot: bool = False,
) -> ReadinessResult:
    """Ask the driver whether *expect* holds for the window, bounded.

    Routes through ``backend.call_tool`` (the Hermes-authorized path; the
    session is injected there), never the raw session/daemon. ``timeout_ms``
    is a hard driver-side deadline; ``stable_samples`` consecutive satisfied
    samples are required before the driver reports success.
    """
    session = getattr(backend, "_session", None)
    has_tool = getattr(session, "_has_tool", None)
    if not callable(has_tool) or not has_tool("verify_state"):
        return ReadinessResult(status="unknown", detail="verify_state not advertised by driver")
    if (
        not isinstance(expect, list)
        or not _EXPECT_MIN <= len(expect) <= _EXPECT_MAX
        or any(not isinstance(p, dict) for p in expect)
    ):
        raise ValueError(f"expect must be a list of {_EXPECT_MIN}-{_EXPECT_MAX} predicate dicts")
    timeout_ms = _clamp(timeout_ms, _TIMEOUT_MS_MIN, _TIMEOUT_MS_MAX)
    stable_samples = _clamp(stable_samples, _STABLE_MIN, _STABLE_MAX)
    payload = {
        "pid": int(pid),
        "window_id": int(window_id),
        "expect": expect,
        "timeout_ms": timeout_ms,
        "stable_samples": stable_samples,
        "include_screenshot": bool(include_screenshot),
    }
    t0 = time.perf_counter()
    try:
        # Transport timeout covers the driver deadline plus bridge overhead.
        out = backend.call_tool("verify_state", payload, timeout=timeout_ms / 1000.0 + 10.0)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        logger.warning("verify_state call failed: %s", e)
        return ReadinessResult(status="error", detail=f"verify_state call failed: {e}", duration_ms=ms)
    duration_ms = (time.perf_counter() - t0) * 1000
    structured = (out.get("structuredContent") if isinstance(out, dict) else None) or {}
    status = structured.get("status") if isinstance(structured, dict) else None
    if status not in _DRIVER_STATUSES:
        return ReadinessResult(
            status="unknown",
            detail="driver returned no usable verdict",
            duration_ms=duration_ms,
            raw=structured if isinstance(structured, dict) else {},
        )
    preds = structured.get("predicates") or []
    summary = ", ".join(f"#{p.get('index')}:{p.get('status')}" for p in preds if isinstance(p, dict))
    detail = f"driver status: {status}" + (f" ({summary})" if summary else "")
    return ReadinessResult(
        status=status, detail=detail, duration_ms=duration_ms, raw=structured
    )
