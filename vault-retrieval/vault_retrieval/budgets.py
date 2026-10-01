"""Per-turn budget accounting and large-file refusal.

The runtime forwards the actual Hermes ``turn_id`` to the tool handler as
a dispatch kwarg (alongside ``task_id``/``session_id``/``user_task``),
and the handler keys the per-turn counter on it. Multiple calls in one
turn share one 24,000-character ceiling via the per-turn registry.

Counters are reset/expired at:

  - explicit reset_turn(turn_id) — the plugin hooks on_session_end /
    on_session_reset to call this for the matching session, and
    on_session_finalize to drop anything left over;
  - bounded TTL — ``expire_older_than(seconds)`` is called by
    process-wide lazy cleanup paths (start of each handle) so a stale
    counter from a crashed/abandoned session cannot leak past the
    bounded TTL even if the lifecycle hooks never fired.

Acceptance contract:
  - Default request ≤ 12,000 evidence chars; exactly 12,000 accepted.
  - 12,001 without expansion → ``data_hold``.
  - Approved expansion → 24,000 accepted; 24,001 rejected before any read.
  - Two calls sharing one ``turn_id`` share the cumulative ceiling.
  - Files >20,000 chars need an explicit selector; without one →
    ``large_file_selector_required``.
  - Large-file extraction: ≤ 2 ranges, ≤ 120 lines per range, ≤ 8,000 chars per range.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_BUDGET = 12_000
DEFAULT_HARD_CEILING = 24_000
DEFAULT_LARGE_FILE_CHARS = 20_000
LARGE_FILE_RANGES_MAX = 2
LARGE_FILE_LINES_MAX = 120
LARGE_FILE_CHARS_MAX = 8_000

# Bounded TTL for turn counters in seconds. A real Hermes session turn
# finishes in seconds; 1 hour is a safety net so a crashed/abandoned
# session cannot leak process-global state indefinitely. Lower bound:
# roughly an order of magnitude above any plausible live turn duration.
DEFAULT_TURN_TTL_SECONDS = 3600


@dataclass
class BudgetConfig:
    default_budget_chars: int = DEFAULT_BUDGET
    hard_ceiling_chars: int = DEFAULT_HARD_CEILING
    large_file_chars: int = DEFAULT_LARGE_FILE_CHARS
    max_primary_extracts: int = 3
    max_expansion_extracts: int = 2
    max_range_lines: int = LARGE_FILE_LINES_MAX
    max_range_chars: int = LARGE_FILE_CHARS_MAX


@dataclass(frozen=True)
class BudgetDecision:
    status: str  # "ok" | "data_hold"
    reason_code: Optional[str] = None
    detail: Optional[str] = None
    remaining_chars: int = 0
    used_chars: int = 0
    expanded: bool = False


@dataclass
class TurnBudgets:
    """Per-turn counter state. Owned by the tool handler."""

    turn_id: str
    cfg: BudgetConfig
    used_chars: int = 0
    expansion_used: bool = False
    last_used_at: float = field(default_factory=time.monotonic)

    @property
    def remaining_chars(self) -> int:
        return max(0, self.cfg.hard_ceiling_chars - self.used_chars)


# Process-wide turn registry so two calls in one turn share a counter.
# Cleaned up by:
#   - explicit reset_turn(turn_id) (called from session-end hooks)
#   - bounded TTL via expire_older_than(DEFAULT_TURN_TTL_SECONDS)
# Tests clear this via ``_turn_registry.clear()`` between cases.
_turn_registry: Dict[str, TurnBudgets] = {}
_turn_lock = threading.Lock()


def get_or_create_turn(turn_id: str, cfg: BudgetConfig) -> TurnBudgets:
    with _turn_lock:
        tb = _turn_registry.get(turn_id)
        if tb is None:
            tb = TurnBudgets(turn_id=turn_id, cfg=cfg)
            _turn_registry[turn_id] = tb
        else:
            # Touch the timestamp so an active turn never expires under
            # us, even if the TTL is short.
            tb.last_used_at = time.monotonic()
        return tb


def reset_turn(turn_id: str) -> None:
    """Called at turn completion / session end."""
    with _turn_lock:
        _turn_registry.pop(turn_id, None)


def reset_session_turns(prefix: str) -> int:
    """Drop every counter whose key starts with ``prefix``.

    Used by ``on_session_end`` to evict counters keyed on ``runtime:<task>:<session>``
    when a session ends; also by tests to assert cross-session isolation.
    Returns the count of dropped counters.
    """
    if not prefix:
        return 0
    dropped = 0
    with _turn_lock:
        keys = [k for k in list(_turn_registry) if k.startswith(prefix)]
        for k in keys:
            _turn_registry.pop(k, None)
            dropped += 1
    return dropped


def expire_older_than(seconds: float) -> int:
    """Drop every counter last touched more than ``seconds`` ago.

    Called by the lazy cleanup path at the start of each handle and by
    test fixtures; bounded TTL guarantees a stale counter from a crashed
    session cannot leak forever even if the lifecycle hooks never fire.
    """
    if seconds <= 0:
        return 0
    now = time.monotonic()
    cutoff = now - seconds
    dropped = 0
    with _turn_lock:
        keys = [k for k, tb in list(_turn_registry.items())
                if tb.last_used_at < cutoff]
        for k in keys:
            _turn_registry.pop(k, None)
            dropped += 1
    return dropped


def _registry_snapshot() -> Dict[str, TurnBudgets]:
    """Snapshot of the current registry. Test-only helper."""
    with _turn_lock:
        return dict(_turn_registry)


def _registry_keys_for_test() -> Tuple[str, ...]:
    """Snapshot the registry keys for assertions. Test-only helper."""
    with _turn_lock:
        return tuple(_turn_registry.keys())


def decide_budget(
    tb: TurnBudgets,
    *,
    request_chars: int,
    allow_expansion: bool,
    expansion_reason: str = "",
) -> BudgetDecision:
    """Decide whether a single request fits the per-turn budget.

    Caller passes the per-turn counter (``TurnBudgets``) and the request's
    planned character count. The decision is deterministic given the same
    inputs — no time, no randomness.
    """
    if request_chars < 0:
        return BudgetDecision(
            status="data_hold",
            reason_code="negative_request",
            detail="request_chars must be >= 0",
        )

    hard_ceiling = tb.cfg.hard_ceiling_chars
    default_budget = tb.cfg.default_budget_chars

    # Approved expansion path
    if allow_expansion:
        if not expansion_reason:
            return BudgetDecision(
                status="data_hold",
                reason_code="expansion_reason_required",
                detail="allow_bounded_expansion=true requires a non-empty expansion_reason",
            )
        if request_chars > hard_ceiling:
            return BudgetDecision(
                status="data_hold",
                reason_code="hard_ceiling_exceeded",
                detail=f"request_chars {request_chars} exceeds hard ceiling {hard_ceiling}",
            )
        if tb.used_chars + request_chars > hard_ceiling:
            return BudgetDecision(
                status="data_hold",
                reason_code="hard_ceiling_exceeded",
                detail=(
                    f"turn already used {tb.used_chars}; "
                    f"additional {request_chars} would exceed hard ceiling {hard_ceiling}"
                ),
            )
        tb.used_chars += request_chars
        tb.expansion_used = True
        return BudgetDecision(
            status="ok",
            remaining_chars=tb.cfg.hard_ceiling_chars - tb.used_chars,
            used_chars=tb.used_chars,
            expanded=True,
        )

    # Default (non-expansion) path
    if request_chars > default_budget:
        return BudgetDecision(
            status="data_hold",
            reason_code="budget_exceeded",
            detail=(
                f"request_chars {request_chars} exceeds default {default_budget}; "
                "set allow_bounded_expansion=true with a non-empty expansion_reason"
            ),
        )
    if tb.used_chars + request_chars > default_budget:
        return BudgetDecision(
            status="data_hold",
            reason_code="budget_exceeded",
            detail=(
                f"turn already used {tb.used_chars}; "
                f"additional {request_chars} would exceed default {default_budget}"
            ),
        )
    tb.used_chars += request_chars
    return BudgetDecision(
        status="ok",
        remaining_chars=default_budget - tb.used_chars,
        used_chars=tb.used_chars,
        expanded=False,
    )


# ---------------------------------------------------------------------------
# Large file handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LargeFileDecision:
    is_large: bool
    decision: str  # "ok" | "large_file_selector_required" | "too_many_ranges" | "range_too_long" | "range_too_long_chars"
    detail: Optional[str] = None


def check_large_file(
    *, cfg: BudgetConfig, file_chars: int, selectors: List[Dict[str, Any]]
) -> LargeFileDecision:
    """Decide whether a large-file request is acceptable.

    If file_chars <= large_file_chars: not a large file, no selector needed
    (decision == 'ok' for non-large files for uniformity; is_large=False).

    If file_chars > large_file_chars:
      - 0 selectors -> 'large_file_selector_required'
      - > 2 selectors -> 'too_many_ranges'
      - any selector with > max_range_lines -> 'range_too_long'
      - any selector with > max_range_chars estimated_chars -> 'range_too_long_chars'
      - else 'ok'
    """
    threshold = cfg.large_file_chars
    if file_chars <= threshold:
        return LargeFileDecision(is_large=False, decision="ok")
    if not selectors:
        return LargeFileDecision(
            is_large=True,
            decision="large_file_selector_required",
            detail=(
                f"file_chars {file_chars} exceeds large_file threshold {threshold}; "
                "a heading, stable ID, date, owner/status key, or explicit line selector is required"
            ),
        )
    if len(selectors) > LARGE_FILE_RANGES_MAX:
        return LargeFileDecision(
            is_large=True,
            decision="too_many_ranges",
            detail=f"{len(selectors)} selectors exceeds LARGE_FILE_RANGES_MAX {LARGE_FILE_RANGES_MAX}",
        )
    for sel in selectors:
        ls = sel.get("line_start")
        le = sel.get("line_end")
        if isinstance(ls, int) and isinstance(le, int):
            span = le - ls + 1
            if span > cfg.max_range_lines:
                return LargeFileDecision(
                    is_large=True,
                    decision="range_too_long",
                    detail=f"range {ls}-{le} has {span} lines; max {cfg.max_range_lines}",
                )
        ec = sel.get("estimated_chars")
        if isinstance(ec, int) and ec > cfg.max_range_chars:
            return LargeFileDecision(
                is_large=True,
                decision="range_too_long_chars",
                detail=f"range estimated_chars {ec} exceeds max {cfg.max_range_chars}",
            )
    return LargeFileDecision(is_large=True, decision="ok")
