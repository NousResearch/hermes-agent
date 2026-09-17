"""Identity-retention trend tracking across guarded runs (#112734).

Answers "does element identity survive reconciliation across steps": every
reconciled capture records one retention sample per session; the per-run trend
summarizes the samples, and ``aggregate_identity_trends`` rolls the runs up into
an across-runs report. All values are scalar statistics — no element text,
geometry, or tokens leave a step, so trends stay content-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# Bounded like the shadow state stores: a long GUI session cannot grow these without
# limit, and the oldest session evicts first when the cap is hit.
_MAX_SESSIONS = 64
_MAX_STEPS = 1024
# Retention shift between first and second half that counts as a real direction.
_TREND_THRESHOLD = 0.05


@dataclass(frozen=True)
class RetentionSample:
    """One reconciled step: identity_retention plus the counts that produced it."""

    revision: int
    retention: float
    matched: int
    removed: int
    ambiguous: int
    mean_confidence: float


class IdentityTrend:
    """Ordered retention samples for one session (one run's guarded captures)."""

    def __init__(self) -> None:
        self._samples: List[RetentionSample] = []

    def record(self, sample: RetentionSample) -> None:
        self._samples.append(sample)
        if len(self._samples) > _MAX_STEPS:  # drop the oldest; the trend reads forward
            del self._samples[: len(self._samples) - _MAX_STEPS]

    @property
    def steps(self) -> int:
        return len(self._samples)

    def mean_retention(self) -> float:
        return sum(s.retention for s in self._samples) / len(self._samples) if self._samples else 1.0

    def min_retention(self) -> float:
        return min((s.retention for s in self._samples), default=1.0)

    def trend(self) -> float:
        # Second-half mean minus first-half mean: positive = identity getting stickier.
        n = len(self._samples)
        if n < 2:
            return 0.0
        half = n // 2
        first = sum(s.retention for s in self._samples[:half]) / half
        second = sum(s.retention for s in self._samples[half:]) / (n - half)
        return second - first

    @staticmethod
    def direction(trend: float) -> str:
        if trend > _TREND_THRESHOLD:
            return "improving"
        if trend < -_TREND_THRESHOLD:
            return "degrading"
        return "stable"

    def summary(self) -> Dict[str, Any]:
        """Per-run aggregate: the report half of the trend tracking."""
        t = self.trend()
        return {
            "steps": self.steps,
            "mean_retention": self.mean_retention(),
            "min_retention": self.min_retention(),
            "trend": t,
            "trend_direction": self.direction(t),
            "total_matched": sum(s.matched for s in self._samples),
            "total_removed": sum(s.removed for s in self._samples),
            "total_ambiguous": sum(s.ambiguous for s in self._samples),
        }


_trends: Dict[str, IdentityTrend] = {}


def record_identity_step(session_id: str, *, retention: float, matched: int, removed: int,
                         ambiguous: int, mean_confidence: float, revision: int) -> None:
    """Append one reconciled step to the session's trend (no-op-safe on odd input)."""
    if session_id not in _trends and len(_trends) >= _MAX_SESSIONS:
        _trends.pop(next(iter(_trends)))
    trend = _trends.setdefault(session_id, IdentityTrend())
    trend.record(RetentionSample(revision=revision, retention=max(0.0, min(1.0, retention)),
                                matched=max(0, matched), removed=max(0, removed),
                                ambiguous=max(0, ambiguous), mean_confidence=mean_confidence))


def get_identity_trend(session_id: str) -> Dict[str, Any]:
    """Per-run summary for a session (empty before any reconciled capture)."""
    trend = _trends.get(session_id)
    return trend.summary() if trend else {}


def aggregate_identity_trends() -> Dict[str, Any]:
    """Across-runs aggregate: one report over every tracked session."""
    summaries = {sid: trend.summary() for sid, trend in _trends.items()}
    means = [s["mean_retention"] for s in summaries.values()]
    directions: Dict[str, int] = {"improving": 0, "stable": 0, "degrading": 0}
    for s in summaries.values():
        directions[s["trend_direction"]] += 1
    worst = min(summaries.items(), key=lambda kv: kv[1]["mean_retention"], default=(None, None))
    return {
        "sessions": len(summaries),
        "total_steps": sum(s["steps"] for s in summaries.values()),
        "mean_retention": (sum(means) / len(means)) if means else 1.0,
        "min_retention": min(means) if means else 1.0,
        "trend_directions": directions,
        "worst_session": worst[0],
        "per_session": summaries,
    }


def reset_identity_trends_for_tests() -> None:  # pragma: no cover — test seam
    _trends.clear()
