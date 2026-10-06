"""#6779 — the live delegation progress snapshot carries which model each running child is on
and how many tokens it has consumed, so `/tasks` (an alias of `/agents`) can answer "what is
running, on what model, at what cost" without waiting for a completion.

Drives the REAL progress-token sampler the registry samples for a live record: the tuple a real
dispatch builds from its children -> ``_children_activity_from_token`` -> the per-child dict
``/agents`` and ``/tasks`` render. The token is the stall monitor's liveness signal, so the
contract pinned here is the relationship between the sampler's tuple shape and what the UIs read:
model/tokens survive that round trip for every child, and a foreign/legacy 2-tuple still
degrades instead of raising.
"""

import pytest

from tools import async_delegation as ad
from tools.delegate_tool_dispatch import _batch_progress_token


class _Child:
    """Minimal stand-in for a live AIAgent child: the three fields the sampler reads."""

    def __init__(self, model="sonnet-4", total_tokens=12400, api_call_count=7, current_tool="web_search"):
        self.model = model
        self.session_total_tokens = total_tokens
        self._api_call_count = api_call_count
        self._current_tool = current_tool

    def get_activity_summary(self):
        return {"api_call_count": self._api_call_count, "current_tool": self._current_tool,
                "last_activity_ts": 1000.0}


def test_live_children_activity_reports_model_and_tokens_per_child():
    """Every live child of a running delegation reports the model it is on and the tokens it has
    burned, alongside the existing liveness fields — the three facts `/tasks` is asked to show."""
    children = [_Child("sonnet-4", 12400), _Child("haiku-3.5", 3100, api_call_count=2, current_tool=None)]
    token, in_tool = _batch_progress_token(children)

    activity = ad._children_activity_from_token(token, now=1000.0)

    assert activity is not None, "a real sampler tuple must project into per-child activity"
    assert [entry["model"] for entry in activity] == ["sonnet-4", "haiku-3.5"]
    # Tokens are monotonically accumulated per child, so the sum across children equals each
    # child's own running total (never the batch's, which would hide a heavy child behind a light one).
    assert [entry["tokens"] for entry in activity] == [12400, 3100]
    # The liveness fields the stall monitor and the existing /agents rows depend on are unchanged.
    assert [entry["api_calls"] for entry in activity] == [7, 2]
    assert [entry["current_tool"] for entry in activity] == ["web_search", None]
    assert in_tool is True


def test_progress_token_shape_change_does_not_break_a_legacy_two_tuple():
    """The sampler tuple widened, so a token from an older/foreign producer carrying only the
    original (api_calls, current_tool) pair must still project — degraded, never raising."""
    activity = ad._children_activity_from_token(((3, "terminal"),), now=1000.0)

    assert activity is not None
    (entry,) = activity
    assert entry["api_calls"] == 3 and entry["current_tool"] == "terminal"
    assert entry.get("model") is None
    assert not entry.get("tokens"), "an absent token count must read as unknown, not as zero spend"