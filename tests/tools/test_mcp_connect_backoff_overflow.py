"""A connect cooldown keeps its cap after an arbitrarily long failure streak.

``_record_connect_failure`` computed ``min(30.0 * (2 ** (n - 1)), 600.0)``. Python evaluates the
product before the ``min``, so the 1025th consecutive failure raised ``OverflowError: int too
large to convert to float``: the cooldown was never recorded and the exception escaped the
discovery pass. The clamp must not change any backoff the 600s cap already allowed.
"""

import time

import pytest

from tools.mcp_tool_scope import _server_key


@pytest.fixture
def _clean_failure_state():
    from tools import mcp_tool

    def _reset():
        with mcp_tool._lock:
            key = _server_key("srv")
            mcp_tool._server_connect_failures.pop(key, None)
            mcp_tool._server_connect_retry_after.pop(key, None)

    _reset()
    yield
    _reset()


@pytest.mark.no_isolate
def test_backoff_stays_capped_after_many_consecutive_failures(_clean_failure_state):
    """2000 failures in a row: the cooldown stays at the cap instead of raising at #1025."""
    from tools import mcp_tool
    from tools.mcp_tool_discovery import _record_connect_failure

    for _ in range(2000):
        _record_connect_failure("srv")

    key = _server_key("srv")
    assert mcp_tool._server_connect_failures[key] == 2000
    remaining = mcp_tool._server_connect_retry_after[key] - time.monotonic()
    assert remaining == pytest.approx(600.0, abs=5.0)


@pytest.mark.no_isolate
def test_short_streaks_keep_the_upstream_backoff_curve(_clean_failure_state):
    """n=1..6 still yields 30/60/120/240/480/600s, so the clamp only governs absurd streaks."""
    from tools import mcp_tool
    from tools.mcp_tool_discovery import _record_connect_failure

    seen = []
    for _ in range(6):
        _record_connect_failure("srv")
        key = _server_key("srv")
        seen.append(mcp_tool._server_connect_retry_after[key] - time.monotonic())

    assert seen == pytest.approx([30.0, 60.0, 120.0, 240.0, 480.0, 600.0], abs=1.0)
