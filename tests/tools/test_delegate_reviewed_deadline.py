import pytest

from tools.delegate_tool_deadline import ReviewedDeadline


class Clock:
    now = 100.0
    def __call__(self):
        return self.now


def test_report_and_heartbeat_do_not_extend_deadline():
    clock = Clock()
    lease = ReviewedDeadline(600, clock=clock)
    clock.now = 650
    lease.report(1)
    assert lease.remaining() == 50
    lease.report(2)
    assert lease.remaining() == 50
    clock.now = 700
    assert lease.remaining() == 0
    with pytest.raises(ValueError):
        lease.review(2, approve=True)


def test_parent_approval_resets_from_now_not_accumulated_deadline():
    clock = Clock()
    lease = ReviewedDeadline(600, clock=clock)
    clock.now = 650
    lease.report(1)
    assert lease.review(1, approve=True)["remaining_seconds"] == 600
    clock.now = 701
    assert lease.remaining() == 549
    with pytest.raises(ValueError):
        lease.review(1, approve=True)


def test_stale_denied_and_late_reviews_cannot_renew():
    clock = Clock()
    lease = ReviewedDeadline(600, clock=clock)
    lease.report(1)
    lease.report(2)
    with pytest.raises(ValueError):
        lease.review(1, approve=True)
    clock.now = 650
    lease.review(2, approve=False)
    assert lease.remaining() == 50
    with pytest.raises(ValueError):
        lease.review(2, approve=True)
    lease.close()
    with pytest.raises(ValueError):
        lease.report(3)


@pytest.mark.parametrize("seconds", [None, True, False, "600", 0, -1, float("inf"), float("nan")])
def test_positive_finite_timeout_required(seconds):
    with pytest.raises(ValueError):
        ReviewedDeadline(seconds)
