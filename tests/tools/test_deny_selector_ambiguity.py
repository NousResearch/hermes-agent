"""Unresolved preservation conflict: RED only, no actual payload execution."""
from tests.tools.test_background_approval_routing import rig
from tests.tools.test_approval_followup import foreground_wait
from tools import approval as ap


def test_malformed_bound_deny_selector_must_not_resolve_unrelated_foreground(rig):
    rig.spawn(); rig.start(); assert rig.notified.wait(2)
    rid = ap.list_gateway_approvals(rig.key)[0]['request_id']
    with foreground_wait(rig) as (entry, results):
        rig.answer('deny', rid[:-1])
        assert entry.result is None, 'legacy /deny <reason> fallback denied unrelated wait'
