"""``plan_usage`` on billing.state / subscription.state: window plans only.

Every other account must behave like main: no picker-usage lookup (and so no extra portal call).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_cli.model_picker_usage as mpu
import hermes_cli.nous_account as na
from tests.usage_window_fixtures import usage_window
from tui_gateway import billing_view as srv

_WINDOW = usage_window(day_used_bp=1000, week_used_bp=500, month_used_bp=250, top_tier_used_bp=0,
                       day_resets_at_ms=1, week_resets_at_ms=2, month_resets_at_ms=3)


@pytest.fixture
def lookups(monkeypatch):
    calls: list[bool] = []

    def _fake(*, warm=False, any_age=False):
        calls.append(warm)
        return mpu.picker_usage_from_account({"logged_in": True, "subscription": {"plan": "Window Plan"},
                                              "usage_window": _WINDOW})

    monkeypatch.setattr(mpu, "nous_picker_usage", _fake)
    return calls


def _snapshot(monkeypatch, info):
    monkeypatch.setattr(na, "cached_nous_portal_account_info", lambda *, any_age=False: info)


@pytest.mark.parametrize("info", [
    None,  # no snapshot (account fetch failed): no lookup
    SimpleNamespace(usage_window=None),  # credit grant / PAYG
    SimpleNamespace(usage_window={"day_used_bp": 1}),  # no gauge list: not a window plan
])
def test_non_window_accounts_never_look_up_plan_usage(monkeypatch, lookups, info):
    _snapshot(monkeypatch, info)
    assert srv._plan_usage_payload(SimpleNamespace(logged_in=True)) is None
    assert lookups == []


def test_logged_out_never_looks_up(monkeypatch, lookups):
    _snapshot(monkeypatch, SimpleNamespace(usage_window=_WINDOW))
    assert srv._plan_usage_payload(SimpleNamespace(logged_in=False)) is None
    assert lookups == []


def test_window_account_gets_every_gauge(monkeypatch, lookups):
    _snapshot(monkeypatch, SimpleNamespace(usage_window=_WINDOW))
    payload = srv._plan_usage_payload(SimpleNamespace(logged_in=True))
    assert lookups == [True]
    assert payload["kind"] == "usage_window"
    assert [g["label"] for g in payload["gauges"]] == ["day", "week", "month", "top_tier"]
    assert [g["scope"] for g in payload["gauges"]] == ["all", "all", "all", "top_tier"]
