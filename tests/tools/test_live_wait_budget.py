"""Regression for #125337: the live Bot Chat reply-wait budget must be configurable.

One hard-coded 300s covered BOTH legs of a live delivery (queue wait + reply
production), so agent work outliving it settled its mailbox receipt after every
waiter had exited — the reply stranded in a settled ticket nobody reads again.
When ``bot_mode.live_wait_seconds`` is SET, every live-wait lane must resolve
its budget through ``tools.bot_mode_dm.live_wait_seconds`` (one budget spanning
queue wait + reply wait, so the lanes cannot drift); when it is UNSET, every
lane must keep its own default — the gateway lanes' historical 300s, and the
local DM runner's 30-minute cap polled in 300-second windows.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def bot_mode_cfg(monkeypatch):
    """Point the lazy config read at a mutable dict — the seam production resolves.

    ``live_wait_seconds`` late-imports ``tools.bot_relay._bot_mode_cfg`` inside the
    call, so patching the defining module's attribute is where production reads.
    """
    import tools.bot_relay as relay

    cfg: dict = {}
    monkeypatch.setattr(relay, "_bot_mode_cfg",
                        lambda key, *, loader: (cfg.get("bot_mode") or {}).get(key))
    return cfg


def test_unset_knob_resolves_to_none(bot_mode_cfg):
    """Unset -> None: every lane falls back to its own built-in budget."""
    from tools.bot_mode_dm import live_wait_seconds

    assert live_wait_seconds() is None


def test_configured_budget_overrides_every_lane(bot_mode_cfg):
    from tools.bot_mode_dm import live_wait_seconds

    bot_mode_cfg["bot_mode"] = {"live_wait_seconds": 7200}
    assert live_wait_seconds() == 7200.0


def test_invalid_configured_budget_resolves_to_none(bot_mode_cfg):
    from tools.bot_mode_dm import live_wait_seconds

    for value in ("5m", "", "30m", float("inf"), float("nan"), -1):
        bot_mode_cfg["bot_mode"] = {"live_wait_seconds": value}
        assert live_wait_seconds() is None


def test_local_dm_waiter_unset_keeps_upstream_cap_and_windows(bot_mode_cfg, monkeypatch):
    """Unset knob: the local runner keeps upstream's 30-minute cap as the deadline
    and the 300-second poll windows — a raised cap must NOT silently shrink here."""
    from tools import bot_mode_dm

    seen: list[float] = []

    def fake_await(home, delivery_id, timeout, **kwargs):
        seen.append(timeout)
        return {"status": "settled", "delivery_id": delivery_id, "reply": "late reply"}

    monkeypatch.setattr("tools.bot_live_delivery.await_delivery", fake_await)
    assert bot_mode_dm._wait_live_dm("/tmp/no-such-home", "d" * 64) == 0
    assert seen == [300.0]


def test_local_dm_waiter_set_passes_one_full_budget_window(bot_mode_cfg, monkeypatch):
    """Set knob: deadline AND poll window widen to the single budget, so one await
    call can cover the whole budget instead of slicing it into 300-second polls."""
    from tools import bot_mode_dm

    bot_mode_cfg["bot_mode"] = {"live_wait_seconds": 5400}
    seen: list[float] = []

    def fake_await(home, delivery_id, timeout, **kwargs):
        seen.append(timeout)
        return {"status": "settled", "delivery_id": delivery_id, "reply": "late reply"}

    monkeypatch.setattr("tools.bot_live_delivery.await_delivery", fake_await)
    assert bot_mode_dm._wait_live_dm("/tmp/no-such-home", "d" * 64) == 0
    # The window is min(budget, deadline - now): a few microseconds under the budget.
    assert seen == [pytest.approx(5400.0, abs=0.5)]


def test_gateway_peer_dm_waiter_set_uses_the_configured_budget(bot_mode_cfg, monkeypatch):
    """``_await_live_bot_chat_receipt`` (peer-DM lane) must pass the configured
    budget — not the module constant — as its deadline."""
    from gateway.platforms import api_server
    from types import SimpleNamespace

    bot_mode_cfg["bot_mode"] = {"live_wait_seconds": 5400}
    seen: list[float] = []

    async def fake_await_async(home, delivery_id, timeout, **kwargs):
        seen.append(timeout)
        return {"status": "settled", "delivery_id": delivery_id, "reply": "late reply"}

    monkeypatch.setattr("tools.bot_live_delivery.await_delivery_async", fake_await_async)
    # A runner without session_authority sends the waiter down the legacy mailbox
    # door (await_delivery_async, patched above); the waiter touches no other state.
    adapter = SimpleNamespace(gateway_runner=SimpleNamespace())
    result = asyncio.run(
        api_server.APIServerAdapter._await_live_bot_chat_receipt(
            adapter, "/tmp/no-such-home", {"status": "queued", "delivery_id": "d" * 64}))
    # The waiter passes its remaining wall budget, so a few microseconds under 5400
    # is the correct shape — the point is it derives from the CONFIGURED value.
    assert seen == [pytest.approx(5400.0, abs=0.5)]
    assert result["status"] == "settled"


def test_gateway_peer_dm_waiter_unset_keeps_the_300s_budget(bot_mode_cfg, monkeypatch):
    from gateway.platforms import api_server
    from types import SimpleNamespace

    seen: list[float] = []

    async def fake_await_async(home, delivery_id, timeout, **kwargs):
        seen.append(timeout)
        return {"status": "settled", "delivery_id": delivery_id, "reply": "late reply"}

    monkeypatch.setattr("tools.bot_live_delivery.await_delivery_async", fake_await_async)
    adapter = SimpleNamespace(gateway_runner=SimpleNamespace())
    result = asyncio.run(
        api_server.APIServerAdapter._await_live_bot_chat_receipt(
            adapter, "/tmp/no-such-home", {"status": "queued", "delivery_id": "d" * 64}))
    assert seen == [pytest.approx(300.0, abs=0.5)]
    assert result["status"] == "settled"
