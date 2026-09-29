"""Regression for #125337: the live Bot Chat reply-wait budget must be configurable.

One hard-coded 300s covered BOTH legs of a live delivery (queue wait + reply
production), so agent work outliving it settled its mailbox receipt after every
waiter had exited — the reply stranded in a settled ticket nobody reads again.
All live-wait lanes must resolve their budget through
``tools.bot_mode_dm.live_wait_seconds`` so a raised ``bot_mode.live_wait_seconds``
keeps a late reply's waiter alive, and the lanes cannot drift.
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


def test_default_budget_is_the_historical_300s(bot_mode_cfg):
    from tools.bot_mode_dm import _LIVE_WAIT_SECONDS, live_wait_seconds

    assert live_wait_seconds() == _LIVE_WAIT_SECONDS == 300.0


def test_configured_budget_extends_the_wait(bot_mode_cfg):
    from tools.bot_mode_dm import live_wait_seconds

    bot_mode_cfg["bot_mode"] = {"live_wait_seconds": 7200}
    assert live_wait_seconds() == 7200.0


def test_invalid_configured_budget_falls_back(bot_mode_cfg):
    from tools.bot_mode_dm import live_wait_seconds

    for value in ("5m", "", "30m", float("inf"), float("nan"), -1):
        bot_mode_cfg["bot_mode"] = {"live_wait_seconds": value}
        assert live_wait_seconds() == 300.0


def test_local_dm_waiter_waits_on_the_configured_budget(bot_mode_cfg, monkeypatch, capsys):
    """``_wait_live_dm`` (local lane) must pass the configured budget — not the
    module constant — to the shared await primitive."""
    from tools import bot_mode_dm

    bot_mode_cfg["bot_mode"] = {"live_wait_seconds": 5400}
    seen: list[float] = []

    def fake_await(home, delivery_id, timeout, **kwargs):
        seen.append(timeout)
        return {"status": "settled", "delivery_id": delivery_id, "reply": "late reply"}

    monkeypatch.setattr("tools.bot_live_delivery.await_delivery", fake_await)
    assert bot_mode_dm._wait_live_dm("/tmp/no-such-home", "d" * 64) == 0
    assert seen == [5400.0]


def test_gateway_peer_dm_waiter_waits_on_the_configured_budget(bot_mode_cfg, monkeypatch):
    """``_await_live_bot_chat_receipt`` (peer-DM lane) must pass the configured
    budget — not the module constant — to the shared async await primitive."""
    from gateway.platforms import api_server
    from tools import bot_mode_dm

    bot_mode_cfg["bot_mode"] = {"live_wait_seconds": 5400}
    seen: list[float] = []

    async def fake_await_async(home, delivery_id, timeout, **kwargs):
        seen.append(timeout)
        return {"status": "settled", "delivery_id": delivery_id, "reply": "late reply"}

    monkeypatch.setattr("tools.bot_live_delivery.await_delivery_async", fake_await_async)
    # The waiter touches no instance state; None stands in for the adapter.
    result = asyncio.run(
        api_server.APIServerAdapter._await_live_bot_chat_receipt(
            None, "/tmp/no-such-home", {"status": "queued", "delivery_id": "d" * 64}))
    # The waiter passes its remaining wall budget, so a few microseconds under 5400
    # is the correct shape — the point is it derives from the CONFIGURED value.
    assert seen == [pytest.approx(5400.0, abs=0.5)]
    assert result["status"] == "settled"
