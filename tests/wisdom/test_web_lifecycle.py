"""Wisdom's background checker must not install or survive cancellation."""

import asyncio

import pytest

from hermes_cli import web_server_lifecycle as lifecycle


@pytest.mark.parametrize("enabled", [False, True])
def test_checker_obeys_opt_in_and_does_not_apply_updates(monkeypatch, enabled):
    calls = []

    class Service:
        def require_setup(self):
            calls.append("setup")

        def process_professionalism_reviews(self, *, max_jobs):
            calls.append(("reviews", max_jobs))

        def check(self, *, apply_automatic):
            calls.append(("check", apply_automatic))

    async def stop(interval):
        assert interval == 17
        raise asyncio.CancelledError

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"wisdom": {"enabled": enabled}})
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", Service)
    monkeypatch.setattr(lifecycle.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(lifecycle._wisdom_checker_loop(interval=17))
    assert calls == (["setup", ("reviews", 4), ("check", False)] if enabled else [])


def test_checker_recovers_from_service_failure_on_next_cycle(monkeypatch):
    attempts = []
    cycles = []

    class Service:
        def require_setup(self):
            attempts.append("setup")
            if len(attempts) == 1:
                raise RuntimeError("Gateway temporarily unavailable")

        def process_professionalism_reviews(self, **kwargs):
            attempts.append("reviews")

        def check(self, **kwargs):
            attempts.append("check")

    async def next_cycle(interval):
        cycles.append(interval)
        if len(cycles) == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"wisdom": {"enabled": True}})
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", Service)
    monkeypatch.setattr(lifecycle.asyncio, "sleep", next_cycle)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(lifecycle._wisdom_checker_loop(interval=17))
    assert attempts == ["setup", "setup", "reviews", "check"]
    assert cycles == [17, 17]
