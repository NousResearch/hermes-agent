"""A check_fn that fails inside the grace window is not re-probed on every lookup."""

import time as _time

import pytest

import tools.registry as reg


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __getattr__(self, name):
        return getattr(_time, name)

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(reg, "time", c)
    reg.invalidate_check_fn_cache()
    yield c
    reg.invalidate_check_fn_cache()


def test_flaky_check_fn_is_not_reprobed_on_every_lookup(clock):
    calls = []

    def check():
        calls.append(clock.now)
        return len(calls) == 1  # up once, then down

    assert reg._check_fn_cached(check) is True
    clock.now += reg._CHECK_FN_TTL_SECONDS + 1
    assert reg._check_fn_cached(check) is True  # failed within grace: last-good served
    assert len(calls) == 2
    clock.now += 1
    assert reg._check_fn_cached(check) is True
    assert len(calls) == 2  # served from the bounded grace entry, no probe storm
    clock.now += 5
    reg._check_fn_cached(check)
    assert len(calls) == 3  # the grace entry expires and the backend is re-probed


def test_healthy_check_fn_is_cached_for_the_ttl(clock):
    """Positive control: a passing probe is cached, then re-probed after the TTL."""
    calls = []

    def check():
        calls.append(clock.now)
        return True

    reg._check_fn_cached(check)
    clock.now += 1
    reg._check_fn_cached(check)
    assert len(calls) == 1
    clock.now += reg._CHECK_FN_TTL_SECONDS
    reg._check_fn_cached(check)
    assert len(calls) == 2

