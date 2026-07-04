import logging

import pytest

from tui_gateway.loop_monitor import (
    install_event_loop_lag_watchdog,
    log_loop_lag_if_stalled,
)


class _FakeHandle:
    def __init__(self):
        self._cancelled = False

    def cancelled(self):
        return self._cancelled


class _FakeLoop:
    def __init__(self):
        self.now = 100.0
        self.calls = []

    def time(self):
        return self.now

    def call_later(self, delay, callback, *args):
        handle = _FakeHandle()
        self.calls.append((delay, callback, args, handle))
        return handle


def test_log_loop_lag_if_stalled_threshold_and_component(caplog):
    logger = logging.getLogger("tests.loop_monitor")

    with caplog.at_level(logging.WARNING, logger="tests.loop_monitor"):
        assert not log_loop_lag_if_stalled(logger, lag_s=5.0, warn_after_s=5.0)
        assert log_loop_lag_if_stalled(
            logger,
            lag_s=5.1,
            warn_after_s=5.0,
            component="tui ws",
        )

    assert "tui ws event loop stalled 5.1s" in caplog.text
    assert "GIL pressure suspected" in caplog.text


def test_install_event_loop_lag_watchdog_rearms_and_logs_late_ticks(caplog):
    loop = _FakeLoop()
    logger = logging.getLogger("tests.loop_monitor.install")

    handle = install_event_loop_lag_watchdog(
        loop=loop,
        logger=logger,
        interval_s=2.0,
        warn_after_s=5.0,
    )

    assert handle is loop.calls[0][3]
    delay, callback, args, _ = loop.calls[0]
    assert delay == pytest.approx(2.0)
    assert args == (102.0,)

    loop.now = 109.25
    with caplog.at_level(logging.WARNING, logger="tests.loop_monitor.install"):
        callback(*args)

    assert "event loop stalled" in caplog.text
    assert len(loop.calls) == 2
    next_delay, _, next_args, _ = loop.calls[1]
    assert next_delay == pytest.approx(2.0)
    assert next_args == (111.25,)
