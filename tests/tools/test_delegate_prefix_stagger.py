"""Tests for the delegation prefix-stagger launch delay.

``delegation.prefix_stagger_seconds`` staggers the first concurrent wave of
a delegate_task batch so sibling children don't all miss the provider
prompt cache at once. Inspired by Claude Code v2.1.229's workflow fan-out
prefix stagger (``CLAUDE_CODE_WORKFLOW_PREFIX_STAGGER_MS``).
"""

import time
from unittest.mock import patch

from tools.delegate_tool import (
    _get_prefix_stagger_seconds,
    _sleep_interruptible,
    _stagger_delay_for,
)


class TestGetPrefixStaggerSeconds:
    def test_default_zero(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            assert _get_prefix_stagger_seconds() == 0.0

    def test_configured_value(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"prefix_stagger_seconds": 2.5},
        ):
            assert _get_prefix_stagger_seconds() == 2.5

    def test_int_accepted(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"prefix_stagger_seconds": 3},
        ):
            assert _get_prefix_stagger_seconds() == 3.0

    def test_negative_clamped_to_zero(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"prefix_stagger_seconds": -4},
        ):
            assert _get_prefix_stagger_seconds() == 0.0

    def test_garbage_fails_open(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"prefix_stagger_seconds": "fast"},
        ):
            assert _get_prefix_stagger_seconds() == 0.0


class TestStaggerDelayFor:
    def test_disabled_means_no_delay(self):
        assert _stagger_delay_for(0, 3, 0.0) == 0.0
        assert _stagger_delay_for(2, 3, 0.0) == 0.0

    def test_first_child_never_delayed(self):
        assert _stagger_delay_for(0, 3, 5.0) == 0.0

    def test_first_wave_ramp(self):
        assert _stagger_delay_for(1, 3, 5.0) == 5.0
        assert _stagger_delay_for(2, 3, 5.0) == 10.0

    def test_beyond_first_wave_not_delayed(self):
        # Children queued past the pool width start when a slot frees;
        # the prefix is cached by then.
        assert _stagger_delay_for(3, 3, 5.0) == 0.0
        assert _stagger_delay_for(7, 3, 5.0) == 0.0


class TestSleepInterruptible:
    class _Parent:
        _interrupt_requested = False

    def test_sleeps_roughly_the_delay(self):
        parent = self._Parent()
        start = time.monotonic()
        _sleep_interruptible(0.25, parent)
        assert time.monotonic() - start >= 0.2

    def test_wakes_early_on_interrupt(self):
        parent = self._Parent()
        parent._interrupt_requested = True
        start = time.monotonic()
        _sleep_interruptible(5.0, parent)
        assert time.monotonic() - start < 1.0

    def test_none_parent_does_not_crash(self):
        start = time.monotonic()
        _sleep_interruptible(0.15, None)
        assert time.monotonic() - start >= 0.1
