"""Regression tests for #83220: oversized approvals.timeout must never
overflow platform wait primitives (macOS time_t OverflowError).

The clamp lives at the single config-read site (_get_approval_timeout), so
every consumer — CLI prompt thread.join, gateway poll deadline, human-wait
ceiling, and the tool_executor authorization gate — is covered at once.

Extended for the Windows follow-up: the platform ceiling is NOT always time_t.
Windows lock waits bottom out in WaitForSingleObject, whose timeout is a DWORD of
milliseconds, so threading.TIMEOUT_MAX is only ~4294967s (~49.7 days) — 7.3x below the
one-year cap that closed #83220. Every bound this module guards must therefore stay
within threading.TIMEOUT_MAX, and the margin added on top of the cap must be re-clamped.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from agent.deadline import MAX_SAFE_TIMEOUT_S


def _with_configured_timeout(value):
    return patch(
        "tools.approval_context._get_approval_config",
        return_value={"timeout": value},
    )


class TestApprovalTimeoutOverflowClamp:
    def test_normal_and_default_values_pass_through(self):
        from tools.approval_context import _get_approval_timeout

        for config, expected in (({}, 300), ({"timeout": 30}, 30), ({"timeout": 60}, 60)):
            with patch("tools.approval_context._get_approval_config", return_value=config):
                assert _get_approval_timeout() == expected

    def test_oversized_value_clamped(self):
        from tools.approval_context import _get_approval_timeout

        with _with_configured_timeout(10**18):
            assert _get_approval_timeout() == int(MAX_SAFE_TIMEOUT_S)

    @pytest.mark.parametrize("value", ["soon", float("inf")])
    def test_invalid_value_falls_back_to_default(self, value):
        from tools.approval_context import _get_approval_timeout

        with _with_configured_timeout(value):
            assert _get_approval_timeout() == 300

    def test_oversized_float_value_clamped(self):
        # YAML `1e18` arrives as a float, not an int — different int() path
        # than the string/int forms; the clamp must cover it too.
        from tools.approval_context import _get_approval_timeout

        with _with_configured_timeout(1e18):
            assert _get_approval_timeout() == int(MAX_SAFE_TIMEOUT_S)

    def test_clamp_engagement_logs_warning(self, caplog):
        # Capping silently changes behavior for every consumer; operators
        # must see it happen.
        import tools.approval as approval_mod
        from tools import approval_context

        with _with_configured_timeout(10**18):
            with caplog.at_level("WARNING", logger=approval_mod.__name__):
                approval_context._get_approval_timeout()
        assert "exceeds the platform-safe maximum" in caplog.text

    def test_deadline_import_failure_fails_closed(self, monkeypatch):
        # If agent.deadline ever fails to import, the clamp must fail CLOSED
        # (a finite safe cap) — returning the raw value would re-open the
        # exact time_t overflow this fix exists to prevent.
        import builtins

        from tools.approval_context import _get_approval_timeout

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "agent.deadline" or name.startswith("agent.deadline."):
                raise ImportError("simulated packaging failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with _with_configured_timeout(10**18):
            value = _get_approval_timeout()
        # A bare one year is itself over the Windows wait limit, so the fail-closed
        # fallback is capped by threading.TIMEOUT_MAX as well.
        assert value == int(min(365 * 24 * 3600, threading.TIMEOUT_MAX))
        # Still platform-safe for the crashing primitive.
        lock = threading.Lock()
        assert lock.acquire(timeout=value)
        lock.release()

    def test_clamped_value_safe_for_lock_acquire(self):
        # The exact primitive that crashed in #83220: Lock.acquire on macOS
        # converts the relative timeout to an absolute time_t timestamp.
        from tools.approval_context import _get_approval_timeout

        with _with_configured_timeout(10**18):
            timeout = _get_approval_timeout()
        lock = threading.Lock()
        assert lock.acquire(timeout=timeout)  # would raise OverflowError unclamped
        lock.release()

    def test_clamped_value_safe_for_thread_join(self):
        # Sibling crash site: the CLI prompt fallback joins the input thread
        # with the configured timeout (tools/approval.py get_input path).
        from tools.approval_context import _get_approval_timeout

        with _with_configured_timeout(10**18):
            timeout = _get_approval_timeout()
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join(timeout=timeout)  # would raise OverflowError unclamped
        assert not t.is_alive()

    def test_human_wait_ceiling_inherits_clamp(self):
        from tools.approval_human_wait import HUMAN_WAIT_MARGIN_S, human_wait_ceiling

        with _with_configured_timeout(10**18):
            ceiling = human_wait_ceiling()
        # The margin is added after the configured value is capped, so the final sum is
        # clamped against the runtime ceiling. Reserved headroom preserves the full margin.
        assert ceiling == min(
            float(int(MAX_SAFE_TIMEOUT_S)) + HUMAN_WAIT_MARGIN_S,
            threading.TIMEOUT_MAX,
        )
        assert ceiling <= threading.TIMEOUT_MAX
        lock = threading.Lock()
        assert lock.acquire(timeout=ceiling)
        lock.release()

    def test_authorization_gate_timeout_safe_and_extends_with_config(self):
        # The gate bound must (a) be platform-safe with an oversized config
        # and (b) still EXTEND beyond the 360s fallback when approvals.timeout
        # is legitimately larger — clamping it down to the fallback would
        # break serialization while a real prompt is still answerable (#79719).
        from agent.tool_executor import (
            _AUTHORIZATION_GATE_LOCK_TIMEOUT_S,
            _authorization_gate_lock_timeout,
        )

        with _with_configured_timeout(10**18):
            bound = _authorization_gate_lock_timeout()
        lock = threading.Lock()
        assert lock.acquire(timeout=bound)
        lock.release()

        with _with_configured_timeout(3600):
            bound = _authorization_gate_lock_timeout()
        assert bound > _AUTHORIZATION_GATE_LOCK_TIMEOUT_S
        assert bound == 3600 + 60.0  # approvals.timeout + HUMAN_WAIT_MARGIN_S


class TestWindowsWaitLimitCeiling:
    """Exercise the real Windows wait ceiling on the Windows CI lane."""

    @pytest.mark.windows_only
    def test_field_value_is_safe_across_thread_wait_consumers(self):
        from agent.deadline import (
            _LOOP_BLOCKED_DUMP_GRACE_S,
            _PLATFORM_WAIT_HEADROOM_S,
            clamp_timeout,
        )
        from agent.tool_executor import _ConcurrentToolAuthorizationGate
        from tools.approval_human_wait import HUMAN_WAIT_MARGIN_S, human_wait_ceiling
        from tools.clarify_gateway import resolve_clarify_timeout

        field_value = 315_360_000
        assert MAX_SAFE_TIMEOUT_S <= threading.TIMEOUT_MAX - _PLATFORM_WAIT_HEADROOM_S
        assert MAX_SAFE_TIMEOUT_S + HUMAN_WAIT_MARGIN_S <= threading.TIMEOUT_MAX
        assert MAX_SAFE_TIMEOUT_S + _LOOP_BLOCKED_DUMP_GRACE_S <= threading.TIMEOUT_MAX
        assert clamp_timeout(field_value) == MAX_SAFE_TIMEOUT_S
        assert resolve_clarify_timeout(
            {"agent": {"clarify_timeout": field_value}}
        ) == int(MAX_SAFE_TIMEOUT_S)

        with _with_configured_timeout(field_value):
            ceiling = human_wait_ceiling()
            gate = _ConcurrentToolAuthorizationGate(session_key="test-overflow")
            assert gate.run(lambda: "ran") == "ran"

        assert ceiling == MAX_SAFE_TIMEOUT_S + HUMAN_WAIT_MARGIN_S
        lock = threading.Lock()
        assert lock.acquire(timeout=ceiling)
        lock.release()
