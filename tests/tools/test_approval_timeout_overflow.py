"""Regression tests for #83220: oversized approvals.timeout must never
overflow platform wait primitives (macOS time_t OverflowError).

The clamp lives at the single config-read site (_get_approval_timeout), so
every consumer — CLI prompt thread.join, gateway poll deadline, human-wait
ceiling, and the tool_executor authorization gate — is covered at once.
"""

from __future__ import annotations

import threading

from unittest.mock import patch

from agent.deadline import MAX_SAFE_TIMEOUT_S


def _with_configured_timeout(value):
    return patch(
        "tools.approval._get_approval_config",
        return_value={"timeout": value},
    )


class TestApprovalTimeoutOverflowClamp:
    def test_normal_value_passes_through(self):
        from tools.approval import _get_approval_timeout

        with _with_configured_timeout(300):
            assert _get_approval_timeout() == 300

    def test_oversized_value_clamped(self):
        from tools.approval import _get_approval_timeout

        with _with_configured_timeout(10**18):
            assert _get_approval_timeout() == int(MAX_SAFE_TIMEOUT_S)

    def test_invalid_value_falls_back_to_default(self):
        from tools.approval import _get_approval_timeout

        with _with_configured_timeout("soon"):
            assert _get_approval_timeout() == 300

    def test_oversized_float_value_clamped(self):
        # YAML `1e18` arrives as a float, not an int — different int() path
        # than the string/int forms; the clamp must cover it too.
        from tools.approval import _get_approval_timeout

        with _with_configured_timeout(1e18):
            assert _get_approval_timeout() == int(MAX_SAFE_TIMEOUT_S)

    def test_clamp_engagement_logs_warning(self, caplog):
        # Capping silently changes behavior for every consumer; operators
        # must see it happen.
        import tools.approval as approval_mod

        with _with_configured_timeout(10**18):
            with caplog.at_level("WARNING", logger=approval_mod.__name__):
                approval_mod._get_approval_timeout()
        assert "exceeds the platform-safe maximum" in caplog.text

    def test_deadline_import_failure_fails_closed(self, monkeypatch):
        # If agent.deadline ever fails to import, the clamp must fail CLOSED
        # (a finite safe cap) — returning the raw value would re-open the
        # exact time_t overflow this fix exists to prevent.
        import builtins

        from tools.approval import _get_approval_timeout

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "agent.deadline" or name.startswith("agent.deadline."):
                raise ImportError("simulated packaging failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with _with_configured_timeout(10**18):
            value = _get_approval_timeout()
        assert value == 365 * 24 * 3600
        # Still platform-safe for the crashing primitive.
        lock = threading.Lock()
        assert lock.acquire(timeout=value)
        lock.release()

    def test_clamped_value_safe_for_lock_acquire(self):
        # The exact primitive that crashed in #83220: Lock.acquire on macOS
        # converts the relative timeout to an absolute time_t timestamp.
        from tools.approval import _get_approval_timeout

        with _with_configured_timeout(10**18):
            timeout = _get_approval_timeout()
        lock = threading.Lock()
        assert lock.acquire(timeout=timeout)  # would raise OverflowError unclamped
        lock.release()

    def test_clamped_value_safe_for_thread_join(self):
        # Sibling crash site: the CLI prompt fallback joins the input thread
        # with the configured timeout (tools/approval.py get_input path).
        from tools.approval import _get_approval_timeout

        with _with_configured_timeout(10**18):
            timeout = _get_approval_timeout()
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join(timeout=timeout)  # would raise OverflowError unclamped
        assert not t.is_alive()

    def test_human_wait_ceiling_inherits_clamp(self):
        from tools.approval import HUMAN_WAIT_MARGIN_S, human_wait_ceiling

        with _with_configured_timeout(10**18):
            ceiling = human_wait_ceiling()
        assert ceiling == float(int(MAX_SAFE_TIMEOUT_S)) + HUMAN_WAIT_MARGIN_S
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
