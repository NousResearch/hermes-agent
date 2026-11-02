"""#121048: a failed ``async_delivery_supported()`` probe must fail CLOSED.

The capability probe guards durable delivery of a detached result. Every other
branch that cannot prove a durable consumer falls back to synchronous execution;
the probe's exception path used to treat an error as "supported" (delegate_task)
or to skip the gate entirely (cron background runs)."""

from __future__ import annotations

from unittest.mock import patch

import tools.cronjob_tools as cronjob_tools
from tools.delegate_tool_dispatch import _resolve_async_wake_sid


def test_delegate_probe_error_falls_back_to_inline():
    with patch("gateway.session_context.async_delivery_supported",
               side_effect=RuntimeError("broken session context")):
        assert _resolve_async_wake_sid("sess-1", True) is None, (
            "a failed capability read must run inline, not detach onto an unproven lane")


def test_delegate_probe_supported_still_detaches_to_push():
    with patch("gateway.session_context.async_delivery_supported", return_value=True):
        assert _resolve_async_wake_sid("sess-1", True) == ""


def test_delegate_probe_unsupported_keeps_history_lane():
    with patch("gateway.session_context.async_delivery_supported", return_value=False):
        assert _resolve_async_wake_sid("sess-1", True) == "sess-1"
        assert _resolve_async_wake_sid("sess-1", False) is None


def test_cron_probe_error_skips_background_dispatch():
    with patch("tools.cronjob_tools._reap_stale_executions"), \
         patch("gateway.session_context.async_delivery_supported",
               side_effect=RuntimeError("broken session context")):
        assert cronjob_tools._try_dispatch_background_run(
            {"id": "job-1", "name": "job one"}) is None, (
            "a failed capability read must not claim-and-dispatch on an unproven lane")


def test_cron_probe_unsupported_skips_background_dispatch():
    with patch("tools.cronjob_tools._reap_stale_executions"), \
         patch("gateway.session_context.async_delivery_supported", return_value=False):
        assert cronjob_tools._try_dispatch_background_run(
            {"id": "job-1", "name": "job one"}) is None
