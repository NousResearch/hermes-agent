"""Regression tests for the ``_memo`` ordering bug in tools/browser_tool_cloud.py.

Before the fix, ``_memo`` set the resolved flag BEFORE ``compute()`` ran, so:

  * a concurrent reader between the flag flip and the cache store saw
    ``resolved=True`` with an unfilled slot and returned the stale module
    default (``None`` / ``False``) as though resolution had succeeded, and
  * a ``compute()`` raise left the flag stuck ``True``, so every later call
    silently returned the unfilled slot instead of retrying.

This is the same ordering bug fixed for ``_get_command_timeout`` under
``#14331``; the sibling ``_cached_browser_cfg`` documents the correct
convention (value stored BEFORE the resolved flag flips).
"""

import threading
from unittest.mock import patch

from tools import browser_tool
from tools import browser_tool_cloud


class TestMemoOrdering:
    def setup_method(self):
        self._orig_flag = browser_tool._headed_mode_resolved
        self._orig_cache = browser_tool._cached_headed_mode
        browser_tool._headed_mode_resolved = False
        browser_tool._cached_headed_mode = None

    def teardown_method(self):
        browser_tool._headed_mode_resolved = self._orig_flag
        browser_tool._cached_headed_mode = self._orig_cache

    def test_flag_not_set_while_compute_is_running(self):
        """A reader observing the flag mid-compute must see unresolved."""
        seen = []

        def probe(*args, **kwargs):
            seen.append(browser_tool._headed_mode_resolved)
            return False

        with patch.object(browser_tool, "_browser_cfg", side_effect=probe):
            browser_tool_cloud._is_headed_mode()

        assert seen == [False]

    def test_compute_raise_leaves_slot_unresolved_and_retries(self):
        """A transient compute failure must not permanently poison the slot."""
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient config read failure")
            return True

        with patch.object(browser_tool, "_browser_cfg", side_effect=flaky):
            try:
                browser_tool_cloud._is_headed_mode()
                raised = False
            except RuntimeError:
                raised = True
            assert raised

            # Pre-fix the flag was stuck True and this call silently returned
            # the unfilled slot (None) instead of recomputing.
            assert browser_tool._headed_mode_resolved is False
            assert browser_tool_cloud._is_headed_mode() is True
            assert browser_tool._headed_mode_resolved is True
            assert browser_tool._cached_headed_mode is True

    def test_concurrent_reader_during_compute_does_not_see_stale_slot(self):
        """A reader arriving while a writer is inside compute must not read
        the unfilled slot."""
        entered = threading.Event()
        release = threading.Event()
        orig_cfg = browser_tool._browser_cfg
        worker_thread = []
        errors = []
        worker_result = []

        def blocking_cfg(*args, **kwargs):
            if threading.current_thread() in worker_thread:
                entered.set()
                assert release.wait(timeout=10)
            return orig_cfg(*args, **kwargs)

        def worker():
            try:
                worker_result.append(browser_tool_cloud._is_headed_mode())
            except Exception as exc:  # pragma: no cover - surfaced by assert below
                errors.append(exc)

        with patch.object(browser_tool, "_browser_cfg", side_effect=blocking_cfg):
            t = threading.Thread(target=worker)
            worker_thread.append(t)
            t.start()
            assert entered.wait(timeout=10), "worker never entered compute"

            main_result = browser_tool_cloud._is_headed_mode()

            release.set()
            t.join(timeout=10)

        assert not errors
        # Post-fix the reader computed its own value (a bool); pre-fix it saw
        # resolved=True and returned the unfilled slot (None).
        assert main_result is not None
        assert worker_result and worker_result[0] is not None

    def test_auto_local_private_urls_reader_never_sees_stale_default(self):
        """The fail-open direction: ``_cached_auto_local_for_private_urls``
        defaults ``True`` at module level, so a mid-compute reader must not
        observe the stale default when the profile configured ``False``."""
        from tools import browser_tool as bt

        orig_flag = bt._auto_local_for_private_urls_resolved
        orig_cache = bt._cached_auto_local_for_private_urls
        bt._auto_local_for_private_urls_resolved = False
        bt._cached_auto_local_for_private_urls = True
        entered = threading.Event()
        release = threading.Event()
        worker_thread = []
        errors = []

        def blocking_cfg(*args, **kwargs):
            if threading.current_thread() in worker_thread:
                entered.set()
                assert release.wait(timeout=10)
            return False  # the profile's configured value

        def worker():
            try:
                browser_tool_cloud._auto_local_for_private_urls()
            except Exception as exc:  # pragma: no cover - surfaced by assert below
                errors.append(exc)

        try:
            with patch.object(browser_tool, "_browser_cfg", side_effect=blocking_cfg):
                t = threading.Thread(target=worker)
                worker_thread.append(t)
                t.start()
                assert entered.wait(timeout=10), "worker never entered compute"

                reader_result = browser_tool_cloud._auto_local_for_private_urls()

                release.set()
                t.join(timeout=10)
        finally:
            release.set()
            bt._auto_local_for_private_urls_resolved = orig_flag
            bt._cached_auto_local_for_private_urls = orig_cache

        assert not errors
        # Post-fix the reader computes False itself; pre-fix it returned the
        # stale True default while the writer was mid-compute.
        assert reader_result is False

    def test_allow_private_urls_raise_retries_instead_of_poisoning(self):
        """``_cached_allow_private_urls`` defaults ``None``: a pre-fix
        compute() raise left the flag stuck True so every later call silently
        returned None (falsy deny) even when the profile allows private URLs."""
        from tools import browser_tool as bt

        orig_flag = bt._allow_private_urls_resolved
        orig_cache = bt._cached_allow_private_urls
        bt._allow_private_urls_resolved = False
        bt._cached_allow_private_urls = None
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient config read failure")
            return True

        try:
            with patch.object(browser_tool, "_browser_cfg", side_effect=flaky):
                try:
                    browser_tool_cloud._allow_private_urls()
                    raised = False
                except RuntimeError:
                    raised = True
                assert raised
                assert bt._allow_private_urls_resolved is False
                assert browser_tool_cloud._allow_private_urls() is True
        finally:
            bt._allow_private_urls_resolved = orig_flag
            bt._cached_allow_private_urls = orig_cache

    def test_routed_profile_scope_bypasses_the_slot(self, tmp_path):
        """Control: under a routed HERMES_HOME override the process slot is not
        consulted even when already resolved."""
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        calls = []
        browser_tool._headed_mode_resolved = True
        browser_tool._cached_headed_mode = False

        token = set_hermes_home_override(str(tmp_path))
        try:
            with patch.object(
                browser_tool, "_browser_cfg", side_effect=lambda *a, **k: calls.append(1) or True
            ):
                assert browser_tool_cloud._is_headed_mode() is True
                assert browser_tool_cloud._is_headed_mode() is True
        finally:
            reset_hermes_home_override(token)

        assert len(calls) == 2
        assert browser_tool._cached_headed_mode is False
