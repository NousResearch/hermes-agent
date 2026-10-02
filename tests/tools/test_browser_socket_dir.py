"""Socket-dir path budget tests (#131231).

agent-browser derives the daemon socket filename from the session name and rejects a
socket path beyond 103 bytes, so the FULL path — root + ``/agent-browser-<name>`` +
``/<name>.sock`` — must fit, and the session name itself must be bounded for the
shortest root (/tmp) to be enough.
"""

import os
from unittest.mock import patch

from tools import browser_tool_session as bt_session


def _full_socket_path(socket_dir: str, session_name: str) -> str:
    return os.path.join(socket_dir, f"{session_name}.sock")


class TestSessionSocketDir:
    def test_prefers_scratch_root_when_the_full_path_fits(self):
        # pure path computation — the root does not need to exist
        with patch("tools.browser_tool_session._session_socket_roots", return_value=("/scrtch",)):
            assert bt_session._session_socket_dir("h_abc1234567") == \
                "/scrtch/agent-browser-h_abc1234567"

    def test_falls_back_to_the_short_root_when_the_scratch_root_overflows(self, tmp_path):
        """The #131231 repro shape: a 50-char scratch root (the longest ``socket_safe_tmpdir``
        accepts) with an ordinary cloud session name overflows the 103-byte budget."""
        root = "/" + "x" * 49
        session_name = "hermes_browser-exec-default_126667d7"  # 36 chars, the repro's shape
        with patch("tools.browser_tool_session._session_socket_roots", return_value=(root, "/tmp")):
            socket_dir = bt_session._session_socket_dir(session_name)
        assert socket_dir == f"/tmp/agent-browser-{session_name}"
        assert len(_full_socket_path(socket_dir, session_name)) <= 103

    def test_single_root_that_fits_is_not_double_listed(self):
        # darwin already resolves to /tmp; the dedupe contract is what matters here
        with patch("tools.browser_tool._socket_safe_tmpdir", return_value="/tmp"):
            assert bt_session._session_socket_roots() == ("/tmp",)
        with patch("tools.browser_tool._socket_safe_tmpdir", return_value="/scratch"):
            assert bt_session._session_socket_roots() == ("/scratch", "/tmp")

    def test_windows_has_no_tmp_fallback_and_overflows_stay_on_the_scratch_root(self):
        """A bare ``/tmp/...`` path resolves onto the cwd's drive on Windows (outside
        %TEMP%), and the AF_UNIX budget is a POSIX constraint that never binds there,
        so the scratch root is the only root even for an overflowing session name."""
        root = "C:/Users/admin/AppData/Local/Temp"  # 35 chars — fits the tmpdir bound
        session_name = "hermes_" + "x" * 14 + "-abcdefg_h1234567"  # 38 chars, the truncation shape
        assert len(root) + 2 * len(session_name) + 21 > 103  # would overflow on POSIX
        with patch("tools.browser_tool._socket_safe_tmpdir", return_value=root), \
             patch("tools.browser_tool_session._IS_WINDOWS", True):
            assert bt_session._session_socket_roots() == (root,)
            assert bt_session._session_socket_dir(session_name) == \
                os.path.join(root, f"agent-browser-{session_name}")


class TestSessionNameBound:
    def _session_name(self, task_id):
        from plugins.browser._common import CloudBrowserProvider
        return CloudBrowserProvider._session_name(task_id)

    def test_short_task_id_keeps_the_name_shape(self):
        name = self._session_name("browser-exec-default")
        assert name.startswith("hermes_browser-exec-default_")
        assert len(name) == len("hermes_browser-exec-default_") + 8

    def test_long_task_id_is_bounded_and_still_distinguishes_ids(self):
        long_a = "t" * 80
        long_b = "u" * 80
        name_a1 = self._session_name(long_a)
        name_a2 = self._session_name(long_a)
        name_b = self._session_name(long_b)
        # bounded: /tmp + 2*len(name) + 21 <= 103
        assert len(name_a1) <= 39
        assert 4 + 2 * len(name_a1) + 21 <= 103
        # the hash tail keeps distinct ids distinct and stays stable per id
        assert name_a1[: len("hermes_")] + name_a1.split("_")[-2] == \
            name_a2[: len("hermes_")] + name_a2.split("_")[-2]
        assert name_a1 != name_a2  # uuid suffix still unique per call
        assert name_a1.split("_")[-2] != name_b.split("_")[-2]

    def test_bounded_name_fits_the_short_root_end_to_end(self):
        """Layer 1 + layer 2 together: even a 50-char scratch root and an 80-char task_id
        land on a socket path within the budget."""
        root = "/" + "x" * 49
        session_name = self._session_name("t" * 80)
        with patch("tools.browser_tool_session._session_socket_roots", return_value=(root, "/tmp")):
            socket_dir = bt_session._session_socket_dir(session_name)
        assert len(_full_socket_path(socket_dir, session_name)) <= 103
