"""agent-browser socket paths must fit its 103-byte limit for any root ``socket_safe_tmpdir`` allows.

agent-browser binds ``<AGENT_BROWSER_SOCKET_DIR>/<session>.sock`` and refuses longer paths with
"Use a shorter session name or set AGENT_BROWSER_SOCKET_DIR to a shorter path". Hermes sets the
socket dir to ``<root>/agent-browser-<session>``, so the session name counts twice. The old
real-profile name ("hermes-real-profile") overflowed under every named profile's scratch dir
(``~/.hermes/profiles/cfo/cache/scratch`` -> 105 bytes), breaking real-profile browsing in
profile cron workers while gateway turns (default-home scratch) worked.
"""
import os
from unittest.mock import patch

import pytest

from hermes_constants import SOCKET_TMPDIR_MAX_LEN

AGENT_BROWSER_MAX_SOCKET_PATH = 103


def _socket_path(root: str, session: str) -> str:
    return os.path.join(root, f"agent-browser-{session}", f"{session}.sock")


def test_real_profile_session_socket_fits_for_longest_allowed_root():
    import tools.browser_tool as bt

    root = "/" + "x" * (SOCKET_TMPDIR_MAX_LEN - 1)
    assert len(_socket_path(root, bt._REAL_PROFILE_SESSION).encode()) <= AGENT_BROWSER_MAX_SOCKET_PATH


@pytest.mark.parametrize("prefix", ["h", "rp", "lp", "cdp", "h_cfb"])
def test_random_session_sockets_fit_for_longest_allowed_root(prefix):
    root = "/" + "x" * (SOCKET_TMPDIR_MAX_LEN - 1)
    session = f"{prefix}_{'a' * 10}"
    assert len(_socket_path(root, session).encode()) <= AGENT_BROWSER_MAX_SOCKET_PATH


def test_named_profile_scratch_real_profile_socket_fits():
    """The regression: a profile cron worker's TMPDIR is ``<home>/profiles/<name>/cache/scratch``."""
    import tools.browser_tool as bt

    root = "/home/alice/.hermes/profiles/cfo/cache/scratch"  # 46 chars, a typical named-profile root
    assert len(root) <= SOCKET_TMPDIR_MAX_LEN  # socket_safe_tmpdir keeps it, no /tmp fallback
    assert len(_socket_path(root, bt._REAL_PROFILE_SESSION).encode()) <= AGENT_BROWSER_MAX_SOCKET_PATH


def test_legacy_real_profile_dir_is_still_reaped_when_owner_is_dead(tmp_path):
    import tools.browser_tool as bt
    from tools.browser_tool_lifecycle import _reap_orphaned_browser_sessions

    legacy = bt._LEGACY_REAL_PROFILE_SESSIONS[0]
    d = tmp_path / f"agent-browser-{legacy}"
    d.mkdir()
    (d / f"{legacy}.pid").write_text("4242")
    (d / f"{legacy}.owner_pid").write_text("99999")
    killed = []

    with patch.object(bt, "_socket_safe_tmpdir", return_value=str(tmp_path)), \
         patch("gateway.status._pid_exists", side_effect=lambda pid: pid == 4242), \
         patch("gateway.status.get_process_start_time", return_value=777), \
         patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
         patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
               side_effect=lambda pid, expected_start=None: killed.append(pid)):
        _reap_orphaned_browser_sessions()

    assert killed == [4242]
    assert not d.exists()
