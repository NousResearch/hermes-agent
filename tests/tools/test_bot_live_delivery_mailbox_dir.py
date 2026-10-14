"""The live-delivery mailbox lock dir (``tools.bot_live_delivery._locked``).

POSIX tightens the directory to 0700 on every entry. On Windows a chmod rewrites the NTFS ACL and
the owner then gets Permission denied opening ``.lock`` (the Telegram adapter crash-looped on it,
tracker #134889 W2), so the lock must stay re-enterable there.
"""
import stat

import pytest

from tools.bot_live_delivery import _locked, _root


@pytest.mark.platforms("posix")
def test_mailbox_dir_is_tightened_to_owner_only(tmp_path):
    root = _root(tmp_path)
    root.mkdir(parents=True, mode=0o755)
    root.chmod(0o755)
    with _locked(tmp_path):
        pass
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


@pytest.mark.platforms("windows")
def test_mailbox_lock_is_reenterable_on_windows(tmp_path):
    for _ in range(3):  # the live poller re-enters it twice a second
        with _locked(tmp_path):
            pass
    (_root(tmp_path) / ".lock").open("a").close()
