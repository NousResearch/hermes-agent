"""The live-delivery mailbox lock dir (``tools.bot_live_delivery._locked``).

POSIX tightens the directory to 0700 on every entry. On Windows the owner-only DACL comes from
``mkdir(mode=0o700)`` (CPython 3.13+) and chmod only toggles the read-only attribute, so the
Windows tests pin that the owner keeps re-entering the lock, including in a dir that the older
chmod-on-every-entry code created.
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


@pytest.mark.platforms("windows")
def test_mailbox_dir_chmodded_by_older_code_stays_lockable_on_windows(tmp_path):
    root = _root(tmp_path)
    root.parent.mkdir(parents=True)
    root.mkdir(mode=0o700)
    root.chmod(0o700)  # what _locked did on every entry before it skipped chmod on Windows
    for _ in range(3):
        with _locked(tmp_path):
            pass
