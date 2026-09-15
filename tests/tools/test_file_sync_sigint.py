"""Cross-platform regression for the deferred-SIGINT re-delivery in sync-back.

``_sync_back_once`` defers a Ctrl+C that lands mid-sync, then re-delivers it once
the sync completes. It must do so via ``signal.raise_signal`` — which invokes the
handler through C ``raise()`` on every platform — and NOT via
``os.kill(os.getpid(), signal.SIGINT)``: on Windows the latter routes SIGINT (2)
to ``TerminateProcess`` and hard-kills the whole CLI instead of raising
``KeyboardInterrupt``.

Unlike ``test_file_sync_back.py`` this module does not depend on ``fcntl`` (the
locked sync body is stubbed), so it runs on Windows too — the platform the bug
actually manifests on.
"""

from __future__ import annotations

import os
import signal

from tools.environments.file_sync import FileSyncManager


def _make_manager() -> FileSyncManager:
    return FileSyncManager(
        get_files_fn=lambda: {},
        upload_fn=lambda *a, **k: None,
        delete_fn=lambda *a, **k: None,
    )


def test_deferred_sigint_redelivered_via_raise_signal(tmp_path, monkeypatch):
    mgr = _make_manager()

    # Simulate a Ctrl+C arriving during the sync body: invoke the deferring
    # handler that _sync_back_once installed, so `deferred_sigint` is populated.
    def fake_locked(lock_path):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

    monkeypatch.setattr(mgr, "_sync_back_locked", fake_locked)

    raised: list[int] = []
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "tools.environments.file_sync.signal.raise_signal", raised.append
    )
    monkeypatch.setattr(
        "tools.environments.file_sync.os.kill",
        lambda pid, sig: killed.append((pid, sig)),
    )

    mgr._sync_back_once(tmp_path / "sync.lock")

    # The deferred Ctrl+C is re-delivered cross-platform via raise_signal,
    assert raised == [signal.SIGINT]
    # and never through os.kill(getpid, SIGINT) (which hard-kills on Windows).
    assert (os.getpid(), signal.SIGINT) not in killed
