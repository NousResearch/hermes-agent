"""Which creationflags a not-owned background job gets on Windows.

The layer-3 kill-on-close job (#132069) finally holds what a holder spawns, so
_spawn_job's not-owned branch — the jobs that must OUTLIVE the holder (#41225
persist_on_release, gateway checkpoint re-adoption, managed-worker handover) —
has to leave that job at spawn: CREATE_BREAKAWAY_FROM_JOB, the same escape the
Windows restart watcher uses. These tests pin the flags logic itself, mocked,
on every platform; the kernel truth (IsProcessInJob on a real spawn) is pinned
live on Windows in tests/hermes_cli/test_process_identity.py.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from tools.process_registry_ownership import ProcessOwnershipMixin


class _NotOwnedRegistry(ProcessOwnershipMixin):
    """The Windows truth without the import chain: nothing is owned (layer 3 covers it)."""

    def _owns_job(self, persist_on_release: bool) -> bool:
        return False


class _OwnedRegistry(ProcessOwnershipMixin):
    def _owns_job(self, persist_on_release: bool) -> bool:
        return True


class _PopenProbe:
    """Records the kwargs of each subprocess.Popen call; optionally raises on the first."""

    def __init__(self, fail_first_with: BaseException | None = None):
        self.calls: list[dict] = []
        self._fail_first_with = fail_first_with

    def __call__(self, argv, **kwargs):
        self.calls.append(kwargs)
        if self._fail_first_with is not None:
            exc, self._fail_first_with = self._fail_first_with, None
            raise exc
        return SimpleNamespace(pid=4242)


def test_not_owned_job_breaks_away_from_the_kill_on_close_job_on_windows():
    from hermes_cli import _subprocess_compat as sc

    probe = _PopenProbe()
    with patch.object(sc, "IS_WINDOWS", True), patch.object(subprocess, "Popen", probe):
        _NotOwnedRegistry()._spawn_job(
            SimpleNamespace(id="s"), ["cmd"], persist_on_release=True,
            creationflags=sc.windows_hide_flags())

    assert len(probe.calls) == 1
    assert probe.calls[0]["start_new_session"] is True
    # The hiding flags are upgraded to detaching ones (breakaway + group + no window), not replaced.
    assert probe.calls[0]["creationflags"] == sc.windows_hide_flags() | sc.windows_detach_flags()


def test_not_owned_job_retries_once_without_breakaway_when_the_outer_job_forbids_it():
    from hermes_cli import _subprocess_compat as sc

    probe = _PopenProbe(fail_first_with=OSError("breakaway denied"))
    with patch.object(sc, "IS_WINDOWS", True), patch.object(subprocess, "Popen", probe):
        _NotOwnedRegistry()._spawn_job(
            SimpleNamespace(id="s"), ["cmd"], persist_on_release=True,
            creationflags=sc.windows_hide_flags())

    assert len(probe.calls) == 2
    assert probe.calls[1]["start_new_session"] is True
    assert probe.calls[1]["creationflags"] == (
        sc.windows_hide_flags() | sc.windows_detach_flags_without_breakaway())


def test_not_owned_posix_spawn_keeps_its_exact_old_shape():
    from hermes_cli import _subprocess_compat as sc

    probe = _PopenProbe()
    with patch.object(sc, "IS_WINDOWS", False), patch.object(subprocess, "Popen", probe):
        _NotOwnedRegistry()._spawn_job(SimpleNamespace(id="s"), ["cmd"], persist_on_release=True)

    # POSIX: byte-for-byte the pre-breakaway call — no creationflags key at all.
    assert probe.calls == [{"start_new_session": True}]


def test_owned_job_is_spawned_owned_and_never_detached():
    owned: list[tuple] = []

    def fake_spawn_owned(argv, purpose, **kwargs):
        owned.append((argv, purpose, kwargs))
        return SimpleNamespace(pid=1)

    probe = _PopenProbe()
    import tools.process_registry_ownership as pr_own

    with patch.object(pr_own, "spawn_owned", fake_spawn_owned), \
         patch.object(subprocess, "Popen", probe):
        session = SimpleNamespace(id="s")
        _OwnedRegistry()._spawn_job(session, ["cmd"], persist_on_release=False)

    # Owned jobs stay inside the holder's kill-on-close job: that net is their ownership on Windows.
    assert owned == [(["cmd"], "terminal-background", {})]
    assert session._owned_pgid == 1
    assert probe.calls == []
