"""Sync-back must not stall the thread that garbage-collects an environment, and must not cost
staged x mapped filesystem resolves.

Incident: a gateway on the SSH backend exited 75 (event loop missed its liveness probes) several
times a day. Thread dumps showed the asyncio loop thread inside ``BaseEnvironment.__del__`` ->
``SSHEnvironment.cleanup`` -> ``FileSyncManager.sync_back`` -> ``fcntl.flock`` while the idle
reaper thread held the profile-wide sync lock in ``_infer_host_path`` -> ``Path.resolve``; each
sync-back of a remote ``.hermes`` with ~5k unmapped files took 10-30 minutes.
"""

import asyncio
import gc
import io
import tarfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.environments import file_sync
from tools.environments import ssh as ssh_env
from tools.environments.base import BaseEnvironment
from tools.environments.file_sync import FileSyncManager, _SyncBackPathIndex


# ---------------------------------------------------------------------------
# _SyncBackPathIndex: same answers as the linear scans
# ---------------------------------------------------------------------------


def _manager(mapping):
    return FileSyncManager(get_files_fn=lambda: mapping, upload_fn=MagicMock(), delete_fn=MagicMock())


def _mapping(tmp_path: Path) -> tuple[list[tuple[str, str]], set[str]]:
    host = tmp_path / "host"
    creds = host / "creds"
    creds.mkdir(parents=True)
    (creds / "token.json").write_text("{}")
    mapping = [
        # An upload-only credential that sits at the remote .hermes root: inference must skip it,
        # or every unmapped file under .hermes would be pulled into the credentials dir.
        (str(creds / "token.json"), "/home/u/.hermes/token.json"),
        (str(host / "skills" / "a" / "SKILL.md"), "/home/u/.hermes/skills/a/SKILL.md"),
        (str(host / "skills" / "a" / "ref.md"), "/home/u/.hermes/skills/a/ref.md"),
        # Same remote dir as the next entry with a different host parent: the first entry wins.
        (str(host / "skills-x" / "b" / "one.md"), "/home/u/.hermes/skills/b/one.md"),
        (str(host / "skills-y" / "b" / "two.md"), "/home/u/.hermes/skills/b/two.md"),
        (str(host / "skills" / "c" / "d" / "deep.md"), "/home/u/.hermes/skills/c/d/deep.md"),
        (str(host / "cache" / "web" / "page.html"), "/home/u/.hermes/cache/web/page.html"),
        # Duplicate remote path: _resolve_host_path returns the first.
        (str(host / "dup-first.md"), "/home/u/.hermes/skills/a/SKILL.md"),
        # Odd shapes the prefix test must treat exactly like the scan.
        (str(host / "rootfile"), "/rootfile"),
        (str(host / "relative"), "relative"),
    ]
    upload_only = {file_sync._resolve_host_path_str(str(creds / "token.json"))}
    return mapping, upload_only


_PROBES = [
    "/home/u/.hermes/skills/a/SKILL.md",
    "/home/u/.hermes/skills/a/new.md",
    "/home/u/.hermes/skills/a/sub/new.md",
    "/home/u/.hermes/skills/b/three.md",
    "/home/u/.hermes/skills/c/d/e/f.md",
    "/home/u/.hermes/skills/c/x.md",
    "/home/u/.hermes/skills/zz/new.md",
    "/home/u/.hermes/cache/web/new.html",
    "/home/u/.hermes/skill-rollbacks/old/SKILL.md",
    "/home/u/.hermes/token.json",
    "/home/u/.hermes/other.json",
    "/x",
    "//double",
    "relative/child",
    "",
]


@pytest.mark.parametrize("remote_path", _PROBES)
def test_index_matches_linear_lookups(tmp_path, remote_path):
    mapping, upload_only = _mapping(tmp_path)
    mgr = _manager(mapping)
    index = _SyncBackPathIndex(mapping, upload_only)

    assert index.resolve(remote_path) == mgr._resolve_host_path(remote_path, mapping)
    assert index.infer(remote_path) == mgr._infer_host_path(
        remote_path, mapping, upload_only_host_paths=upload_only)


def test_index_skips_upload_only_entry_but_keeps_a_later_one_for_the_same_dir(tmp_path):
    host = tmp_path / "host"
    secret, normal = str(host / "a" / "secret.json"), str(host / "b" / "normal.md")
    mapping = [(secret, "/r/.hermes/secret.json"), (normal, "/r/.hermes/normal.md")]
    upload_only = {file_sync._resolve_host_path_str(secret)}
    mgr = _manager(mapping)
    index = _SyncBackPathIndex(mapping, upload_only)

    expected = mgr._infer_host_path("/r/.hermes/new.md", mapping, upload_only_host_paths=upload_only)
    assert expected == str(host / "b" / "new.md")
    assert index.infer("/r/.hermes/new.md") == expected


# ---------------------------------------------------------------------------
# sync_back cost: one resolve per mapping entry, not per (staged file x mapping entry)
# ---------------------------------------------------------------------------


def _tar_download(files: dict[str, bytes]):
    def download(dest: Path) -> None:
        with tarfile.open(dest, "w") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return download


def test_sync_back_resolves_scale_with_files_not_their_product(tmp_path, monkeypatch):
    mapped, unmapped = 300, 400
    host_skills = tmp_path / "host" / "skills"
    mapping = []
    for i in range(mapped):
        host = host_skills / f"s{i}" / "SKILL.md"
        host.parent.mkdir(parents=True)
        host.write_text(f"skill {i}")
        mapping.append((str(host), f"/home/u/.hermes/skills/s{i}/SKILL.md"))
    remote = {f"home/u/.hermes/skills/s{i}/SKILL.md": f"skill {i}".encode() for i in range(mapped)}
    remote.update({f"home/u/.hermes/skill-rollbacks/r{i}/SKILL.md": b"old" for i in range(unmapped)})
    remote["home/u/.hermes/skills/s0/new.md"] = b"new remote file"

    mgr = FileSyncManager(get_files_fn=lambda: mapping, upload_fn=MagicMock(), delete_fn=MagicMock(),
                          bulk_download_fn=_tar_download(remote))
    for host, remote_path in mapping:
        mgr._pushed_hashes[remote_path] = file_sync._sha256_file(host)

    calls = 0
    real = file_sync._resolve_host_path_str

    def counting(path: str) -> str:
        nonlocal calls
        calls += 1
        return real(path)

    monkeypatch.setattr(file_sync, "_resolve_host_path_str", counting)
    mgr.sync_back(hermes_home=tmp_path / ".hermes")

    assert (host_skills / "s0" / "new.md").read_bytes() == b"new remote file"
    assert not (tmp_path / "host" / "skill-rollbacks").exists()
    # Linear scans paid ~unmapped * mapped (120k here); the index pays about mapped + applied.
    assert calls <= mapped + 10, calls


# ---------------------------------------------------------------------------
# __del__ never blocks the collecting thread on a sync-back
# ---------------------------------------------------------------------------


class _Env(BaseEnvironment):
    def __init__(self, sync_manager=None, on_cleanup=None):
        super().__init__(cwd="/tmp", timeout=10)
        self._sync_manager = sync_manager
        self._on_cleanup = on_cleanup

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError

    def cleanup(self):
        if self._on_cleanup:
            self._on_cleanup()


def test_collecting_an_env_with_a_sync_manager_does_not_wait_for_its_cleanup():
    release, done = threading.Event(), threading.Event()
    seen = {}

    def blocking_cleanup():
        seen["thread"] = threading.current_thread()
        release.wait(10)  # stands in for flock() held by the idle reaper's sync-back
        done.set()

    env = _Env(sync_manager=object(), on_cleanup=blocking_cleanup)
    started = time.monotonic()
    del env
    gc.collect()
    assert time.monotonic() - started < 2.0, "the collecting thread waited on the sync-back"

    release.set()
    assert done.wait(10)
    assert seen["thread"] is not threading.current_thread()
    assert seen["thread"].daemon


def test_collecting_an_env_without_a_sync_manager_still_cleans_up_inline():
    seen = {}
    env = _Env(sync_manager=None, on_cleanup=lambda: seen.setdefault("thread", threading.current_thread()))
    del env
    gc.collect()
    assert seen["thread"] is threading.current_thread()


def test_event_loop_keeps_running_while_a_collected_env_waits_for_the_sync_lock():
    """The incident shape: the env is dropped while the loop runs, its sync-back waits on a lock."""
    lock = threading.Lock()
    lock.acquire()  # the idle reaper's sync-back, still running
    cleaned = threading.Event()

    def cleanup_waiting_for_lock():
        if lock.acquire(timeout=5):  # bounded so a regression fails instead of hanging the suite
            cleaned.set()
            lock.release()

    async def main() -> float:
        started = time.monotonic()
        env = _Env(sync_manager=object(), on_cleanup=cleanup_waiting_for_lock)
        del env
        gc.collect()  # GC runs the finalizer on the loop thread
        for _ in range(5):
            await asyncio.sleep(0)  # the loop still turns
        return time.monotonic() - started

    assert asyncio.run(main()) < 2.0
    assert not cleaned.is_set()
    lock.release()
    assert cleaned.wait(10)


# ---------------------------------------------------------------------------
# SSHEnvironment.cleanup() pulls back once
# ---------------------------------------------------------------------------


@pytest.fixture
def _ssh_env(monkeypatch, tmp_path):
    monkeypatch.setattr(ssh_env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(ssh_env.shutil, "which", lambda _name: "/usr/bin/ssh")
    for name, value in (("_establish_connection", None), ("_ensure_remote_dirs", None), ("init_session", None)):
        monkeypatch.setattr(ssh_env.SSHEnvironment, name, MagicMock(return_value=value))
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_detect_remote_home", MagicMock(return_value="/home/alice"))
    manager = MagicMock()
    monkeypatch.setattr(ssh_env, "FileSyncManager", MagicMock(return_value=manager))
    monkeypatch.setattr(ssh_env.subprocess, "run", MagicMock())
    env = ssh_env.SSHEnvironment(host="example.com", user="alice")
    yield env, manager
    env._sync_manager = None  # nothing left for a late __del__ to pull back


def test_ssh_cleanup_twice_syncs_back_once(_ssh_env):
    env, manager = _ssh_env
    env.cleanup()
    env.cleanup()  # e.g. __del__ after the idle reaper's teardown
    manager.sync_back.assert_called_once_with()
    assert env._sync_manager is None
