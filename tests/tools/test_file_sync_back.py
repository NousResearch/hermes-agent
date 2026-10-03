"""Tests for FileSyncManager.sync_back() — pull remote changes to host."""

import io
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.file_sync import (
    FileSyncManager,
    _cleanup_stale_sync_back_temp,
    _sha256_file,
    _SYNC_BACK_MAX_RETRIES,
    _SYNC_BACK_STALE_SECONDS,
    _SYNC_BACK_TEMP_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """PID of a child that has already been reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    proc.wait()
    return proc.pid

def _make_tar(files: dict[str, bytes], dest: Path):
    """Write a tar archive containing the given arcname->content pairs."""
    with tarfile.open(dest, "w") as tar:
        for arcname, content in files.items():
            info = tarfile.TarInfo(name=arcname)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def _make_download_fn(files: dict[str, bytes]):
    """Return a bulk_download_fn that writes a tar of the given files."""
    def download(dest: Path):
        _make_tar(files, dest)
    return download


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes (for test convenience)."""
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _write_file(path: Path, content: bytes) -> str:
    """Write bytes to *path*, creating parents, and return the string path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _make_manager(
    tmp_path: Path,
    file_mapping: list[tuple[str, str]] | None = None,
    bulk_download_fn=None,
    seed_pushed_state: bool = True,
) -> FileSyncManager:
    """Create a FileSyncManager wired for testing.

    *file_mapping* is a list of (host_path, remote_path) tuples that
    ``get_files_fn`` returns.  If *None* an empty list is used.

    When *seed_pushed_state* is True (default), populate ``_pushed_hashes``
    from the mapping so sync_back doesn't early-return on the "nothing
    previously pushed" guard. Set False to test the noop path.
    """
    mapping = file_mapping or []
    mgr = FileSyncManager(
        get_files_fn=lambda: mapping,
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=bulk_download_fn,
    )
    if seed_pushed_state:
        # Seed _pushed_hashes so sync_back's "nothing previously pushed"
        # guard does not early-return. Populate from the mapping when we
        # can; otherwise drop a sentinel entry.
        for host_path, remote_path in mapping:
            if os.path.exists(host_path):
                mgr._pushed_hashes[remote_path] = _sha256_file(host_path)
            else:
                mgr._pushed_hashes[remote_path] = "0" * 64
        if not mgr._pushed_hashes:
            mgr._pushed_hashes["/_sentinel"] = "0" * 64
    return mgr


class TestStaleSyncBackTempCleanup:
    """Sync-back temp entries leaked by a hard kill are reclaimed by the next sync-back (#110812)."""

    def test_removes_only_stale_prefixed_entries(self, tmp_path, monkeypatch):
        stale_tar = tmp_path / "hermes-sync-back-stale.tar"
        stale_dir = tmp_path / "hermes-sync-back-stale-staging"
        recent = tmp_path / "hermes-sync-back-recent.tar"
        unrelated = tmp_path / "other-process.tar"
        for path in (stale_tar, recent, unrelated):
            path.write_bytes(b"tar")
        stale_dir.mkdir()
        (stale_dir / "root").mkdir()
        now = 10_000.0
        for path in (stale_tar, stale_dir):
            os.utime(path, (now - _SYNC_BACK_STALE_SECONDS - 1,) * 2)
        os.utime(recent, (now - _SYNC_BACK_STALE_SECONDS + 1,) * 2)
        monkeypatch.setattr("tools.environments.file_sync.time.time", lambda: now)

        assert _cleanup_stale_sync_back_temp(tmp_path) == 2

        assert not stale_tar.exists()
        assert not stale_dir.exists()
        assert recent.exists()
        assert unrelated.exists()

    def test_sync_back_sweeps_leaked_entry_and_uses_identifiable_tar(self, tmp_path, monkeypatch):
        tmp_root = tmp_path / "tmproot"
        tmp_root.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_root))
        leaked = tmp_root / "hermes-sync-back-leaked.tar"
        leaked.write_bytes(b"x" * 1024)
        old = time.time() - _SYNC_BACK_STALE_SECONDS - 60
        os.utime(leaked, (old, old))

        seen = {}

        def download(dest: Path):
            seen["tar"] = dest
            _make_tar({"root/.hermes/x.txt": b"hi"}, dest)

        mgr = _make_manager(tmp_path, bulk_download_fn=download)
        mgr.sync_back()

        assert seen["tar"].name.startswith(f"{_SYNC_BACK_TEMP_PREFIX}{os.getpid()}-")
        assert not leaked.exists()
        assert list(tmp_root.iterdir()) == []

    def test_owner_liveness_decides_before_age(self, tmp_path):
        """A hard-killed owner's fresh entry is reclaimed at once; a live owner's fresh entry and
        a legacy (no-PID) fresh entry are kept; a live owner's entry past the cutoff still goes
        (a recycled PID must not pin a leak forever)."""
        dead_fresh = tmp_path / f"{_SYNC_BACK_TEMP_PREFIX}{_dead_pid()}-a.tar"
        live_fresh = tmp_path / f"{_SYNC_BACK_TEMP_PREFIX}{os.getpid()}-b"
        legacy_fresh = tmp_path / f"{_SYNC_BACK_TEMP_PREFIX}legacy.tar"
        live_old = tmp_path / f"{_SYNC_BACK_TEMP_PREFIX}{os.getpid()}-c.tar"
        for path in (dead_fresh, legacy_fresh, live_old):
            path.write_bytes(b"x")
        live_fresh.mkdir()
        old = time.time() - _SYNC_BACK_STALE_SECONDS - 60
        os.utime(live_old, (old, old))

        assert _cleanup_stale_sync_back_temp(tmp_path) == 2

        assert not dead_fresh.exists() and not live_old.exists()
        assert live_fresh.exists() and legacy_fresh.exists()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


        # Nothing to assert beyond "no exception raised"




class TestSyncBackAppliesChanged:
    """Remote file differs from pushed version -- gets copied to host."""

    def test_sync_back_applies_changed_file(self, tmp_path):
        host_file = tmp_path / "host" / "skill.py"
        original_content = b"print('v1')"
        _write_file(host_file, original_content)

        remote_path = "/root/.hermes/skill.py"
        mapping = [(str(host_file), remote_path)]

        remote_content = b"print('v2 - edited on remote')"
        download_fn = _make_download_fn({
            "root/.hermes/skill.py": remote_content,
        })

        mgr = _make_manager(tmp_path, file_mapping=mapping, bulk_download_fn=download_fn)
        mgr._pushed_hashes[remote_path] = _sha256_bytes(original_content)

        mgr.sync_back(hermes_home=tmp_path / ".hermes")

        assert host_file.read_bytes() == remote_content


class TestSyncBackNewRemoteFile:
    """File created on remote (not in _pushed_hashes) is applied via _infer_host_path."""

    def test_sync_back_detects_new_remote_file(self, tmp_path):
        # Existing mapping gives _infer_host_path a prefix to work with
        existing_host = tmp_path / "host" / "skills" / "existing.py"
        _write_file(existing_host, b"existing")
        mapping = [(str(existing_host), "/root/.hermes/skills/existing.py")]

        # Remote has a NEW file in the same directory that was never pushed
        new_remote_content = b"# brand new skill created on remote"
        download_fn = _make_download_fn({
            "root/.hermes/skills/new_skill.py": new_remote_content,
        })

        mgr = _make_manager(tmp_path, file_mapping=mapping, bulk_download_fn=download_fn)
        # No entry in _pushed_hashes for the new file

        mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # The new file should have been inferred and written to the host
        expected_host_path = tmp_path / "host" / "skills" / "new_skill.py"
        assert expected_host_path.exists()
        assert expected_host_path.read_bytes() == new_remote_content


class TestSyncBackConflict:
    """Host AND remote both changed since push -- warning logged, remote wins."""

    def test_sync_back_conflict_warns(self, tmp_path, caplog):
        host_file = tmp_path / "host" / "config.json"
        original_content = b'{"v": 1}'
        _write_file(host_file, original_content)

        remote_path = "/root/.hermes/config.json"
        mapping = [(str(host_file), remote_path)]

        # Host was modified after push
        host_file.write_bytes(b'{"v": 2, "host-edit": true}')

        # Remote was also modified
        remote_content = b'{"v": 3, "remote-edit": true}'
        download_fn = _make_download_fn({
            "root/.hermes/config.json": remote_content,
        })

        mgr = _make_manager(tmp_path, file_mapping=mapping, bulk_download_fn=download_fn)
        mgr._pushed_hashes[remote_path] = _sha256_bytes(original_content)

        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # Conflict warning was logged
        assert any("conflict" in r.message.lower() for r in caplog.records)

        # Remote version wins (last-write-wins)
        assert host_file.read_bytes() == remote_content


class TestSyncBackRetries:
    """Retry behaviour with exponential backoff."""

    @patch("tools.environments.file_sync._sleep")
    def test_sync_back_retries_on_failure(self, mock_sleep, tmp_path):
        call_count = 0

        def flaky_download(dest: Path):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"network error #{call_count}")
            # Third attempt succeeds -- write a valid (empty) tar
            _make_tar({}, dest)

        mgr = _make_manager(tmp_path, bulk_download_fn=flaky_download)
        mgr.sync_back(hermes_home=tmp_path / ".hermes")

        assert call_count == 3

    @patch("tools.environments.file_sync._sleep")
    def test_sync_back_all_retries_exhausted(self, mock_sleep, tmp_path, caplog):
        def always_fail(dest: Path):
            raise RuntimeError("persistent failure")

        mgr = _make_manager(tmp_path, bulk_download_fn=always_fail)

        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            # Should NOT raise -- failures are logged, not propagated
            mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # All retries were attempted
        assert mock_sleep.call_count == _SYNC_BACK_MAX_RETRIES - 1

        # Final "all attempts failed" warning was logged
        assert any("all" in r.message.lower() and "failed" in r.message.lower() for r in caplog.records)






class TestInferHostPath:
    """Edge cases for _infer_host_path prefix matching."""

    def test_infer_no_matching_prefix(self, tmp_path):
        """Remote path in unmapped directory should return None."""
        host_file = tmp_path / "host" / "skills" / "a.py"
        _write_file(host_file, b"content")
        mapping = [(str(host_file), "/root/.hermes/skills/a.py")]

        mgr = _make_manager(tmp_path, file_mapping=mapping)
        result = mgr._infer_host_path(
            "/root/.hermes/cache/new.json",
            file_mapping=mapping,
        )
        assert result is None




class TestSyncBackSIGINT:
    """SIGINT deferral during sync-back."""


    def test_sync_back_skips_signal_on_worker_thread(self, tmp_path):
        """From a non-main thread, signal.signal should NOT be called."""
        import threading

        download_fn = _make_download_fn({})
        mgr = _make_manager(tmp_path, bulk_download_fn=download_fn)

        signal_called = []

        def tracking_signal(*args):
            signal_called.append(args)

        with patch("tools.environments.file_sync.signal.signal", side_effect=tracking_signal):
            # Run from a worker thread
            exc = []
            def run():
                try:
                    mgr.sync_back(hermes_home=tmp_path / ".hermes")
                except Exception as e:
                    exc.append(e)

            t = threading.Thread(target=run)
            t.start()
            t.join(timeout=10)

        assert not exc, f"sync_back raised: {exc}"
        # signal.signal should NOT have been called from the worker thread
        assert len(signal_called) == 0


class TestSyncBackSizeCap:
    """The size cap refuses to extract tars above the configured limit."""

    def test_sync_back_refuses_oversized_tar(self, tmp_path, caplog):
        """A tar larger than terminal.sync_back_max_bytes should be skipped with a warning."""
        # Build a download_fn that writes a small tar, but lower the configured cap
        # so the test doesn't need to produce a 2 GiB file.
        skill_host = _write_file(tmp_path / "host_skill.md", b"original")
        files = {"root/.hermes/skill.md": b"remote_version"}
        download_fn = _make_download_fn(files)

        mgr = _make_manager(
            tmp_path,
            file_mapping=[(skill_host, "/root/.hermes/skill.md")],
            bulk_download_fn=download_fn,
        )

        # Cap at 1 byte so any non-empty tar exceeds it
        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            with patch("hermes_cli.config.load_config", return_value={"terminal": {"sync_back_max_bytes": 1}}):
                mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # Host file should be untouched because extraction was skipped
        assert Path(skill_host).read_bytes() == b"original"
        # Warning should mention the cap
        assert any("cap" in r.message for r in caplog.records)


    def test_cap_override_config_key_raises_the_cap(self, tmp_path, monkeypatch, caplog):
        """config.yaml ``terminal.sync_back_max_bytes`` overrides the 2 GiB default; a
        non-integer value is ignored with a warning and the default applies. The env var
        the first cut used is gone — non-secret settings live in config.yaml."""
        host_file = _write_file(tmp_path / "host_skill.md", b"original")
        files = {"root/.hermes/skill.md": b"remote_version"}
        mgr = _make_manager(tmp_path, file_mapping=[(host_file, "/root/.hermes/skill.md")],
                            bulk_download_fn=_make_download_fn(files))

        monkeypatch.setenv("HERMES_SYNC_BACK_MAX_BYTES", "1")  # the first cut's env var: must be ignored
        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: {"terminal": {"sync_back_max_bytes": 1}})
        mgr.sync_back(hermes_home=tmp_path / ".hermes")
        assert Path(host_file).read_bytes() == b"original"  # 1-byte cap: skipped

        monkeypatch.setattr("hermes_cli.config.load_config",
                            lambda: {"terminal": {"sync_back_max_bytes": "lots"}})
        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            mgr.sync_back(hermes_home=tmp_path / ".hermes")
        assert Path(host_file).read_bytes() == b"remote_version"  # default cap applies
        assert any("sync_back_max_bytes" in r.message for r in caplog.records)


class TestSyncBackWindowsHost:
    """#76267: sync_back on a Windows host. The staging tar must be reopenable for writing by
    the backend (NamedTemporaryFile held an exclusive handle → PermissionError), and remote
    keys/parents must stay POSIX (relpath/Path stringify with backslashes on Windows, so no
    staged file ever matched its mapping and every edit was dropped as "no host mapping")."""

    def test_backend_can_reopen_the_tar_path_for_writing(self, tmp_path):
        """Contract on every host: the download callback receives a path nothing else holds open."""
        seen = {}

        def download(dest: Path) -> None:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo(name="root/.hermes/skill.py")
                info.size = 2
                tar.addfile(info, io.BytesIO(b"v2"))
            with open(dest, "wb") as fh:  # the SSH/Modal backends write exactly like this
                fh.write(buf.getvalue())
            seen["dest"] = dest

        host_file = tmp_path / "host" / "skill.py"
        _write_file(host_file, b"v1")
        mgr = _make_manager(tmp_path, [(str(host_file), "/root/.hermes/skill.py")], bulk_download_fn=download)
        mgr._pushed_hashes["/root/.hermes/skill.py"] = _sha256_bytes(b"v1")
        mgr.sync_back(hermes_home=tmp_path / ".hermes")
        assert host_file.read_bytes() == b"v2"
        assert not seen["dest"].exists()  # staging tar removed after use

    @pytest.mark.platforms("windows")
    def test_posix_remote_keys_match_on_windows(self, tmp_path):
        host_file = tmp_path / "host" / "skill.py"
        _write_file(host_file, b"v1")
        mapping = [(str(host_file), "/root/.hermes/skills/a/skill.py")]
        mgr = _make_manager(tmp_path, mapping, bulk_download_fn=_make_download_fn({
            "root/.hermes/skills/a/skill.py": b"v2", "root/.hermes/skills/a/new.md": b"new"}))
        mgr._pushed_hashes["/root/.hermes/skills/a/skill.py"] = _sha256_bytes(b"v1")
        mgr.sync_back(hermes_home=tmp_path / ".hermes")
        assert host_file.read_bytes() == b"v2"  # relpath key was 'root\\.hermes\\...' → skipped
        assert (tmp_path / "host" / "new.md").read_bytes() == b"new"  # _infer_host_path parent match


class TestSyncBackRefreshableCredentials:
    """#128233: a credential file DECLARED refreshable (rotating token store, e.g.
    ``google_token.json`` refreshed inside an SSH sandbox) writes back on sync_back.
    Undeclared credentials stay upload-only; targets outside the ACTIVE profile home are
    refused so one sandbox can never apply another profile's token."""

    def test_refreshable_credential_written_back(self, tmp_path, monkeypatch, caplog):
        """The reported case: token refreshed in-sandbox, SHA differs from push, applied."""
        hermes_home = tmp_path / "home" / ".hermes"
        host_token = hermes_home / "google_token.json"
        _write_file(host_token, b'{"token": "stale"}')

        remote_path = "/root/.hermes/google_token.json"
        monkeypatch.setattr("tools.environments.file_sync._credential_host_paths", lambda: {str(host_token)})
        monkeypatch.setattr("tools.environments.file_sync._refreshable_credential_declarations",
                            lambda: {"google_token.json": str(host_token)})
        monkeypatch.setattr("tools.environments.file_sync.get_hermes_home", lambda: hermes_home)

        refreshed = b'{"token": "fresh"}'
        mgr = _make_manager(tmp_path, file_mapping=[(str(host_token), remote_path)],
                            bulk_download_fn=_make_download_fn({"root/.hermes/google_token.json": refreshed}))
        mgr._pushed_hashes[remote_path] = _sha256_bytes(b'{"token": "stale"}')

        with caplog.at_level(logging.INFO, logger="tools.environments.file_sync"):
            mgr.sync_back(hermes_home=tmp_path / "hermes")

        assert host_token.read_bytes() == refreshed
        assert any("refreshable credential" in r.message for r in caplog.records)

    def test_undeclared_credential_stays_upload_only(self, tmp_path, monkeypatch):
        """No declaration: the pre-#128233 behavior, skip and leave the host copy alone."""
        host_token = tmp_path / "home" / ".hermes" / "google_token.json"
        _write_file(host_token, b'{"token": "stale"}')
        remote_path = "/root/.hermes/google_token.json"

        monkeypatch.setattr("tools.environments.file_sync._credential_host_paths", lambda: {str(host_token)})
        monkeypatch.setattr("tools.environments.file_sync._refreshable_credential_declarations", lambda: {})

        mgr = _make_manager(tmp_path, file_mapping=[(str(host_token), remote_path)],
                            bulk_download_fn=_make_download_fn({"root/.hermes/google_token.json": b'{"token": "fresh"}'}))
        mgr._pushed_hashes[remote_path] = _sha256_bytes(b'{"token": "stale"}')

        mgr.sync_back(hermes_home=tmp_path / "hermes")

        assert host_token.read_bytes() == b'{"token": "stale"}'

    def test_refreshable_outside_active_home_refused(self, tmp_path, monkeypatch):
        """Fail-closed boundary: a declaration resolving outside the ACTIVE profile home
        (a multiplexed gateway running another profile's sync) must not write back."""
        other_home_token = tmp_path / "other-home" / ".hermes" / "google_token.json"
        _write_file(other_home_token, b'{"token": "stale"}')
        active_home = tmp_path / "active" / ".hermes"
        remote_path = "/root/.hermes/google_token.json"

        monkeypatch.setattr("tools.environments.file_sync._credential_host_paths", lambda: {str(other_home_token)})
        monkeypatch.setattr("tools.environments.file_sync._refreshable_credential_declarations",
                            lambda: {"google_token.json": str(other_home_token)})
        monkeypatch.setattr("tools.environments.file_sync.get_hermes_home", lambda: active_home)

        mgr = _make_manager(tmp_path, file_mapping=[(str(other_home_token), remote_path)],
                            bulk_download_fn=_make_download_fn({"root/.hermes/google_token.json": b'{"token": "fresh"}'}))
        mgr._pushed_hashes[remote_path] = _sha256_bytes(b'{"token": "stale"}')

        mgr.sync_back(hermes_home=tmp_path / "hermes")

        assert other_home_token.read_bytes() == b'{"token": "stale"}'  # untouched

    def test_credential_created_in_sandbox_writes_back(self, tmp_path, monkeypatch):
        """No host file at push time (OAuth consent completed inside the sandbox): the
        declared rel path still pins a remapped SSH remote base to its profile-home target."""
        hermes_home = tmp_path / "home" / ".hermes"
        existing_skill = hermes_home / "skills" / "a.md"
        _write_file(existing_skill, b"skill")
        remapped_base = "/home" + "/" + "remote-user"  # built at runtime, not a literal
        mapping = [(str(existing_skill), remapped_base + "/.hermes/skills/a.md")]

        monkeypatch.setattr("tools.environments.file_sync._credential_host_paths", lambda: set())
        monkeypatch.setattr("tools.environments.file_sync._refreshable_credential_declarations",
                            lambda: {"google_token.json": str(hermes_home / "google_token.json")})
        monkeypatch.setattr("tools.environments.file_sync.get_hermes_home", lambda: hermes_home)

        new_token = b'{"first": "consent-inside-sandbox"}'
        tar_key = remapped_base.lstrip("/") + "/.hermes/google_token.json"
        mgr = _make_manager(tmp_path, file_mapping=mapping,
                            bulk_download_fn=_make_download_fn({tar_key: new_token}))
        mgr._pushed_hashes[mapping[0][1]] = _sha256_bytes(b"skill")

        mgr.sync_back(hermes_home=tmp_path / "hermes")

        assert (hermes_home / "google_token.json").read_bytes() == new_token

    def test_undeclared_rel_created_in_sandbox_not_inferred(self, tmp_path, monkeypatch):
        """A sandbox-created credential whose rel was never declared refreshable must NOT
        be written back via the declared-rel path (nor leak through _infer_host_path)."""
        hermes_home = tmp_path / "home" / ".hermes"
        existing_skill = hermes_home / "skills" / "a.md"
        _write_file(existing_skill, b"skill")
        mapping = [(str(existing_skill), "/root/.hermes/skills/a.md")]

        monkeypatch.setattr("tools.environments.file_sync._credential_host_paths", lambda: set())
        monkeypatch.setattr("tools.environments.file_sync._refreshable_credential_declarations",
                            lambda: {"google_token.json": str(hermes_home / "google_token.json")})
        monkeypatch.setattr("tools.environments.file_sync.get_hermes_home", lambda: hermes_home)

        mgr = _make_manager(tmp_path, file_mapping=mapping, bulk_download_fn=_make_download_fn(
            {"root/.hermes/other_token.json": b'{"undeclared": true}'}))
        mgr._pushed_hashes["/root/.hermes/skills/a.md"] = _sha256_bytes(b"skill")

        mgr.sync_back(hermes_home=tmp_path / "hermes")

        assert not (hermes_home / "other_token.json").exists()

    def test_refreshable_conflict_keeps_host_version(self, tmp_path, monkeypatch, caplog):
        """A host token rotated AFTER push (a second client rotating the same token) must
        survive: the sandbox copy is stale by definition, and last-write-wins would
        silently destroy the newer host credential (#128233)."""
        hermes_home = tmp_path / "home" / ".hermes"
        host_token = hermes_home / "google_token.json"
        _write_file(host_token, b'{"rt": "pushed-old"}')

        remote_path = "/root/.hermes/google_token.json"
        monkeypatch.setattr("tools.environments.file_sync._credential_host_paths", lambda: {str(host_token)})
        monkeypatch.setattr("tools.environments.file_sync._refreshable_credential_declarations",
                            lambda: {"google_token.json": str(host_token)})
        monkeypatch.setattr("tools.environments.file_sync.get_hermes_home", lambda: hermes_home)

        host_token.write_bytes(b'{"rt": "host-refreshed"}')  # rotated on the host after push
        mgr = _make_manager(tmp_path, file_mapping=[(str(host_token), remote_path)],
                            bulk_download_fn=_make_download_fn(
                                {"root/.hermes/google_token.json": b'{"rt": "sandbox-version"}'}))
        mgr._pushed_hashes[remote_path] = _sha256_bytes(b'{"rt": "pushed-old"}')

        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            mgr.sync_back(hermes_home=tmp_path / "hermes")

        assert host_token.read_bytes() == b'{"rt": "host-refreshed"}'  # host version kept
        assert any("refusing write-back" in r.message for r in caplog.records)

    def test_refreshable_write_back_never_truncates_host_file(self, tmp_path, monkeypatch):
        """A failed copy must leave the host token COMPLETE (old version), never 0 bytes:
        the install goes through a same-dir temp + os.replace, so the target is always
        a whole file."""
        hermes_home = tmp_path / "home" / ".hermes"
        host_token = hermes_home / "google_token.json"
        _write_file(host_token, b'{"token": "stale"}')

        remote_path = "/root/.hermes/google_token.json"
        monkeypatch.setattr("tools.environments.file_sync._credential_host_paths", lambda: {str(host_token)})
        monkeypatch.setattr("tools.environments.file_sync._refreshable_credential_declarations",
                            lambda: {"google_token.json": str(host_token)})
        monkeypatch.setattr("tools.environments.file_sync.get_hermes_home", lambda: hermes_home)
        monkeypatch.setattr("tools.environments.file_sync._sleep", lambda s: None)

        def _failing_copy(src, dst, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("tools.environments.file_sync.shutil.copy2", _failing_copy)

        mgr = _make_manager(tmp_path, file_mapping=[(str(host_token), remote_path)],
                            bulk_download_fn=_make_download_fn(
                                {"root/.hermes/google_token.json": b'{"token": "fresh"}'}))
        mgr._pushed_hashes[remote_path] = _sha256_bytes(b'{"token": "stale"}')

        mgr.sync_back(hermes_home=tmp_path / "hermes")

        assert host_token.read_bytes() == b'{"token": "stale"}'  # intact, not truncated
        assert not (hermes_home / "google_token.json.hermes-sync-back.tmp").exists()
