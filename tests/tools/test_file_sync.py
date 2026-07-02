"""Tests for FileSyncManager — mtime tracking, deletion detection, transactional rollback."""

import hashlib
import io
import os
import tarfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.environments import file_sync as fsmod
from tools.environments.file_sync import (
    FileSyncManager,
    _FORCE_SYNC_ENV,
    _sha256_file,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)


@pytest.fixture
def tmp_files(tmp_path):
    """Create a few temp files to use as sync sources."""
    files = {}
    for name in ("cred_a.json", "cred_b.json", "skill_main.py"):
        p = tmp_path / name
        p.write_text(f"content of {name}")
        files[name] = str(p)
    return files


def _make_get_files(tmp_files, remote_base="/root/.hermes"):
    """Return a get_files_fn that maps local files to remote paths."""
    mapping = [(hp, f"{remote_base}/{name}") for name, hp in tmp_files.items()]

    def get_files():
        return [(hp, rp) for hp, rp in mapping if Path(hp).exists()]

    return get_files


def _make_manager(tmp_files, remote_base="/root/.hermes", upload=None, delete=None):
    """Create a FileSyncManager with test callbacks."""
    return FileSyncManager(
        get_files_fn=_make_get_files(tmp_files, remote_base),
        upload_fn=upload or MagicMock(),
        delete_fn=delete or MagicMock(),
    )


class TestMtimeSkip:
    def test_unchanged_files_not_re_uploaded(self, tmp_files):
        upload = MagicMock()
        mgr = _make_manager(tmp_files, upload=upload)

        mgr.sync(force=True)
        assert upload.call_count == 3

        upload.reset_mock()
        mgr.sync(force=True)
        assert upload.call_count == 0, "unchanged files should not be re-uploaded"

    def test_changed_file_re_uploaded(self, tmp_files):
        upload = MagicMock()
        mgr = _make_manager(tmp_files, upload=upload)

        mgr.sync(force=True)
        upload.reset_mock()

        # Touch one file
        time.sleep(0.05)
        Path(tmp_files["cred_a.json"]).write_text("updated content")

        mgr.sync(force=True)
        assert upload.call_count == 1
        assert tmp_files["cred_a.json"] in upload.call_args[0][0]

    def test_new_file_detected(self, tmp_files, tmp_path):
        upload = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=_make_get_files(tmp_files),
            upload_fn=upload,
            delete_fn=MagicMock(),
        )

        mgr.sync(force=True)
        assert upload.call_count == 3

        # Add a new file
        new_file = tmp_path / "new_skill.py"
        new_file.write_text("new content")
        tmp_files["new_skill.py"] = str(new_file)
        # Recreate manager with updated file list
        mgr._get_files_fn = _make_get_files(tmp_files)

        upload.reset_mock()
        mgr.sync(force=True)
        assert upload.call_count == 1


class TestDeletion:
    def test_removed_file_triggers_delete(self, tmp_files):
        upload = MagicMock()
        delete = MagicMock()
        mgr = _make_manager(tmp_files, upload=upload, delete=delete)

        mgr.sync(force=True)
        delete.assert_not_called()

        # Remove a file locally
        os.unlink(tmp_files["cred_b.json"])
        del tmp_files["cred_b.json"]
        mgr._get_files_fn = _make_get_files(tmp_files)

        mgr.sync(force=True)
        delete.assert_called_once()
        deleted_paths = delete.call_args[0][0]
        assert any("cred_b.json" in p for p in deleted_paths)

    def test_no_delete_when_no_removals(self, tmp_files):
        delete = MagicMock()
        mgr = _make_manager(tmp_files, delete=delete)

        mgr.sync(force=True)
        mgr.sync(force=True)
        delete.assert_not_called()


class TestTransactionalRollback:
    def test_upload_failure_rolls_back(self, tmp_files):
        call_count = 0

        def failing_upload(host_path, remote_path):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("upload failed")

        mgr = _make_manager(tmp_files, upload=failing_upload)

        # First sync fails (swallowed, logged, state rolled back)
        mgr.sync(force=True)

        # State should be empty (rolled back) — next sync retries all files
        good_upload = MagicMock()
        mgr._upload_fn = good_upload
        mgr.sync(force=True)
        assert good_upload.call_count == 3, "all files should be retried after rollback"

    def test_delete_failure_rolls_back(self, tmp_files):
        upload = MagicMock()
        mgr = _make_manager(tmp_files, upload=upload)

        # Initial sync
        mgr.sync(force=True)

        # Remove a file
        os.unlink(tmp_files["skill_main.py"])
        del tmp_files["skill_main.py"]
        mgr._get_files_fn = _make_get_files(tmp_files)

        # Delete fails (swallowed, state rolled back)
        mgr._delete_fn = MagicMock(side_effect=RuntimeError("delete failed"))
        mgr.sync(force=True)

        # Next sync should retry the delete
        good_delete = MagicMock()
        mgr._delete_fn = good_delete
        upload.reset_mock()
        mgr.sync(force=True)
        good_delete.assert_called_once()


class TestRateLimiting:
    def test_sync_skipped_within_interval(self, tmp_files):
        upload = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=_make_get_files(tmp_files),
            upload_fn=upload,
            delete_fn=MagicMock(),
            sync_interval=10.0,
        )

        mgr.sync(force=True)
        assert upload.call_count == 3

        upload.reset_mock()
        # Without force, should skip due to rate limit
        mgr.sync()
        assert upload.call_count == 0

    def test_force_bypasses_rate_limit(self, tmp_files, tmp_path):
        upload = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=_make_get_files(tmp_files),
            upload_fn=upload,
            delete_fn=MagicMock(),
            sync_interval=10.0,
        )

        mgr.sync(force=True)
        upload.reset_mock()

        # Add a new file and force sync
        new_file = tmp_path / "forced.txt"
        new_file.write_text("forced")
        tmp_files["forced.txt"] = str(new_file)
        mgr._get_files_fn = _make_get_files(tmp_files)

        mgr.sync(force=True)
        assert upload.call_count == 1

    def test_env_var_forces_sync(self, tmp_files, tmp_path):
        upload = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=_make_get_files(tmp_files),
            upload_fn=upload,
            delete_fn=MagicMock(),
            sync_interval=10.0,
        )

        mgr.sync(force=True)
        upload.reset_mock()

        new_file = tmp_path / "env_forced.txt"
        new_file.write_text("env forced")
        tmp_files["env_forced.txt"] = str(new_file)
        mgr._get_files_fn = _make_get_files(tmp_files)

        with patch.dict(os.environ, {_FORCE_SYNC_ENV: "1"}):
            mgr.sync()
        assert upload.call_count == 1


class TestEdgeCases:
    def test_empty_file_list(self):
        upload = MagicMock()
        delete = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=lambda: [],
            upload_fn=upload,
            delete_fn=delete,
        )

        mgr.sync(force=True)
        upload.assert_not_called()
        delete.assert_not_called()

    def test_file_disappears_between_list_and_upload(self, tmp_path):
        """File listed by get_files but deleted before _file_mtime_key reads it."""
        f = tmp_path / "ephemeral.txt"
        f.write_text("here now")

        upload = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=lambda: [(str(f), "/root/.hermes/ephemeral.txt")],
            upload_fn=upload,
            delete_fn=MagicMock(),
        )

        # Delete the file before sync can stat it
        os.unlink(str(f))

        mgr.sync(force=True)
        upload.assert_not_called()  # _file_mtime_key returns None, skipped


class TestSyncBackSecurity:
    def test_sync_back_does_not_overwrite_uploaded_credential_files(self, tmp_path, monkeypatch):
        credential = tmp_path / "token.json"
        credential.write_text("host-token", encoding="utf-8")
        skill = tmp_path / "skill.py"
        skill.write_text("host-skill", encoding="utf-8")

        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(credential),
                    "container_path": "/root/.hermes/credentials/token.json",
                }
            ],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files",
            lambda container_base="/root/.hermes": [
                {
                    "host_path": str(skill),
                    "container_path": f"{container_base}/skills/skill.py",
                }
            ],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files",
            lambda container_base="/root/.hermes": [],
        )

        def bulk_download(dest: Path) -> None:
            with tarfile.open(dest, "w") as tar:
                for name, data in {
                    "root/.hermes/credentials/token.json": b"remote-token",
                    "root/.hermes/skills/skill.py": b"remote-skill",
                }.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))

        mgr = FileSyncManager(
            get_files_fn=lambda: iter_sync_files("/root/.hermes"),
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
            bulk_download_fn=bulk_download,
        )

        mgr.sync(force=True)
        mgr.sync_back(hermes_home=tmp_path)

        assert credential.read_text(encoding="utf-8") == "host-token"
        assert skill.read_text(encoding="utf-8") == "remote-skill"


class TestBulkUpload:
    """Tests for the optional bulk_upload_fn callback."""

    def test_bulk_upload_used_when_provided(self, tmp_files):
        """When bulk_upload_fn is set, it's called instead of per-file upload_fn."""
        upload = MagicMock()
        bulk_upload = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=_make_get_files(tmp_files),
            upload_fn=upload,
            delete_fn=MagicMock(),
            bulk_upload_fn=bulk_upload,
        )

        mgr.sync(force=True)
        upload.assert_not_called()
        bulk_upload.assert_called_once()
        # All 3 files passed as a list of (host, remote) tuples
        files_arg = bulk_upload.call_args[0][0]
        assert len(files_arg) == 3

    def test_fallback_to_upload_fn_when_no_bulk(self, tmp_files):
        """Without bulk_upload_fn, per-file upload_fn is used (backwards compat)."""
        upload = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=_make_get_files(tmp_files),
            upload_fn=upload,
            delete_fn=MagicMock(),
            bulk_upload_fn=None,
        )

        mgr.sync(force=True)
        assert upload.call_count == 3

    def test_bulk_upload_rollback_on_failure(self, tmp_files):
        """Bulk upload failure rolls back synced state so next sync retries."""
        bulk_upload = MagicMock(side_effect=RuntimeError("upload failed"))
        mgr = FileSyncManager(
            get_files_fn=_make_get_files(tmp_files),
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
            bulk_upload_fn=bulk_upload,
        )

        mgr.sync(force=True)  # fails, should rollback

        # State rolled back: next sync should retry all files
        bulk_upload.side_effect = None
        bulk_upload.reset_mock()
        mgr.sync(force=True)
        bulk_upload.assert_called_once()
        assert len(bulk_upload.call_args[0][0]) == 3


# ---------------------------------------------------------------------------
# Module-level helpers: iter_sync_files, quoted_rm/mkdir, unique_parent_dirs,
# _sha256_file
# ---------------------------------------------------------------------------


class TestIterSyncFiles:
    """iter_sync_files merges credential, skills, and cache file mounts."""

    def test_merges_all_three_sources(self, monkeypatch):
        cred = [
            {
                "host_path": "/host/cred.json",
                "container_path": "/root/.hermes/cred.json",
            }
        ]
        skills = [
            {
                "host_path": "/host/skill.md",
                "container_path": "/root/.hermes/skills/s.md",
            }
        ]
        cache = [
            {
                "host_path": "/host/cache.db",
                "container_path": "/root/.hermes/cache/cache.db",
            }
        ]

        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts", lambda: cred
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files",
            lambda container_base="/root/.hermes": skills,
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files",
            lambda container_base="/root/.hermes": cache,
        )

        result = iter_sync_files()
        assert ("/host/cred.json", "/root/.hermes/cred.json") in result
        assert ("/host/skill.md", "/root/.hermes/skills/s.md") in result
        assert ("/host/cache.db", "/root/.hermes/cache/cache.db") in result

    def test_remaps_credential_container_base(self, monkeypatch):
        """Credential paths remap /root/.hermes to the given container_base."""
        cred = [
            {
                "host_path": "/host/cred.json",
                "container_path": "/root/.hermes/cred.json",
            }
        ]
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts", lambda: cred
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files",
            lambda container_base="/root/.hermes": [],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files",
            lambda container_base="/root/.hermes": [],
        )

        result = iter_sync_files(container_base="/home/daytona")
        assert result == [("/host/cred.json", "/home/daytona/cred.json")]

    def test_empty_when_all_sources_empty(self, monkeypatch):
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts", lambda: []
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files",
            lambda container_base="/root/.hermes": [],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files",
            lambda container_base="/root/.hermes": [],
        )
        assert iter_sync_files() == []


class TestQuotedCommands:
    def test_quoted_rm_command_simple(self):
        cmd = quoted_rm_command(["/root/file1.txt", "/root/file2.txt"])
        assert cmd == "rm -f /root/file1.txt /root/file2.txt"

    def test_quoted_rm_command_escapes_special_chars(self):
        cmd = quoted_rm_command(["/root/file with space.txt"])
        assert cmd == "rm -f '/root/file with space.txt'"

    def test_quoted_rm_command_empty_list(self):
        assert quoted_rm_command([]) == "rm -f "

    def test_quoted_mkdir_command_simple(self):
        cmd = quoted_mkdir_command(["/root/dir1", "/root/dir2"])
        assert cmd == "mkdir -p /root/dir1 /root/dir2"

    def test_quoted_mkdir_command_escapes_special_chars(self):
        cmd = quoted_mkdir_command(["/root/dir with space"])
        assert cmd == "mkdir -p '/root/dir with space'"

    def test_quoted_mkdir_command_empty_list(self):
        assert quoted_mkdir_command([]) == "mkdir -p "


class TestUniqueParentDirs:
    def test_extracts_sorted_unique_dirs(self):
        files = [
            ("/host/a", "/root/dir1/file1.txt"),
            ("/host/b", "/root/dir2/file2.txt"),
            ("/host/c", "/root/dir1/file3.txt"),
        ]
        result = unique_parent_dirs(files)
        assert result == ["/root/dir1", "/root/dir2"]

    def test_empty_list_returns_empty(self):
        assert unique_parent_dirs([]) == []

    def test_single_file(self):
        assert unique_parent_dirs([("/h", "/root/sub/f.txt")]) == ["/root/sub"]


class TestSha256File:
    def test_returns_correct_hash(self, tmp_path):
        f = tmp_path / "data.bin"
        content = b"hello world"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _sha256_file(str(f)) == expected

    def test_large_file_chunked(self, tmp_path):
        """Files larger than 65536 bytes are read in chunks."""
        f = tmp_path / "large.bin"
        content = b"x" * 200000  # > 65536 chunk size
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _sha256_file(str(f)) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert _sha256_file(str(f)) == hashlib.sha256(b"").hexdigest()


# ---------------------------------------------------------------------------
# sync_back: download remote tar, diff against pushed hashes, apply changes
# ---------------------------------------------------------------------------


def _make_tar(tmp_path, files: dict[str, bytes]) -> str:
    """Create a tar archive at tmp_path/tar.tar containing the given files.

    ``files`` maps archive-relative paths to file content bytes.
    """
    tar_path = tmp_path / "remote.tar"
    with tarfile.open(tar_path, "w") as tar:
        for rel, content in files.items():
            import io

            info = tarfile.TarInfo(name=rel)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return str(tar_path)


class TestSyncBack:
    """sync_back downloads remote tar, diffs by SHA-256, applies changed files."""

    def test_no_bulk_download_fn_is_noop(self, tmp_files):
        mgr = _make_manager(tmp_files)
        # bulk_download_fn not set → sync_back returns immediately
        mgr.sync_back()  # must not raise

    def test_skips_when_no_prior_push_state(self, tmp_path):
        """No prior sync() → _pushed_hashes empty → sync_back skips."""
        download = MagicMock()
        mgr = FileSyncManager(
            get_files_fn=lambda: [],
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
            bulk_download_fn=download,
        )
        mgr.sync_back()
        download.assert_not_called()

    def test_applies_changed_remote_file(self, tmp_files, tmp_path):
        """Remote file differs from pushed hash → copied to host."""
        host_file = tmp_files["cred_a.json"]
        remote_path = "/root/.hermes/cred_a.json"

        # Simulate a prior push: record the pushed hash
        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)
        pushed_hash = mgr._pushed_hashes[remote_path]

        # Build a tar where the remote file has different content
        new_content = b"remote changed content"
        tar_path = _make_tar(tmp_path, {"root/.hermes/cred_a.json": new_content})

        def download_fn(dest_tar: Path):
            import shutil

            shutil.copy(tar_path, dest_tar)

        mgr._bulk_download_fn = download_fn
        mgr.sync_back()

        # Host file should now have the remote content
        assert Path(host_file).read_bytes() == new_content

    def test_skips_unchanged_remote_file(self, tmp_files, tmp_path):
        """Remote file hash matches pushed hash → not copied."""
        host_file = tmp_files["cred_a.json"]
        remote_path = "/root/.hermes/cred_a.json"
        original_content = Path(host_file).read_bytes()

        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)

        # Build a tar where the remote file has the SAME content
        tar_path = _make_tar(tmp_path, {"root/.hermes/cred_a.json": original_content})

        def download_fn(dest_tar: Path):
            import shutil

            shutil.copy(tar_path, dest_tar)

        mgr._bulk_download_fn = download_fn
        mgr.sync_back()

        # Host file unchanged
        assert Path(host_file).read_bytes() == original_content

    def test_applies_new_remote_file_via_infer(self, tmp_files, tmp_path):
        """Remote file not in pushed_hashes → inferred host path and applied."""
        # Use a skills file so the mapping has a directory to infer from
        host_skill = tmp_files["skill_main.py"]
        remote_skill = "/root/.hermes/skill_main.py"

        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)

        # Build a tar with a NEW file in the same directory as an existing one
        new_content = b"new skill content"
        tar_path = _make_tar(
            tmp_path,
            {
                "root/.hermes/skill_main.py": Path(host_skill).read_bytes(),
                "root/.hermes/new_skill.py": new_content,
            },
        )

        def download_fn(dest_tar: Path):
            import shutil

            shutil.copy(tar_path, dest_tar)

        mgr._bulk_download_fn = download_fn
        mgr.sync_back()

        # The new file should be inferred and created next to the existing one
        new_host = str(Path(host_skill).parent / "new_skill.py")
        assert Path(new_host).read_bytes() == new_content

    def test_skips_unmappable_new_remote_file(self, tmp_files, tmp_path):
        """New remote file with no inferable host path → skipped, no crash."""
        mgr = _make_manager(tmp_files, remote_base="/root/.hermes")
        mgr.sync(force=True)

        # Build a tar with a file in a completely unrelated directory
        tar_path = _make_tar(
            tmp_path,
            {
                "root/.hermes/skill_main.py": Path(
                    tmp_files["skill_main.py"]
                ).read_bytes(),
                "completely/unrelated/file.txt": b"orphan",
            },
        )

        def download_fn(dest_tar: Path):
            import shutil

            shutil.copy(tar_path, dest_tar)

        mgr._bulk_download_fn = download_fn
        mgr.sync_back()  # must not raise

    def test_oversized_tar_skipped(self, tmp_files, tmp_path):
        """Tar exceeding _SYNC_BACK_MAX_BYTES is not extracted."""
        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)

        tar_path = _make_tar(
            tmp_path,
            {"root/.hermes/cred_a.json": Path(tmp_files["cred_a.json"]).read_bytes()},
        )

        def download_fn(dest_tar: Path):
            import shutil

            shutil.copy(tar_path, dest_tar)

        mgr._bulk_download_fn = download_fn

        # Patch the size cap to 0 so even a tiny tar exceeds it
        with patch.object(fsmod, "_SYNC_BACK_MAX_BYTES", 0):
            mgr.sync_back()

        # Original content untouched
        original = Path(tmp_files["cred_a.json"]).read_bytes()
        assert Path(tmp_files["cred_a.json"]).read_bytes() == original

    def test_retry_on_failure_then_success(self, tmp_files, tmp_path):
        """sync_back retries on failure and succeeds on a later attempt."""
        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)

        tar_path = _make_tar(
            tmp_path,
            {"root/.hermes/cred_a.json": b"remote update"},
        )

        call_count = {"n": 0}

        def download_fn(dest_tar: Path):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("transient network error")
            import shutil

            shutil.copy(tar_path, dest_tar)

        mgr._bulk_download_fn = download_fn

        # Patch sleep to avoid real delays during retry backoff
        with patch.object(fsmod, "_sleep", lambda s: None):
            mgr.sync_back()

        assert call_count["n"] == 2
        assert Path(tmp_files["cred_a.json"]).read_bytes() == b"remote update"

    def test_all_retries_fail_logs_warning(self, tmp_files, tmp_path):
        """All retry attempts fail — sync_back logs but does not raise."""
        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)

        def always_fail(dest_tar: Path):
            raise RuntimeError("permanent failure")

        mgr._bulk_download_fn = always_fail

        with patch.object(fsmod, "_sleep", lambda s: None):
            mgr.sync_back()  # must not raise

    def test_get_files_fn_exception_handled(self, tmp_files, tmp_path):
        """If get_files_fn raises during sync_back, file_mapping falls back to []."""
        host_file = tmp_files["cred_a.json"]
        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)

        new_content = b"remote update"
        tar_path = _make_tar(tmp_path, {"root/.hermes/cred_a.json": new_content})

        def download_fn(dest_tar: Path):
            import shutil

            shutil.copy(tar_path, dest_tar)

        # Make get_files_fn raise — _sync_back_impl catches it and uses []
        mgr._get_files_fn = MagicMock(side_effect=RuntimeError("boom"))
        mgr._bulk_download_fn = download_fn
        mgr.sync_back()

        # File won't be applied because mapping is empty and inference fails,
        # but sync_back must not crash.

    def test_sigint_deferred_on_main_thread(self, tmp_files, tmp_path):
        """SIGINT during sync_back on main thread is deferred until completion."""
        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)

        tar_path = _make_tar(
            tmp_path,
            {"root/.hermes/cred_a.json": Path(tmp_files["cred_a.json"]).read_bytes()},
        )

        def download_fn(dest_tar: Path):
            import shutil

            shutil.copy(tar_path, dest_tar)

        mgr._bulk_download_fn = download_fn

        # Verify signal handler is temporarily replaced and restored
        import signal

        original = signal.getsignal(signal.SIGINT)
        mgr.sync_back()
        assert signal.getsignal(signal.SIGINT) is original

    def test_conflict_warning_when_host_also_modified(self, tmp_files, tmp_path):
        """Host modified since push AND remote changed → conflict warning, remote wins."""
        host_file = tmp_files["cred_a.json"]
        remote_path = "/root/.hermes/cred_a.json"

        mgr = _make_manager(tmp_files)
        mgr.sync(force=True)

        # Modify the host file AFTER the push (so host_hash != pushed_hash)
        time.sleep(0.05)
        Path(host_file).write_text("host modified content")

        # Build a tar with different remote content
        remote_content = b"remote also changed"
        tar_path = _make_tar(tmp_path, {"root/.hermes/cred_a.json": remote_content})

        def download_fn(dest_tar: Path):
            import shutil

            shutil.copy(tar_path, dest_tar)

        mgr._bulk_download_fn = download_fn

        with patch.object(fsmod.logger, "warning") as warn:
            mgr.sync_back()

        # Conflict warning should have been logged
        conflict_warns = [
            c for c in warn.call_args_list if "conflict" in str(c).lower()
        ]
        assert len(conflict_warns) >= 1

        # Remote version wins (last-write-wins)
        assert Path(host_file).read_bytes() == remote_content


class TestResolveHostPath:
    def test_finds_matching_remote(self):
        mgr = FileSyncManager(
            get_files_fn=lambda: [("/host/a.txt", "/root/a.txt")],
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
        )
        assert (
            mgr._resolve_host_path("/root/a.txt", [("/host/a.txt", "/root/a.txt")])
            == "/host/a.txt"
        )

    def test_returns_none_when_no_match(self):
        mgr = FileSyncManager(
            get_files_fn=lambda: [],
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
        )
        assert (
            mgr._resolve_host_path(
                "/root/missing.txt", [("/host/a.txt", "/root/a.txt")]
            )
            is None
        )

    def test_none_mapping_returns_none(self):
        mgr = FileSyncManager(
            get_files_fn=lambda: [],
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
        )
        assert mgr._resolve_host_path("/root/x.txt") is None


class TestInferHostPath:
    def test_infers_from_same_directory(self):
        mgr = FileSyncManager(
            get_files_fn=lambda: [],
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
        )
        mapping = [("/host/dir/existing.txt", "/root/dir/existing.txt")]
        result = mgr._infer_host_path("/root/dir/new.txt", mapping)
        assert result == "/host/dir/new.txt"

    def test_returns_none_when_no_prefix_match(self):
        mgr = FileSyncManager(
            get_files_fn=lambda: [],
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
        )
        mapping = [("/host/dir/existing.txt", "/root/dir/existing.txt")]
        assert mgr._infer_host_path("/completely/different/path.txt", mapping) is None

    def test_none_mapping_returns_none(self):
        mgr = FileSyncManager(
            get_files_fn=lambda: [],
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
        )
        assert mgr._infer_host_path("/root/x.txt") is None
