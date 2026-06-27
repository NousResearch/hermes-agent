"""Tests for FileSyncManager — mtime tracking, deletion detection, transactional rollback."""

import io
import os
import tarfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.file_sync import (
    FileSyncManager,
    _FORCE_SYNC_ENV,
    iter_sync_files,
    remote_parent_dir,
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


class TestRemoteParentDir:
    """Remote sandbox parent dirs must stay POSIX even on a Windows host."""

    def test_posix_path_parent(self):
        assert remote_parent_dir("/home/user/.hermes/skills/a.py") == "/home/user/.hermes/skills"

    def test_nested_posix_path_parent(self):
        assert remote_parent_dir("/root/.hermes/credentials/b.json") == "/root/.hermes/credentials"

    def test_not_host_os_dependent(self):
        # posixpath.dirname must NOT collapse to a backslash path regardless of
        # the host OS. The remote path is always POSIX; str(Path(...).parent)
        # would mangle it on Windows. This asserts the separators are preserved.
        result = remote_parent_dir("/home/user/.hermes/skills/a.py")
        assert "\\" not in result
        assert result.startswith("/")
