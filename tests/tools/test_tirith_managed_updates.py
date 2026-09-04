"""Behavioral tests for Hermes-managed Tirith background updates."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools.tirith_security as tirith


def _state(path: str = "tirith"):
    return tirith._runtime_state(path)


def _config(path: str = "tirith") -> dict:
    return {
        "tirith_enabled": True,
        "tirith_path": path,
        "tirith_timeout": 5,
        "tirith_fail_open": True,
    }


def _write_executable(path: Path, payload: bytes = b"old tirith") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _linux_acl_blob(entries: list[tuple[int, int, int]]) -> bytes:
    value = bytearray((2).to_bytes(4, "little"))
    for tag, permissions, principal_id in entries:
        value.extend(tag.to_bytes(2, "little"))
        value.extend(permissions.to_bytes(2, "little"))
        value.extend(principal_id.to_bytes(4, "little"))
    return bytes(value)


class _CapturedThread:
    instances = []

    def __init__(self, *, target, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started


@pytest.fixture(autouse=True)
def _reset_update_globals():
    tirith._reset_runtime_states_for_tests()
    with tirith._in_process_update_state_lock:
        tirith._in_process_update_states.clear()
    _CapturedThread.instances = []
    yield
    tirith._reset_runtime_states_for_tests()
    with tirith._in_process_update_state_lock:
        tirith._in_process_update_states.clear()


class TestVersionParsing:
    @pytest.mark.parametrize(
        "output, expected",
        [
            ("tirith 0.4.0\n", (0, 4, 0)),
            ("tirith 0.4.1", (0, 4, 1)),
            ("tirith 1.12.3\n", (1, 12, 3)),
        ],
    )
    def test_accepts_stable_release_versions(self, output, expected):
        assert tirith._parse_tirith_version(output) == expected

    @pytest.mark.parametrize(
        "output",
        [
            "",
            "0.4.1",
            "tirith dev",
            "tirith 0.4.1-dev",
            "tirith 0.4",
            "tirith 0.4.1 unexpected",
        ],
    )
    def test_rejects_unparseable_or_development_versions(self, output):
        assert tirith._parse_tirith_version(output) is None

    @pytest.mark.parametrize(
        "payload, expected",
        [
            (b"prefix\x00tirith 0.4.1\nsuffix", (0, 4, 1)),
            (b"prefixshell-version0.2.12tirith.shsuffix", (0, 2, 12)),
        ],
    )
    def test_reads_historical_release_version_without_execution(
        self, payload, expected, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith", payload)

        assert tirith._read_embedded_tirith_version(str(managed)) == (expected, "")

    def test_rejects_ambiguous_embedded_release_versions(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(
            tmp_path / "bin" / "tirith",
            b"tirith 0.4.0\ntirith 0.4.1\n",
        )

        assert tirith._read_embedded_tirith_version(str(managed)) == (
            None,
            "unparseable",
        )


class TestUpdateState:
    def test_successful_check_is_fresh_for_24_hours(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert tirith._write_update_state("current", now=1_000)

        assert not tirith._update_is_due(now=1_000 + tirith._UPDATE_CHECK_TTL - 1)
        assert tirith._update_is_due(now=1_000 + tirith._UPDATE_CHECK_TTL)

    def test_failure_retries_after_shorter_backoff(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert tirith._write_update_state("failed", now=1_000)

        assert not tirith._update_is_due(now=1_000 + tirith._UPDATE_FAILURE_TTL - 1)
        assert tirith._update_is_due(now=1_000 + tirith._UPDATE_FAILURE_TTL)

    def test_corrupt_or_future_state_never_suppresses_checks(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        state = tmp_path / ".tirith-update-state.json"
        state.write_text("not json", encoding="utf-8")
        assert tirith._update_is_due(now=1_000)

        state.write_text(
            json.dumps({
                "schema_version": 1,
                "checked_at": 2_000,
                "outcome": "current",
            }),
            encoding="utf-8",
        )
        assert tirith._update_is_due(now=1_000)

    @pytest.mark.parametrize(
        "checked_at",
        [10**1000, float("inf"), float("-inf"), float("nan")],
    )
    def test_non_finite_or_overflowing_state_never_suppresses_checks(
        self, tmp_path, monkeypatch, checked_at
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        state = tmp_path / ".tirith-update-state.json"
        state.write_text(
            json.dumps({
                "schema_version": 1,
                "checked_at": checked_at,
                "outcome": "current",
            }),
            encoding="utf-8",
        )

        assert tirith._update_is_due(now=1_000)

    def test_persistence_failure_still_throttles_this_process(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            tirith.os,
            "replace",
            MagicMock(side_effect=OSError("read-only filesystem")),
        )

        assert not tirith._write_update_state("failed", now=1_000)
        assert not (tmp_path / ".tirith-update-state.json").exists()
        assert not tirith._update_is_due(
            now=1_000 + tirith._UPDATE_FAILURE_TTL - 1
        )
        assert tirith._update_is_due(now=1_000 + tirith._UPDATE_FAILURE_TTL)


class TestCrossProcessUpdateLock:
    def test_lock_is_process_bound_and_reusable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        lock = tmp_path / ".tirith-update.lock"
        owner = tirith._acquire_update_lock()
        assert owner is not None
        assert lock.is_file()
        tirith._release_update_lock(owner)

        next_owner = tirith._acquire_update_lock()
        assert next_owner is not None
        tirith._release_update_lock(next_owner)

    @pytest.mark.require_symlinks
    def test_lock_path_symlink_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        external = tmp_path / "external-lock"
        external.write_text("outside", encoding="utf-8")
        (tmp_path / ".tirith-update.lock").symlink_to(external)

        lock_fd, status = tirith._acquire_update_lock_with_status()
        assert lock_fd is None
        assert status == "error"
        assert tirith._acquire_update_lock() is None
        assert external.read_text(encoding="utf-8") == "outside"


_LOCK_WORKER = """
import sys
from tools import tirith_security

lock_fd = tirith_security._acquire_update_lock()
print("locked" if lock_fd is not None else "blocked", flush=True)
if lock_fd is not None and sys.argv[1] == "hold":
    sys.stdin.readline()
if lock_fd is not None:
    tirith_security._release_update_lock(lock_fd)
"""


def _assert_process_lock_excludes_competitor(tmp_path):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)
    owner = subprocess.Popen(
        [sys.executable, "-c", _LOCK_WORKER, "hold"],
        cwd=Path(__file__).parents[2],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert owner.stdout is not None
        assert owner.stdout.readline().strip() == "locked"
        contender = subprocess.run(
            [sys.executable, "-c", _LOCK_WORKER, "once"],
            cwd=Path(__file__).parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        assert contender.stdout.strip() == "blocked"
    finally:
        if owner.stdin is not None:
            owner.stdin.write("release\n")
            owner.stdin.flush()
        owner.communicate(timeout=10)

    successor = subprocess.run(
        [sys.executable, "-c", _LOCK_WORKER, "once"],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert successor.stdout.strip() == "locked"


@pytest.mark.live_system_guard_bypass
@pytest.mark.linux_only
def test_process_lock_excludes_competitor_on_linux(tmp_path):
    _assert_process_lock_excludes_competitor(tmp_path)


@pytest.mark.live_system_guard_bypass
@pytest.mark.macos_only
def test_process_lock_excludes_competitor_on_macos(tmp_path):
    _assert_process_lock_excludes_competitor(tmp_path)


class TestAtomicInstall:
    def test_replaces_existing_binary_atomically(self, tmp_path):
        source = _write_executable(tmp_path / "download" / "tirith", b"new tirith")
        destination = _write_executable(tmp_path / "bin" / "tirith", b"old tirith")

        tirith._atomic_replace_binary(str(source), str(destination))

        assert destination.read_bytes() == b"new tirith"
        assert os.access(destination, os.X_OK)
        assert not list(destination.parent.glob(".tirith-install-*"))

    def test_failed_commit_preserves_working_binary(self, tmp_path, monkeypatch):
        source = _write_executable(tmp_path / "download" / "tirith", b"new tirith")
        destination = _write_executable(tmp_path / "bin" / "tirith", b"old tirith")
        monkeypatch.setattr(
            tirith.os, "replace", MagicMock(side_effect=OSError("busy"))
        )

        with pytest.raises(OSError, match="busy"):
            tirith._atomic_replace_binary(str(source), str(destination))

        assert destination.read_bytes() == b"old tirith"
        assert not list(destination.parent.glob(".tirith-install-*"))

    def test_absent_only_commit_does_not_clobber_concurrent_binary(self, tmp_path):
        source = _write_executable(tmp_path / "download" / "tirith", b"new tirith")
        destination = _write_executable(
            tmp_path / "bin" / "tirith", b"concurrent tirith"
        )

        with pytest.raises(FileExistsError):
            tirith._atomic_replace_binary(
                str(source),
                str(destination),
                require_destination_absent=True,
            )

        assert destination.read_bytes() == b"concurrent tirith"
        assert not list(destination.parent.glob(".tirith-install-*"))

    def test_changed_preimage_is_not_replaced(self, tmp_path):
        source = _write_executable(tmp_path / "download" / "tirith", b"new tirith")
        destination = _write_executable(tmp_path / "bin" / "tirith", b"changed")

        with pytest.raises(OSError, match="changed after"):
            tirith._atomic_replace_binary(
                str(source),
                str(destination),
                expected_existing_sha256="0" * 64,
            )

        assert destination.read_bytes() == b"changed"
        assert not list(destination.parent.glob(".tirith-install-*"))

    @pytest.mark.require_symlinks
    def test_redirected_destination_directory_is_rejected(self, tmp_path):
        source = _write_executable(tmp_path / "download" / "tirith", b"new tirith")
        external = tmp_path / "external"
        destination = _write_executable(external / "tirith", b"outside")
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "bin").symlink_to(external, target_is_directory=True)

        with pytest.raises(OSError):
            tirith._atomic_replace_binary(str(source), str(managed / "bin" / "tirith"))

        assert destination.read_bytes() == b"outside"

    @pytest.mark.live_system_guard_bypass
    @pytest.mark.macos_only
    def test_inherited_mutating_acl_on_staging_file_aborts_replacement(
        self, tmp_path
    ):
        source = _write_executable(tmp_path / "download" / "tirith", b"new tirith")
        destination = _write_executable(tmp_path / "bin" / "tirith", b"old tirith")
        subprocess.run(
            [
                "/bin/chmod",
                "+a",
                "everyone allow write,file_inherit",
                str(destination.parent),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            with pytest.raises(PermissionError, match="staging file"):
                tirith._atomic_replace_binary(str(source), str(destination))

            assert destination.read_bytes() == b"old tirith"
            assert not list(destination.parent.glob(".tirith-install-*"))
        finally:
            subprocess.run(
                ["/bin/chmod", "-N", str(destination.parent)],
                check=True,
                capture_output=True,
                text=True,
            )


class TestSignedReplacementFreshness:
    @pytest.mark.parametrize("candidate_version", [(0, 3, 9), (0, 4, 1)])
    def test_older_or_equal_signed_release_cannot_replace_current_binary(
        self, candidate_version, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        destination = _write_executable(
            tmp_path / "home" / "bin" / "tirith", b"current tirith"
        )
        candidate = _write_executable(
            tmp_path / "download" / "tirith", b"replayed tirith"
        )
        expected_sha256 = tirith._sha256_file(str(destination))
        replace = MagicMock()
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-apple-darwin"
        )
        monkeypatch.setattr(
            tirith,
            "_download_verified_tirith",
            lambda *_args, **_kwargs: (str(candidate), "", True, candidate_version),
        )
        monkeypatch.setattr(tirith, "_atomic_replace_binary", replace)

        installed, reason = tirith._install_tirith(
            log_failures=False,
            expected_existing_sha256=expected_sha256,
            current_version=(0, 4, 1),
        )

        assert installed is None
        assert reason == "candidate_not_newer"
        assert destination.read_bytes() == b"current tirith"
        replace.assert_not_called()

    def test_legacy_bootstrap_rejects_candidate_below_self_update_minimum(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        destination = _write_executable(
            tmp_path / "home" / "bin" / "tirith", b"legacy tirith"
        )
        candidate = _write_executable(
            tmp_path / "download" / "tirith", b"newer legacy tirith"
        )
        replace = MagicMock()
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-apple-darwin"
        )
        monkeypatch.setattr(
            tirith,
            "_download_verified_tirith",
            lambda *_args, **_kwargs: (str(candidate), "", True, (0, 4, 0)),
        )
        monkeypatch.setattr(tirith, "_atomic_replace_binary", replace)

        installed, reason = tirith._install_tirith(
            log_failures=False,
            expected_existing_sha256=tirith._sha256_file(str(destination)),
            current_version=(0, 3, 3),
            minimum_candidate_version=tirith._SELF_UPDATE_MIN_VERSION,
        )

        assert installed is None
        assert reason == "candidate_below_minimum"
        assert destination.read_bytes() == b"legacy tirith"
        replace.assert_not_called()

    def test_newer_signed_release_replaces_preimage_bound_binary(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        destination = _write_executable(
            tmp_path / "home" / "bin" / "tirith", b"current tirith"
        )
        candidate = _write_executable(
            tmp_path / "download" / "tirith", b"new tirith"
        )
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-apple-darwin"
        )
        monkeypatch.setattr(
            tirith,
            "_download_verified_tirith",
            lambda *_args, **_kwargs: (str(candidate), "", True, (0, 4, 2)),
        )

        installed, reason = tirith._install_tirith(
            log_failures=False,
            expected_existing_sha256=tirith._sha256_file(str(destination)),
            current_version=(0, 4, 1),
        )

        assert installed == str(destination)
        assert reason == ""
        assert destination.read_bytes() == b"new tirith"

    def test_same_signed_release_can_replace_verified_cross_abi_binary(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        destination = _write_executable(
            tmp_path / "home" / "bin" / "tirith", b"glibc tirith"
        )
        candidate = _write_executable(
            tmp_path / "download" / "tirith", b"musl tirith"
        )
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-unknown-linux-musl"
        )
        monkeypatch.setattr(
            tirith,
            "_download_verified_tirith",
            lambda *_args, **_kwargs: (str(candidate), "", True, (0, 4, 1)),
        )

        installed, reason = tirith._install_tirith(
            log_failures=False,
            expected_existing_sha256=tirith._sha256_file(str(destination)),
            current_version=(0, 4, 1),
            allow_same_version_replacement=True,
        )

        assert installed == str(destination)
        assert reason == ""
        assert destination.read_bytes() == b"musl tirith"

    def test_same_version_replacement_is_termux_musl_only(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        destination = _write_executable(
            tmp_path / "home" / "bin" / "tirith", b"current tirith"
        )
        download = MagicMock()
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-apple-darwin"
        )
        monkeypatch.setattr(tirith, "_download_verified_tirith", download)

        assert tirith._install_tirith(
            log_failures=False,
            expected_existing_sha256=tirith._sha256_file(str(destination)),
            current_version=(0, 4, 1),
            allow_same_version_replacement=True,
        ) == (None, "invalid_replacement_request")
        download.assert_not_called()

    def test_initial_install_is_create_only_when_destination_already_exists(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        destination = _write_executable(
            tmp_path / "home" / "bin" / "tirith", b"existing tirith"
        )
        download = MagicMock()
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-apple-darwin"
        )
        monkeypatch.setattr(tirith, "_download_verified_tirith", download)

        assert tirith._install_tirith(log_failures=False) == (
            None,
            "destination_exists",
        )
        assert destination.read_bytes() == b"existing tirith"
        download.assert_not_called()


class TestManagedCachePlacement:
    def test_source_and_package_installs_keep_historical_cache_path(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            tirith, "_uses_image_managed_tirith_root", lambda: False
        )

        assert tirith._managed_tirith_path() == str(tmp_path / "bin" / "tirith")

    def test_image_cache_is_platform_qualified_away_from_shared_host_binary(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(tirith, "_uses_image_managed_tirith_root", lambda: True)
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "x86_64-unknown-linux-gnu"
        )
        host_binary = _write_executable(tmp_path / "bin" / "tirith")

        expected_root = (
            tmp_path / ".tirith-managed" / "x86_64-unknown-linux-gnu"
        )
        assert tirith._managed_tirith_path() == str(expected_root / "bin" / "tirith")
        assert not tirith._is_managed_tirith(str(host_binary))
        assert tirith._update_state_path() == str(
            expected_root / ".tirith-update-state.json"
        )
        assert tirith._tirith_subprocess_env()["HERMES_HOME"] == str(expected_root)

    def test_tirith_children_do_not_inherit_hermes_credentials(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
        monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
        monkeypatch.setenv("TIRITH_TEST_SAFE_VALUE", "kept")
        monkeypatch.setattr(
            tirith, "_uses_image_managed_tirith_root", lambda: False
        )

        env = tirith._tirith_subprocess_env()

        assert env["HERMES_HOME"] == str(tmp_path)
        assert env["TIRITH_TEST_SAFE_VALUE"] == "kept"
        assert "GITHUB_TOKEN" not in env
        assert "OPENAI_API_KEY" not in env

    @pytest.mark.parametrize(
        "root",
        [
            "/usr/local",
            "/opt/homebrew",
            "/opt/homebrew/Cellar/tirith/0.4.0",
            "/OPT/HOMEBREW/CELLAR/TIRITH/0.4.0",
            "/home/linuxbrew/.linuxbrew/Cellar/tirith/0.4.0",
            "/nix/store/example-tirith",
            "/nix/var/nix/profiles/default",
        ],
    )
    def test_package_manager_roots_never_gain_managed_ownership(
        self, root
    ):
        assert tirith._managed_tirith_root_is_denied(root)

    def _assert_peer_writable_managed_root_is_not_owned(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = _write_executable(home / "bin" / "tirith")
        home.chmod(0o777)
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert not tirith._is_managed_tirith(str(managed))

    @pytest.mark.linux_only
    def test_peer_writable_managed_root_is_not_owned_linux(
        self, tmp_path, monkeypatch
    ):
        self._assert_peer_writable_managed_root_is_not_owned(tmp_path, monkeypatch)

    @pytest.mark.macos_only
    def test_peer_writable_managed_root_is_not_owned_macos(
        self, tmp_path, monkeypatch
    ):
        self._assert_peer_writable_managed_root_is_not_owned(tmp_path, monkeypatch)

    def _assert_managed_binary_requires_private_owner_executable_mode(
        self, mode, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = _write_executable(home / "bin" / "tirith")
        managed.chmod(mode)
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert not tirith._is_managed_tirith(str(managed))

    @pytest.mark.linux_only
    @pytest.mark.parametrize("mode", [0o775, 0o757, 0o777, 0o655])
    def test_managed_binary_requires_private_owner_executable_mode_linux(
        self, mode, tmp_path, monkeypatch
    ):
        self._assert_managed_binary_requires_private_owner_executable_mode(
            mode, tmp_path, monkeypatch
        )

    @pytest.mark.macos_only
    @pytest.mark.parametrize("mode", [0o775, 0o757, 0o777, 0o655])
    def test_managed_binary_requires_private_owner_executable_mode_macos(
        self, mode, tmp_path, monkeypatch
    ):
        self._assert_managed_binary_requires_private_owner_executable_mode(
            mode, tmp_path, monkeypatch
        )

    def _assert_managed_binary_must_be_owned_by_effective_user(
        self, tmp_path, monkeypatch
    ):
        managed = _write_executable(tmp_path / "tirith")
        executable_stat = managed.lstat()
        stat_fields = list(executable_stat)
        stat_fields[4] = os.geteuid() + 1
        monkeypatch.setattr(
            tirith.os,
            "lstat",
            lambda _path: os.stat_result(stat_fields),
        )
        monkeypatch.setattr(
            tirith,
            "_trusted_unix_acl_is_private",
            lambda *_args, **_kwargs: True,
        )

        assert not tirith._is_owned_private_executable(str(managed))

    @pytest.mark.linux_only
    def test_managed_binary_must_be_owned_by_effective_user_linux(
        self, tmp_path, monkeypatch
    ):
        self._assert_managed_binary_must_be_owned_by_effective_user(
            tmp_path, monkeypatch
        )

    @pytest.mark.macos_only
    def test_managed_binary_must_be_owned_by_effective_user_macos(
        self, tmp_path, monkeypatch
    ):
        self._assert_managed_binary_must_be_owned_by_effective_user(
            tmp_path, monkeypatch
        )

    @pytest.mark.parametrize(
        "entries, expected",
        [
            ([(0x01, 7, 0xFFFF_FFFF), (0x02, 7, 31_337)], False),
            ([(0x01, 7, 0xFFFF_FFFF), (0x02, 5, 31_337)], True),
            ([(0x01, 7, 0xFFFF_FFFF), (0x08, 7, 12_345)], False),
            ([(0x01, 7, 0xFFFF_FFFF), (0x02, 7, 501)], True),
        ],
    )
    def test_linux_acl_policy_rejects_only_foreign_mutation_grants(
        self, entries, expected
    ):
        assert tirith._linux_posix_acl_blob_is_private(
            _linux_acl_blob(entries),
            owner_uid=501,
            effective_uid=501,
        ) is expected

    @pytest.mark.parametrize(
        "blob",
        [
            b"",
            (3).to_bytes(4, "little"),
            (2).to_bytes(4, "little") + b"short",
            _linux_acl_blob([(0x40, 7, 1)]),
        ],
    )
    def test_linux_acl_policy_fails_closed_on_malformed_or_unknown_data(
        self, blob
    ):
        assert not tirith._linux_posix_acl_blob_is_private(
            blob,
            owner_uid=501,
            effective_uid=501,
        )

    @pytest.mark.live_system_guard_bypass
    @pytest.mark.macos_only
    @pytest.mark.parametrize("component", ["binary", "directory"])
    def test_macos_mutating_acl_is_rejected_even_with_private_mode_bits(
        self, component, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = _write_executable(home / "bin" / "tirith")
        target = managed if component == "binary" else managed.parent
        monkeypatch.setenv("HERMES_HOME", str(home))
        subprocess.run(
            ["/bin/chmod", "+a", "everyone allow write", str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            assert target.stat().st_mode & 0o777 == 0o755
            assert not tirith._is_managed_tirith(str(managed))
        finally:
            subprocess.run(
                ["/bin/chmod", "-N", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )

    @pytest.mark.live_system_guard_bypass
    @pytest.mark.macos_only
    def test_macos_deny_only_acl_does_not_widen_mutation_authority(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = _write_executable(home / "bin" / "tirith")
        monkeypatch.setenv("HERMES_HOME", str(home))
        subprocess.run(
            ["/bin/chmod", "+a", "everyone deny delete", str(managed)],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            assert tirith._is_managed_tirith(str(managed))
        finally:
            subprocess.run(
                ["/bin/chmod", "-N", str(managed)],
                check=True,
                capture_output=True,
                text=True,
            )

    @pytest.mark.require_symlinks
    def test_image_cache_intermediate_symlink_cannot_redirect_install(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        home.mkdir()
        external = tmp_path / "external"
        target = "x86_64-unknown-linux-gnu"
        outside_binary = _write_executable(
            external / target / "bin" / "tirith", b"outside"
        )
        (home / ".tirith-managed").symlink_to(
            external, target_is_directory=True
        )
        verified = _write_executable(tmp_path / "verified" / "tirith", b"new")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(tirith, "_uses_image_managed_tirith_root", lambda: True)
        monkeypatch.setattr(tirith, "_detect_target", lambda: target)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith,
            "_download_verified_tirith",
            lambda *_args, **_kwargs: (str(verified), "", True, (0, 4, 1)),
        )

        installed, reason = tirith._install_tirith(log_failures=False)

        assert installed is None
        assert reason == "destination_exists"
        assert outside_binary.read_bytes() == b"outside"
        assert not tirith._is_managed_tirith(
            str(home / ".tirith-managed" / target / "bin" / "tirith")
        )


class TestUpdateScheduling:
    def test_unsupported_manager_never_schedules_managed_update(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: False)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(tirith, "_update_is_due", lambda: True)
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        tirith._schedule_managed_update(str(managed), "tirith")

        assert not _CapturedThread.instances

    def test_managed_cache_returns_immediately_and_schedules_background_update(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(tirith, "_load_security_config", lambda: _config())
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(tirith, "_update_is_due", lambda: True)
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        assert tirith.ensure_installed(log_failures=False) == str(managed)

        assert len(_CapturedThread.instances) == 1
        thread = _CapturedThread.instances[0]
        assert thread.target.__name__ == "run"
        assert thread.args == (tirith._background_update, str(managed))
        assert thread.kwargs == {"log_failures": False}
        assert thread.daemon is True
        assert thread.started

    def test_external_path_binary_is_never_updated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        external = _write_executable(tmp_path / "external" / "tirith")
        monkeypatch.setattr(tirith, "_load_security_config", lambda: _config())
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.shutil, "which", lambda _name: str(external))
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        assert tirith.ensure_installed() == str(external)
        assert not _CapturedThread.instances

    @pytest.mark.require_symlinks
    def test_managed_path_symlink_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        external = _write_executable(tmp_path / "external" / "tirith")
        managed = tmp_path / "home" / "bin" / "tirith"
        managed.parent.mkdir(parents=True)
        managed.symlink_to(external)
        monkeypatch.setattr(tirith, "_load_security_config", lambda: _config())
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        assert tirith.ensure_installed() is None
        assert external.read_bytes() == b"old tirith"
        assert not _CapturedThread.instances

    @pytest.mark.require_symlinks
    def test_managed_parent_symlink_is_rejected(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        external_dir = tmp_path / "external"
        external = _write_executable(external_dir / "tirith")
        (home / "bin").symlink_to(external_dir, target_is_directory=True)
        managed = home / "bin" / "tirith"
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(tirith, "_load_security_config", lambda: _config())
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        assert tirith.ensure_installed() is None
        assert managed.samefile(external)
        assert external.read_bytes() == b"old tirith"
        assert not _CapturedThread.instances

    def test_explicit_binary_is_never_updated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        explicit = _write_executable(tmp_path / "custom" / "tirith")
        monkeypatch.setattr(
            tirith, "_load_security_config", lambda: _config(str(explicit))
        )
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        assert tirith.ensure_installed() == str(explicit)
        assert not _CapturedThread.instances

    def test_runtime_install_opt_out_disables_updates_not_scanning(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(tirith, "_load_security_config", lambda: _config())
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: False)
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        assert tirith.ensure_installed() == str(managed)
        assert not _CapturedThread.instances

    def test_fresh_state_skips_thread(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        assert tirith._write_update_state("current")
        monkeypatch.setattr(tirith, "_load_security_config", lambda: _config())
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        assert tirith.ensure_installed() == str(managed)
        assert not _CapturedThread.instances

    def test_only_one_update_thread_is_scheduled_per_process(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(tirith, "_load_security_config", lambda: _config())
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(tirith, "_update_is_due", lambda: True)
        monkeypatch.setattr(tirith.threading, "Thread", _CapturedThread)

        assert tirith.ensure_installed() == str(managed)
        assert tirith.ensure_installed() == str(managed)
        assert len(_CapturedThread.instances) == 1

    def test_long_lived_process_rechecks_after_release_ttl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(tirith, "_load_security_config", lambda: _config())
        monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
        monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        maintain = MagicMock(side_effect=["current", "updated"])
        monkeypatch.setattr(tirith, "_maintain_managed_tirith", maintain)

        # Startup performs the first maintenance check.
        assert tirith.ensure_installed(log_failures=False) == str(managed)
        first_worker = _state().update_thread
        assert first_worker is not None
        first_worker.join(timeout=2)
        assert not first_worker.is_alive()
        first_state = tirith._read_update_state()
        assert first_state is not None
        assert first_state["outcome"] == "current"

        # A normal command in the same process must not launch another worker
        # while the successful-check TTL is fresh.
        assert tirith._resolve_tirith_path("tirith") == str(managed)
        assert _state().update_thread is first_worker

        # Model a Tirith release after startup by expiring the check state.
        assert tirith._write_update_state("current", now=1)
        assert tirith._resolve_tirith_path("tirith") == str(managed)
        second_worker = _state().update_thread
        assert second_worker is not None
        assert second_worker is not first_worker
        second_worker.join(timeout=2)
        assert not second_worker.is_alive()

        assert maintain.call_count == 2
        second_state = tirith._read_update_state()
        assert second_state is not None
        assert second_state["outcome"] == "updated"


class TestLegacyReleaseProof:
    @pytest.mark.parametrize(
        "installed_bytes, expected_reason",
        [
            (b"official release", ""),
            (b"custom build", "binary_mismatch"),
        ],
    )
    def test_only_published_release_bytes_are_trusted(
        self, installed_bytes, expected_reason, tmp_path, monkeypatch
    ):
        managed = _write_executable(tmp_path / "bin" / "tirith", installed_bytes)
        monkeypatch.setattr(tirith, "_detect_target", lambda: "test-target")

        def fake_download(_base_url, _target, workdir, _log):
            released = Path(workdir) / "released-tirith"
            released.write_bytes(b"official release")
            return str(released), "", True, (0, 4, 1)

        monkeypatch.setattr(tirith, "_download_verified_tirith", fake_download)

        digest, reason = tirith._verify_legacy_release_binary(
            str(managed), (0, 4, 0), log_failures=False
        )

        assert reason == expected_reason
        if expected_reason:
            assert digest is None
        else:
            assert digest == tirith._sha256_file(str(managed))


class TestMaintenanceDecision:
    def test_pre_041_cache_is_bootstrapped_with_verified_installer(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 0), "")
        )
        monkeypatch.setattr(
            tirith,
            "_verify_legacy_release_binary",
            lambda *_args, **_kwargs: ("a" * 64, ""),
        )
        install = MagicMock(return_value=(str(managed), ""))
        update = MagicMock()
        monkeypatch.setattr(tirith, "_install_tirith", install)
        monkeypatch.setattr(tirith, "_run_tirith_update", update)

        assert (
            tirith._maintain_managed_tirith(str(managed), log_failures=False)
            == "bootstrapped"
        )
        install.assert_called_once_with(
            log_failures=False,
            expected_existing_sha256="a" * 64,
            current_version=(0, 4, 0),
            minimum_candidate_version=tirith._SELF_UPDATE_MIN_VERSION,
        )
        update.assert_not_called()

    def test_pre_041_replayed_release_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 0), "")
        )
        monkeypatch.setattr(
            tirith,
            "_verify_legacy_release_binary",
            lambda *_args, **_kwargs: ("a" * 64, ""),
        )
        monkeypatch.setattr(
            tirith,
            "_install_tirith",
            lambda **_kwargs: (None, "candidate_not_newer"),
        )

        assert tirith._maintain_managed_tirith(str(managed)) == "current"

    def test_pre_041_custom_build_is_left_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith", b"custom build")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 0), "")
        )
        monkeypatch.setattr(
            tirith,
            "_verify_legacy_release_binary",
            lambda *_args, **_kwargs: (None, "binary_mismatch"),
        )
        install = MagicMock()
        monkeypatch.setattr(tirith, "_install_tirith", install)

        assert tirith._maintain_managed_tirith(str(managed)) == "skipped"
        assert managed.read_bytes() == b"custom build"
        install.assert_not_called()

    def test_unparseable_version_is_left_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: (None, "unparseable")
        )
        install = MagicMock()
        update = MagicMock()
        monkeypatch.setattr(tirith, "_install_tirith", install)
        monkeypatch.setattr(tirith, "_run_tirith_update", update)

        assert (
            tirith._maintain_managed_tirith(str(managed), log_failures=False)
            == "skipped"
        )
        assert managed.read_bytes() == b"old tirith"
        install.assert_not_called()
        update.assert_not_called()

    def test_development_build_is_left_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 1), "")
        )
        monkeypatch.setattr(
            tirith,
            "_probe_tirith_provenance",
            lambda _path: (
                {
                    "version": "0.4.1",
                    "binary_path": str(managed),
                    "install_method": "hermes",
                    "install_method_resolved": True,
                    "dev_build": True,
                },
                "",
            ),
        )
        update = MagicMock()
        monkeypatch.setattr(tirith, "_run_tirith_update", update)

        assert (
            tirith._maintain_managed_tirith(str(managed), log_failures=False)
            == "skipped"
        )
        update.assert_not_called()

    def test_policy_change_during_probes_defers_modern_update(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 1), "")
        )
        monkeypatch.setattr(
            tirith,
            "_probe_tirith_provenance",
            lambda _path: (
                {
                    "version": "0.4.1",
                    "binary_path": str(managed),
                    "install_method": "hermes",
                    "install_method_resolved": True,
                    "dev_build": False,
                },
                "",
            ),
        )
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: False)
        run = MagicMock()
        monkeypatch.setattr(tirith.subprocess, "run", run)

        assert tirith._maintain_managed_tirith(str(managed)) == "deferred"
        run.assert_not_called()

    @pytest.mark.parametrize("version", [(0, 4, 1), (0, 4, 2), (0, 4, 99)])
    def test_compatible_release_build_delegates_to_tirith_self_update(
        self, version, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: (version, "")
        )
        version_text = ".".join(str(part) for part in version)
        monkeypatch.setattr(
            tirith,
            "_probe_tirith_provenance",
            lambda _path: (
                {
                    "version": version_text,
                    "binary_path": str(managed),
                    "install_method": "hermes",
                    "install_method_resolved": True,
                    "dev_build": False,
                },
                "",
            ),
        )
        update = MagicMock(return_value="current")
        monkeypatch.setattr(tirith, "_run_tirith_update", update)
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-apple-darwin"
        )

        assert (
            tirith._maintain_managed_tirith(str(managed), log_failures=False)
            == "current"
        )
        update.assert_called_once_with(str(managed))

    def test_termux_release_build_uses_verified_musl_installer(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        expected_sha256 = tirith._sha256_file(str(managed))
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 1), "")
        )
        monkeypatch.setattr(
            tirith,
            "_probe_tirith_provenance",
            lambda _path: (
                {
                    "version": "0.4.1",
                    "binary_path": str(managed),
                    "install_method": "hermes",
                    "install_method_resolved": True,
                    "dev_build": False,
                },
                "",
            ),
        )
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-unknown-linux-musl"
        )
        install = MagicMock(return_value=(str(managed), ""))
        update = MagicMock()
        monkeypatch.setattr(
            tirith,
            "_verify_termux_release_binary",
            lambda *_args, **_kwargs: (
                expected_sha256,
                "aarch64-unknown-linux-musl",
                "",
            ),
        )
        monkeypatch.setattr(tirith, "_install_tirith", install)
        monkeypatch.setattr(tirith, "_run_tirith_update", update)

        assert (
            tirith._maintain_managed_tirith(str(managed), log_failures=False)
            == "updated"
        )
        install.assert_called_once_with(
            log_failures=False,
            expected_existing_sha256=expected_sha256,
            current_version=(0, 4, 1),
            minimum_candidate_version=(0, 4, 1),
            allow_same_version_replacement=False,
        )
        update.assert_not_called()

    def test_unexecutable_historical_termux_glibc_release_is_migrated(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(
            tmp_path / "bin" / "tirith",
            b"prefix\x00tirith 0.4.1\nsuffix",
        )
        expected_sha256 = tirith._sha256_file(str(managed))
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: (None, "probe_failed")
        )
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-unknown-linux-musl"
        )
        verify = MagicMock(
            return_value=(
                expected_sha256,
                "aarch64-unknown-linux-gnu",
                "",
            )
        )
        provenance = MagicMock()
        install = MagicMock(return_value=(str(managed), ""))
        monkeypatch.setattr(tirith, "_verify_termux_release_binary", verify)
        monkeypatch.setattr(tirith, "_probe_tirith_provenance", provenance)
        monkeypatch.setattr(tirith, "_install_tirith", install)

        assert (
            tirith._maintain_managed_tirith(str(managed), log_failures=False)
            == "updated"
        )
        verify.assert_called_once_with(
            str(managed),
            (0, 4, 1),
            log_failures=False,
        )
        provenance.assert_not_called()
        install.assert_called_once_with(
            log_failures=False,
            expected_existing_sha256=expected_sha256,
            current_version=(0, 4, 1),
            minimum_candidate_version=(0, 4, 1),
            allow_same_version_replacement=True,
        )

    def test_unexecutable_unknown_termux_binary_is_never_replaced(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(
            tmp_path / "bin" / "tirith",
            b"prefix\x00tirith 0.4.1\nsuffix",
        )
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: (None, "probe_failed")
        )
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-unknown-linux-musl"
        )
        monkeypatch.setattr(
            tirith,
            "_verify_termux_release_binary",
            lambda *_args, **_kwargs: (None, None, "binary_mismatch"),
        )
        install = MagicMock()
        monkeypatch.setattr(tirith, "_install_tirith", install)

        assert tirith._maintain_managed_tirith(str(managed)) == "skipped"
        install.assert_not_called()

    def test_termux_replayed_release_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 1), "")
        )
        monkeypatch.setattr(
            tirith,
            "_probe_tirith_provenance",
            lambda _path: (
                {
                    "version": "0.4.1",
                    "binary_path": str(managed),
                    "install_method": "hermes",
                    "install_method_resolved": True,
                    "dev_build": False,
                },
                "",
            ),
        )
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-unknown-linux-musl"
        )
        monkeypatch.setattr(
            tirith,
            "_verify_termux_release_binary",
            lambda *_args, **_kwargs: (
                tirith._sha256_file(str(managed)),
                "aarch64-unknown-linux-musl",
                "",
            ),
        )
        monkeypatch.setattr(
            tirith,
            "_install_tirith",
            lambda **_kwargs: (None, "candidate_not_newer"),
        )

        assert tirith._maintain_managed_tirith(str(managed)) == "current"

    def test_termux_release_with_native_musl_provenance_uses_self_update(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 2), "")
        )
        monkeypatch.setattr(
            tirith,
            "_probe_tirith_provenance",
            lambda _path: (
                {
                    "version": "0.4.2",
                    "binary_path": str(managed),
                    "install_method": "hermes",
                    "install_method_resolved": True,
                    "dev_build": False,
                    "target": "aarch64-unknown-linux-musl",
                },
                "",
            ),
        )
        monkeypatch.setattr(
            tirith, "_detect_target", lambda: "aarch64-unknown-linux-musl"
        )
        install = MagicMock()
        update = MagicMock(return_value="current")
        monkeypatch.setattr(tirith, "_install_tirith", install)
        monkeypatch.setattr(tirith, "_run_tirith_update", update)

        assert tirith._maintain_managed_tirith(str(managed)) == "current"
        install.assert_not_called()
        update.assert_called_once_with(str(managed))

    @pytest.mark.parametrize(
        "override",
        [
            {"install_method": "homebrew"},
            {"install_method_resolved": False},
            {"version": "0.4.2"},
            {"binary_path": "/somewhere/else/tirith"},
        ],
    )
    def test_mismatched_provenance_is_left_untouched(
        self, override, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        provenance = {
            "version": "0.4.1",
            "binary_path": str(managed),
            "install_method": "hermes",
            "install_method_resolved": True,
            "dev_build": False,
        }
        provenance.update(override)
        monkeypatch.setattr(
            tirith, "_probe_tirith_version", lambda _path: ((0, 4, 1), "")
        )
        monkeypatch.setattr(
            tirith,
            "_probe_tirith_provenance",
            lambda _path: (provenance, ""),
        )
        update = MagicMock()
        monkeypatch.setattr(tirith, "_run_tirith_update", update)

        assert tirith._maintain_managed_tirith(str(managed)) == "skipped"
        assert managed.read_bytes() == b"old tirith"
        update.assert_not_called()


class TestSelfUpdateSubprocess:
    @pytest.fixture(autouse=True)
    def _allow_runtime_updates(self, monkeypatch):
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)

    @pytest.mark.parametrize(
        "payload, expected",
        [
            ({"action": "none", "message": "already up to date"}, "current"),
            ({"action": "updated", "new_version": "0.4.2"}, "updated"),
        ],
    )
    def test_invokes_noninteractive_signed_update(
        self, payload, expected, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        run = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(payload), stderr=""
            )
        )
        monkeypatch.setattr(tirith.subprocess, "run", run)

        assert tirith._run_tirith_update(str(managed)) == expected

        args, kwargs = run.call_args
        assert args[0] == [
            str(managed),
            "update",
            "--yes",
            "--format",
            "json",
        ]
        assert "--allow-unsigned" not in args[0]
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["timeout"] == tirith._UPDATE_TIMEOUT
        assert kwargs["env"]["HERMES_HOME"] == str(tmp_path)

    def test_failed_update_keeps_scanner_path_and_binary(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        _state().resolved_path = str(managed)
        monkeypatch.setattr(
            tirith.subprocess,
            "run",
            MagicMock(
                side_effect=subprocess.TimeoutExpired(cmd="tirith update", timeout=120)
            ),
        )

        assert tirith._run_tirith_update(str(managed)) == "failed"
        assert _state().resolved_path == str(managed)
        assert managed.read_bytes() == b"old tirith"

    def _assert_mode_drift_before_update_prevents_spawn(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        managed.chmod(0o775)
        run = MagicMock()
        monkeypatch.setattr(tirith.subprocess, "run", run)

        assert tirith._run_tirith_update(str(managed)) == "failed"
        run.assert_not_called()

    @pytest.mark.linux_only
    def test_mode_drift_before_update_prevents_spawn_linux(
        self, tmp_path, monkeypatch
    ):
        self._assert_mode_drift_before_update_prevents_spawn(tmp_path, monkeypatch)

    @pytest.mark.macos_only
    def test_mode_drift_before_update_prevents_spawn_macos(
        self, tmp_path, monkeypatch
    ):
        self._assert_mode_drift_before_update_prevents_spawn(tmp_path, monkeypatch)

    def test_signature_required_failure_keeps_binary_without_weaker_retry(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        _state().resolved_path = str(managed)
        run = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout=json.dumps({"action": "error"}),
                stderr="release signature verification is required",
            )
        )
        monkeypatch.setattr(tirith.subprocess, "run", run)

        assert tirith._run_tirith_update(str(managed)) == "failed"
        assert _state().resolved_path == str(managed)
        assert managed.read_bytes() == b"old tirith"
        run.assert_called_once()
        assert "--allow-unsigned" not in run.call_args.args[0]

    def test_unexpected_success_payload_is_not_treated_as_fresh(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        monkeypatch.setattr(
            tirith.subprocess,
            "run",
            MagicMock(
                return_value=subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout=json.dumps({"action": "use-package-manager"}),
                    stderr="",
                )
            ),
        )

        assert tirith._run_tirith_update(str(managed)) == "failed"


class TestBackgroundWorkerIsolation:
    def test_failure_sets_backoff_without_disabling_working_scanner(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        _state().resolved_path = str(managed)
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith, "_maintain_managed_tirith", lambda *_a, **_kw: "failed"
        )

        tirith._background_update(str(managed), log_failures=False)

        assert _state().resolved_path == str(managed)
        assert managed.read_bytes() == b"old tirith"
        assert (tmp_path / ".tirith-update.lock").is_file()
        assert not (tmp_path / ".tirith-install-failed").exists()
        assert _state().install_failure_reason == ""
        state = json.loads(
            (tmp_path / ".tirith-update-state.json").read_text(encoding="utf-8")
        )
        assert state["outcome"] == "failed"

    def test_active_process_lock_prevents_duplicate_update(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        lock_fd = tirith._acquire_update_lock()
        assert lock_fd is not None
        maintain = MagicMock(return_value="current")
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(tirith, "_maintain_managed_tirith", maintain)

        try:
            tirith._background_update(str(managed), log_failures=False)
        finally:
            tirith._release_update_lock(lock_fd)

        maintain.assert_not_called()
        assert tirith._read_update_state() is None

    def test_operational_lock_error_sets_short_backoff(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        maintain = MagicMock(return_value="current")
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(
            tirith,
            "_acquire_update_lock_with_status",
            lambda: (None, "error"),
        )
        monkeypatch.setattr(tirith, "_maintain_managed_tirith", maintain)

        tirith._background_update(str(managed), log_failures=False)

        maintain.assert_not_called()
        state = tirith._read_update_state()
        assert state is not None
        assert state["outcome"] == "failed"
        assert not tirith._update_is_due(
            now=state["checked_at"] + tirith._UPDATE_FAILURE_TTL - 1
        )

    def test_shared_fresh_state_is_rechecked_after_lock(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        managed = _write_executable(tmp_path / "bin" / "tirith")
        assert tirith._write_update_state("updated")
        maintain = MagicMock(return_value="current")
        monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)
        monkeypatch.setattr(tirith, "_maintain_managed_tirith", maintain)

        tirith._background_update(str(managed), log_failures=False)

        maintain.assert_not_called()
        assert (tmp_path / ".tirith-update.lock").is_file()


def _assert_managed_update_end_to_end_with_real_subprocess(tmp_path, monkeypatch):
    """Exercise discovery, provenance, update, locking, and persisted state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    managed = tmp_path / "bin" / "tirith"
    provenance = json.dumps({
        "version": "0.4.1",
        "binary_path": str(managed),
        "install_method": "hermes",
        "install_method_resolved": True,
        "dev_build": False,
    })
    script = f"""#!/bin/sh
if [ "$1" = "--version" ]; then
    printf 'tirith 0.4.1\\n'
elif [ "$1" = "version" ]; then
    printf '%s\\n' '{provenance}'
elif [ "$1" = "update" ]; then
    printf '%s\\n' "$*" > "$HERMES_HOME/update-argv"
    printf '%s\\n' '{{"action":"none"}}'
else
    exit 64
fi
"""
    _write_executable(managed, script.encode())
    monkeypatch.setattr(tirith, "_tirith_auto_install_allowed", lambda: True)

    tirith._background_update(str(managed), log_failures=False)

    assert (tmp_path / "update-argv").read_text(encoding="utf-8").strip() == (
        "update --yes --format json"
    )
    state = json.loads(
        (tmp_path / ".tirith-update-state.json").read_text(encoding="utf-8")
    )
    assert state["outcome"] == "current"
    assert (tmp_path / ".tirith-update.lock").is_file()


@pytest.mark.live_system_guard_bypass
@pytest.mark.linux_only
def test_managed_update_end_to_end_on_linux(tmp_path, monkeypatch):
    _assert_managed_update_end_to_end_with_real_subprocess(tmp_path, monkeypatch)


@pytest.mark.live_system_guard_bypass
@pytest.mark.macos_only
def test_managed_update_end_to_end_on_macos(tmp_path, monkeypatch):
    _assert_managed_update_end_to_end_with_real_subprocess(tmp_path, monkeypatch)
