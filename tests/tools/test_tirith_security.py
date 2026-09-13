"""Tests for the tirith security scanning subprocess wrapper."""

import io
import json
import os
import subprocess
import tarfile
import time
from unittest.mock import MagicMock, patch

import pytest

import tools.tirith_security as _tirith_mod
from tools.tirith_security import check_command_security, ensure_installed


@pytest.fixture(autouse=True)
def _reset_resolved_path():
    """Pre-set cached path to skip auto-install in scan tests.
    Tests that specifically test ensure_installed / resolve behavior
    reset this to None themselves.
    """
    _tirith_mod._resolved_path = "tirith"
    _tirith_mod._install_thread = None
    _tirith_mod._install_failure_reason = ""
    _tirith_mod._crash_count = 0
    _tirith_mod._circuit_open = False
    yield
    _tirith_mod._resolved_path = None
    _tirith_mod._install_thread = None
    _tirith_mod._install_failure_reason = ""
    _tirith_mod._crash_count = 0
    _tirith_mod._circuit_open = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_run(returncode=0, stdout="", stderr=""):
    """Build a mock subprocess.CompletedProcess."""
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def _json_stdout(findings=None, summary=""):
    return json.dumps({"findings": findings or [], "summary": summary})


# ---------------------------------------------------------------------------
# Exit code → action mapping
# ---------------------------------------------------------------------------

class TestExitCodeMapping:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_0_allow(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(0, _json_stdout())
        result = check_command_security("echo hello")
        assert result["action"] == "allow"
        assert result["findings"] == []

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_1_block_with_findings(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": "homograph_url", "severity": "high"}]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "homograph detected"))
        result = check_command_security("curl http://gооgle.com")
        assert result["action"] == "block"
        assert len(result["findings"]) == 1
        assert result["summary"] == "homograph detected"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_2_warn_with_findings(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": "shortened_url", "severity": "medium"}]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "shortened URL"))
        result = check_command_security("curl https://bit.ly/abc")
        assert result["action"] == "warn"
        assert len(result["findings"]) == 1
        assert result["summary"] == "shortened URL"


# ---------------------------------------------------------------------------
# JSON parse failure (exit code still wins)
# ---------------------------------------------------------------------------

class TestJsonParseFailure:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_1_invalid_json_still_blocks(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(1, "NOT JSON")
        result = check_command_security("bad command")
        assert result["action"] == "block"
        assert "details unavailable" in result["summary"]

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_exit_0_invalid_json_allows(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(0, "NOT JSON")
        result = check_command_security("safe command")
        assert result["action"] == "allow"


# ---------------------------------------------------------------------------
# Operational failures + fail_open
# ---------------------------------------------------------------------------

class TestOSErrorFailOpen:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_file_not_found_fail_open(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = FileNotFoundError("No such file: tirith")
        result = check_command_security("echo hi")
        assert result["action"] == "allow"
        assert "unavailable" in result["summary"]

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_os_error_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.side_effect = FileNotFoundError("No such file: tirith")
        result = check_command_security("echo hi")
        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]


class TestTimeoutFailOpen:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_timeout_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tirith", timeout=5)
        result = check_command_security("slow command")
        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]


class TestUnknownExitCode:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_unknown_exit_code_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        mock_run.return_value = _mock_run(99, "")
        result = check_command_security("cmd")
        assert result["action"] == "block"
        assert "exit code 99" in result["summary"]


# ---------------------------------------------------------------------------
# Disabled
# ---------------------------------------------------------------------------

class TestDisabled:
    @patch("tools.tirith_security._load_security_config")
    def test_disabled_returns_allow(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": False, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        result = check_command_security("rm -rf /")
        assert result["action"] == "allow"


# ---------------------------------------------------------------------------
# Findings cap + summary cap
# ---------------------------------------------------------------------------

class TestCaps:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_findings_and_summary_capped(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": f"rule_{i}"} for i in range(100)]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "x" * 1000))
        result = check_command_security("cmd")
        assert len(result["findings"]) == 50
        assert len(result["summary"]) == 500


# ---------------------------------------------------------------------------
# Programming errors propagate
# ---------------------------------------------------------------------------

class TestProgrammingErrors:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_attribute_error_propagates(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = AttributeError("unexpected bug")
        with pytest.raises(AttributeError):
            check_command_security("cmd")


# ---------------------------------------------------------------------------
# ensure_installed
# ---------------------------------------------------------------------------

class TestEnsureInstalled:
    @patch("tools.tirith_security._load_security_config")
    def test_disabled_returns_none(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": False, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _tirith_mod._resolved_path = None
        assert ensure_installed() is None

    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/tirith")
    @patch("tools.tirith_security._load_security_config")
    def test_found_on_path_returns_immediately(self, mock_cfg, mock_which):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _tirith_mod._resolved_path = None
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = ensure_installed()
        assert result == "/usr/local/bin/tirith"
        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Unsupported platform (Windows etc.) — silent fast-path everywhere
# ---------------------------------------------------------------------------

class TestUnsupportedPlatform:
    """When _detect_target() returns None (no tirith binary for this OS+arch),
    the entire subsystem must stay silent: no PATH probes, no download thread,
    no disk failure marker, no spawn attempts, no CLI banner. Pattern-matching
    guards still cover the gap; tirith content scanning is just absent."""

    @pytest.mark.parametrize("system, machine, expected", [
        ("Linux", "x86_64", True),
        ("Windows", "AMD64", False),
        ("Linux", "riscv64", False),
    ])
    def test_is_platform_supported(self, system, machine, expected):
        # The patched (system, machine) pairs are table inputs, not a host
        # fake: is_platform_supported() is a pure string mapping that touches
        # no OS facility beneath the check, so there is nothing for a real
        # host to falsify. Two of the rows (Windows/AMD64, Linux/riscv64)
        # could never execute honestly anyway — the second has no CI runner
        # on any lane.
        with patch("tools.tirith_security.platform.system", return_value=system), \
             patch("tools.tirith_security.platform.machine", return_value=machine):
            assert _tirith_mod.is_platform_supported() is expected


    @patch("tools.tirith_security._load_security_config")
    def test_check_command_security_unsupported_allows_silently(self, mock_cfg):
        """Windows: skip the resolver and spawn entirely — return allow with
        an empty summary so callers can't accidentally surface 'tirith
        unavailable' messaging to the user."""
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.subprocess.run") as mock_run, \
             patch("tools.tirith_security._resolve_tirith_path") as mock_resolve:
            result = check_command_security("rm -rf /")
            assert result == {"action": "allow", "findings": [], "summary": ""}
            mock_run.assert_not_called()
            mock_resolve.assert_not_called()

    @patch("tools.tirith_security._load_security_config")
    def test_explicit_path_still_honored_on_unsupported_platform(self, mock_cfg):
        """If a user explicitly configured a tirith_path (e.g. they built it
        themselves under WSL), the unsupported-platform short-circuit must
        NOT override that — explicit config wins."""
        mock_cfg.return_value = {"tirith_enabled": True,
                                 "tirith_path": "/opt/custom/tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = _tirith_mod._resolve_tirith_path("/opt/custom/tirith")
            assert result == "/opt/custom/tirith"
            assert _tirith_mod._resolved_path == "/opt/custom/tirith"


# ---------------------------------------------------------------------------
# Failed download caches the miss (Finding #1)
# ---------------------------------------------------------------------------

class TestFailedDownloadCaching:
    @patch("tools.tirith_security._mark_install_failed")
    @patch("tools.tirith_security._is_install_failed_on_disk", return_value=False)
    @patch("tools.tirith_security._install_tirith", return_value=(None, "download_failed"))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_failed_install_cached_no_retry(self, mock_which, mock_install,
                                             mock_disk_check, mock_mark):
        """After a failed download, subsequent resolves must not retry."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = None

        # First call: tries install, fails
        _resolve_tirith_path("tirith")
        assert mock_install.call_count == 1
        assert _tirith_mod._resolved_path is _INSTALL_FAILED
        mock_mark.assert_called_once_with("download_failed")  # reason persisted

        # Second call: hits the cache, does NOT call _install_tirith again
        _resolve_tirith_path("tirith")
        assert mock_install.call_count == 1  # still 1, not 2

        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Explicit path must not auto-download (Finding #2)
# ---------------------------------------------------------------------------

class TestExplicitPathNoAutoDownload:
    @patch("tools.tirith_security._install_tirith")
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_tilde_explicit_path_missing_no_download(self, mock_which, mock_install):
        """An explicit ~/path that doesn't exist must NOT trigger download."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = None

        result = _resolve_tirith_path("~/bin/tirith")
        mock_install.assert_not_called()
        assert _tirith_mod._resolved_path is _INSTALL_FAILED
        assert "~" not in result  # tilde still expanded

        _tirith_mod._resolved_path = None

    @patch("tools.tirith_security._mark_install_failed")
    @patch("tools.tirith_security._is_install_failed_on_disk", return_value=False)
    @patch("tools.tirith_security._install_tirith", return_value=("/auto/tirith", ""))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_default_path_does_auto_download(self, mock_which, mock_install,
                                              mock_disk_check, mock_mark):
        """The default bare 'tirith' SHOULD trigger auto-download."""
        from tools.tirith_security import _resolve_tirith_path
        _tirith_mod._resolved_path = None

        result = _resolve_tirith_path("tirith")
        mock_install.assert_called_once()
        assert result == "/auto/tirith"

        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Cosign provenance verification (P1)
# ---------------------------------------------------------------------------

class TestCosignVerification:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    def test_cosign_identity_pinned_to_release_workflow(self, mock_which, mock_run):
        """Identity regexp must pin to the release workflow, not the whole repo."""
        from tools.tirith_security import _verify_cosign
        mock_run.return_value = _mock_run(0, "Verified OK")
        _verify_cosign("/tmp/checksums.txt", "/tmp/sig", "/tmp/cert")
        args = mock_run.call_args[0][0]
        # Find the value after --certificate-identity-regexp
        idx = args.index("--certificate-identity-regexp")
        identity = args[idx + 1]
        # The identity contains regex-escaped dots
        assert "workflows/release" in identity
        assert "refs/tags/v" in identity


    @patch("tools.tirith_security.tarfile.open")
    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._download_file")
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_proceeds_without_cosign(self, mock_target, mock_dl,
                                              mock_which, mock_checksum,
                                              mock_tarfile):
        """_install_tirith proceeds with SHA-256 only when cosign is not on PATH."""
        from tools.tirith_security import _install_tirith
        mock_tar = MagicMock()
        mock_tar.__enter__ = MagicMock(return_value=mock_tar)
        mock_tar.__exit__ = MagicMock(return_value=False)
        mock_tar.getmembers.return_value = []
        mock_tarfile.return_value = mock_tar

        path, reason = _install_tirith()
        # Reaches extraction (no binary in mock archive), but got past cosign
        assert path is None
        assert reason == "binary_not_in_archive"
        assert mock_checksum.called  # SHA-256 verification ran


class TestInstallArchiveMemberValidation:
    def _write_archive(self, tmp_path, member: tarfile.TarInfo, data: bytes | None = None):
        archive = tmp_path / "tirith-aarch64-apple-darwin.tar.gz"
        checksums = tmp_path / "checksums.txt"
        with tarfile.open(archive, "w:gz") as tar:
            if data is None:
                tar.addfile(member)
            else:
                tar.addfile(member, io.BytesIO(data))
        checksums.write_text(
            "ignored  tirith-aarch64-apple-darwin.tar.gz\n",
            encoding="utf-8",
        )
        return archive, checksums

    def _download_side_effect(self, archive, checksums):
        def _download(url, dest, timeout=10):
            del timeout
            if url.endswith(".tar.gz"):
                with open(archive, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                return
            if url.endswith("checksums.txt"):
                with open(checksums, "rb") as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                return
            raise AssertionError(f"unexpected download URL: {url}")

        return _download

    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_extracts_regular_tirith_member(self, mock_target, mock_which,
                                                    mock_checksum, tmp_path, monkeypatch):
        """A valid regular-file tirith member is installed as a plain file."""
        del mock_target, mock_which, mock_checksum
        from tools.tirith_security import _install_tirith

        payload = b"#!/bin/sh\nexit 0\n"
        member = tarfile.TarInfo("bin/tirith")
        member.mode = 0o755
        member.size = len(payload)
        archive, checksums = self._write_archive(tmp_path, member, payload)

        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("tools.tirith_security._download_file",
                   side_effect=self._download_side_effect(archive, checksums)):
            path, reason = _install_tirith(log_failures=False)

        assert reason == ""
        assert path == str(hermes_home / "bin" / "tirith")
        assert os.path.isfile(path)
        assert not os.path.islink(path)
        with open(path, "rb") as f:
            assert f.read() == payload

    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_rejects_non_regular_tirith_member(self, mock_target, mock_which,
                                                       mock_checksum, tmp_path, monkeypatch):
        """Symlink or hardlink tar members must not be installed as tirith."""
        del mock_target, mock_which, mock_checksum
        from tools.tirith_security import _install_tirith

        member = tarfile.TarInfo("bin/tirith")
        member.type = tarfile.SYMTYPE
        member.linkname = "/bin/sh"
        archive, checksums = self._write_archive(tmp_path, member)

        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("tools.tirith_security._download_file",
                   side_effect=self._download_side_effect(archive, checksums)):
            path, reason = _install_tirith(log_failures=False)

        assert path is None
        assert reason == "binary_not_regular_file"
        assert not os.path.lexists(hermes_home / "bin" / "tirith")


# ---------------------------------------------------------------------------
# Background install / non-blocking startup (P2)
# ---------------------------------------------------------------------------

class TestBackgroundInstall:
    def test_ensure_installed_non_blocking(self):
        """ensure_installed must return immediately when download needed."""
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security._load_security_config",
                   return_value={"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=False), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            MockThread.return_value = mock_thread

            result = ensure_installed()
            assert result is None  # not available yet
            MockThread.assert_called_once()
            mock_thread.start.assert_called_once()

        _tirith_mod._resolved_path = None

    def test_resolve_returns_default_when_thread_alive(self):
        """_resolve_tirith_path returns default while background thread runs."""
        from tools.tirith_security import _resolve_tirith_path
        _tirith_mod._resolved_path = None
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        _tirith_mod._install_thread = mock_thread

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"):
            result = _resolve_tirith_path("tirith")
            assert result == "tirith"  # returns configured default, doesn't block

        _tirith_mod._install_thread = None
        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Disk failure marker persistence (P2)
# ---------------------------------------------------------------------------

class TestDiskFailureMarker:
    def test_expired_marker_ignored(self):
        """Marker older than TTL should be ignored."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        marker = os.path.join(tmpdir, ".tirith-install-failed")
        with patch("tools.tirith_security._failure_marker_path", return_value=marker):
            from tools.tirith_security import _mark_install_failed, _is_install_failed_on_disk
            assert not _is_install_failed_on_disk()
            _mark_install_failed("download_failed")
            assert _is_install_failed_on_disk()
            # Backdate the file past 24h TTL
            old_time = time.time() - 90000  # 25 hours ago
            os.utime(marker, (old_time, old_time))
            assert not _is_install_failed_on_disk()


    def test_in_memory_cosign_exec_failed_not_retried(self):
        """In-memory _INSTALL_FAILED with cosign_exec_failed is NOT retried."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = _INSTALL_FAILED
        _tirith_mod._install_failure_reason = "cosign_exec_failed"

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._install_tirith") as mock_install:
            result = _resolve_tirith_path("tirith")
            assert result == "tirith"  # fallback
            mock_install.assert_not_called()

        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# HERMES_HOME isolation
# ---------------------------------------------------------------------------

class TestHermesHomeIsolation:
    def test_hermes_bin_dir_respects_hermes_home(self):
        """_hermes_bin_dir must use HERMES_HOME, not hardcoded ~/.hermes."""
        from tools.tirith_security import _hermes_bin_dir
        import tempfile
        tmpdir = tempfile.mkdtemp()
        with patch.dict(os.environ, {"HERMES_HOME": tmpdir}):
            result = _hermes_bin_dir()
        assert result == os.path.join(tmpdir, "bin")
        assert os.path.isdir(result)


# ---------------------------------------------------------------------------
# Warn-once dedupe (issue: tirith spawn failed spamming on Windows)
# ---------------------------------------------------------------------------

class TestSpawnWarningDedup:
    """When tirith isn't installed yet (background install in flight, or
    install marked failed), every terminal command spammed an identical
    ``tirith spawn failed: [WinError 2]`` warning to ``errors.log``. The
    dedupe set in ``_warn_once`` collapses repeats by ``(exc class, errno)``
    while still surfacing the first occurrence so users see the failure.
    """

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_repeated_spawn_failure_logs_once(self, mock_cfg, mock_run, caplog):
        mock_cfg.return_value = {
            "tirith_enabled": True, "tirith_path": "tirith",
            "tirith_timeout": 5, "tirith_fail_open": True,
        }
        mock_run.side_effect = FileNotFoundError("[WinError 2]")
        # Fresh dedupe state — clear any keys left by other tests.
        _tirith_mod._warned_messages.clear()

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            for i in range(15):
                result = check_command_security("echo hi")
                # Behavior must remain the same on every call —
                # fail-open allow, with the exception captured in summary.
                assert result["action"] == "allow"
                if i < _tirith_mod._CRASH_LIMIT:
                    # Before circuit breaker opens, summary has the exception
                    assert "unavailable" in result["summary"]
                else:
                    # After circuit breaker opens, summary is generic
                    assert "circuit breaker" in result["summary"]

        spawn_warnings = [
            rec for rec in caplog.records
            if "tirith spawn failed" in rec.message
        ]
        assert len(spawn_warnings) == 1, (
            f"expected exactly 1 spawn-failed warning across 15 commands, "
            f"got {len(spawn_warnings)}: {[r.message for r in spawn_warnings]}"
        )


# ---------------------------------------------------------------------------
# .app TLD suppression (issue #24461)
# ---------------------------------------------------------------------------

_CFG = {"tirith_enabled": True, "tirith_path": "tirith",
        "tirith_timeout": 5, "tirith_fail_open": True}


class TestAppTldSuppression:
    """warn verdicts whose only finding is lookalike_tld/.app are downgraded to allow."""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_app_only_warn_downgraded_to_allow(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        findings = [{"rule_id": "lookalike_tld", "value": ".app",
                     "message": "Domain uses '.app' TLD which can be confused with file extensions"}]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, ".app TLD warning"))
        result = check_command_security("curl https://example.app")
        assert result["action"] == "allow"
        assert result["findings"] == []
        assert result["summary"] == ""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_mixed_findings_preserve_warn(self, mock_cfg, mock_run):
        """If .app finding is accompanied by another finding, warn is preserved."""
        mock_cfg.return_value = _CFG
        findings = [
            {"rule_id": "lookalike_tld", "value": ".app"},
            {"rule_id": "shortened_url", "severity": "medium"},
        ]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "mixed"))
        result = check_command_security("curl https://bit.ly/test.app")
        assert result["action"] == "warn"
        assert len(result["findings"]) == 2

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_block_verdict_never_suppressed(self, mock_cfg, mock_run):
        """block exit code is never downgraded, even if finding looks like .app."""
        mock_cfg.return_value = _CFG
        findings = [{"rule_id": "lookalike_tld", "value": ".app"}]
        mock_run.return_value = _mock_run(1, _json_stdout(findings, "block"))
        result = check_command_security("curl https://example.app")
        assert result["action"] == "block"


class TestIsAppTldFinding:
    """Unit tests for the _is_app_tld_finding helper."""

    @pytest.mark.parametrize("finding, expected", [
        ({"rule_id": "lookalike_tld", "value": ".APP"}, True),   # case-insensitive
        ({"rule_id": "lookalike_tld", "message": "Domain uses '.app' TLD"}, True),
        ({"rule_id": "shortened_url", "value": ".app"}, False),  # wrong rule_id
        ({"rule_id": "lookalike_tld", "value": ".zip"}, False),  # other TLD
    ])
    def test_app_tld_detection(self, finding, expected):
        from tools.tirith_security import _is_app_tld_finding
        assert _is_app_tld_finding(finding) is expected


# ---------------------------------------------------------------------------
# mkdtemp OSError → no_space (disk-full leak prevention)
# ---------------------------------------------------------------------------

class TestMkdtempOSErrorNoSpace:
    """When tempfile.mkdtemp raises OSError (e.g. disk full), _install_tirith
    must return (None, "no_space") instead of propagating the exception.
    This prevents the unbounded retry + temp-dir leak described in #51826.
    """

    def test_mkdtemp_oserror_returns_no_space(self):
        from tools.tirith_security import _install_tirith

        with patch("tools.tirith_security.tempfile.mkdtemp",
                   side_effect=OSError(28, "No space left on device")):
            result, reason = _install_tirith(log_failures=False)
            assert result is None
            assert reason == "no_space"

    def test_mkdtemp_oserror_does_not_leak_tempdir(self):
        """No temp directory should remain after a mkdtemp failure."""
        import glob
        from tools.tirith_security import _install_tirith

        before = set(glob.glob("/tmp/tirith-install-*"))
        with patch("tools.tirith_security.tempfile.mkdtemp",
                   side_effect=OSError(28, "No space left on device")):
            _install_tirith(log_failures=False)
        after = set(glob.glob("/tmp/tirith-install-*"))
        assert after - before == set()


# ---------------------------------------------------------------------------
# Config helpers — _env_bool / _env_int / _load_security_config
# ---------------------------------------------------------------------------

class TestEnvBool:
    """_env_bool: unset -> default; set -> truthy set membership."""

    def test_unset_returns_default(self, monkeypatch):
        from tools.tirith_security import _env_bool
        monkeypatch.delenv("TIRITH_TEST_FLAG", raising=False)
        assert _env_bool("TIRITH_TEST_FLAG", True) is True
        assert _env_bool("TIRITH_TEST_FLAG", False) is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes"])
    def test_truthy_values_return_true(self, monkeypatch, value):
        from tools.tirith_security import _env_bool
        monkeypatch.setenv("TIRITH_TEST_FLAG", value)
        assert _env_bool("TIRITH_TEST_FLAG", False) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_values_return_false(self, monkeypatch, value):
        from tools.tirith_security import _env_bool
        monkeypatch.setenv("TIRITH_TEST_FLAG", value)
        assert _env_bool("TIRITH_TEST_FLAG", True) is False


class TestEnvInt:
    """_env_int: unset -> default; valid int -> value; invalid -> default."""

    def test_unset_returns_default(self, monkeypatch):
        from tools.tirith_security import _env_int
        monkeypatch.delenv("TIRITH_TEST_INT", raising=False)
        assert _env_int("TIRITH_TEST_INT", 7) == 7

    def test_valid_int_is_parsed(self, monkeypatch):
        from tools.tirith_security import _env_int
        monkeypatch.setenv("TIRITH_TEST_INT", "42")
        assert _env_int("TIRITH_TEST_INT", 7) == 42

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        from tools.tirith_security import _env_int
        monkeypatch.setenv("TIRITH_TEST_INT", "abc")
        assert _env_int("TIRITH_TEST_INT", 7) == 7


class TestLoadSecurityConfig:
    """_load_security_config merges config.yaml defaults with env overrides."""

    @patch("hermes_cli.config.load_config_readonly")
    def test_reads_config_security_section(self, mock_load, monkeypatch):
        self._clear_env(monkeypatch)
        mock_load.return_value = {"security": {
            "tirith_enabled": False, "tirith_path": "/cfg/tirith",
            "tirith_timeout": 9, "tirith_fail_open": False,
        }}
        cfg = _tirith_mod._load_security_config()
        assert cfg == {"tirith_enabled": False, "tirith_path": "/cfg/tirith",
                       "tirith_timeout": 9, "tirith_fail_open": False}

    @patch("hermes_cli.config.load_config_readonly")
    def test_env_vars_override_config(self, mock_load):
        mock_load.return_value = {"security": {"tirith_enabled": True,
                                               "tirith_path": "/cfg/tirith",
                                               "tirith_timeout": 9,
                                               "tirith_fail_open": False}}
        with patch.dict(os.environ, {"TIRITH_ENABLED": "false",
                                     "TIRITH_BIN": "/env/tirith",
                                     "TIRITH_TIMEOUT": "3",
                                     "TIRITH_FAIL_OPEN": "true"}):
            cfg = _tirith_mod._load_security_config()
        assert cfg == {"tirith_enabled": False, "tirith_path": "/env/tirith",
                       "tirith_timeout": 3, "tirith_fail_open": True}

    @patch("hermes_cli.config.load_config_readonly")
    def test_missing_security_section_returns_defaults(self, mock_load, monkeypatch):
        self._clear_env(monkeypatch)
        mock_load.return_value = {}
        cfg = _tirith_mod._load_security_config()
        assert cfg == {"tirith_enabled": True, "tirith_path": "tirith",
                       "tirith_timeout": 5, "tirith_fail_open": True}

    @patch("hermes_cli.config.load_config_readonly",
           side_effect=RuntimeError("config corrupt"))
    def test_config_load_exception_uses_defaults(self, mock_load, monkeypatch):
        self._clear_env(monkeypatch)
        cfg = _tirith_mod._load_security_config()
        assert cfg == {"tirith_enabled": True, "tirith_path": "tirith",
                       "tirith_timeout": 5, "tirith_fail_open": True}

    def _clear_env(self, monkeypatch):
        """The repo-wide hermetic fixture sets TIRITH_ENABLED=false; these tests
        exercise the real config loader, so neutralize all TIRITH_* overrides."""
        for var in ("TIRITH_ENABLED", "TIRITH_BIN", "TIRITH_TIMEOUT",
                    "TIRITH_FAIL_OPEN"):
            monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Disk marker + download + checksum + cosign
# ---------------------------------------------------------------------------

class TestDiskMarkerHelpers:
    def test_is_install_failed_cleared_when_cosign_appears(self):
        """cosign_missing marker is auto-cleared once cosign is on PATH."""
        with patch("tools.tirith_security._read_failure_reason",
                   return_value="cosign_missing"), \
             patch("tools.tirith_security.shutil.which",
                   return_value="/usr/bin/cosign"), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear:
            assert _tirith_mod._is_install_failed_on_disk() is False
        mock_clear.assert_called_once()

    def test_mark_install_failed_oserror_is_swallowed(self):
        """OSError while persisting the marker must not propagate."""
        with patch("tools.tirith_security.os.makedirs",
                   side_effect=OSError(13, "Permission denied")):
            # Should not raise
            assert _tirith_mod._mark_install_failed("reason") is None


class TestDownloadFile:
    def _mock_urlopen(self, payload=b"tirith-binary"):
        body = io.BytesIO(payload)
        resp = MagicMock()
        resp.__enter__.return_value = body
        resp.__exit__.return_value = False
        return resp

    def test_download_with_token_adds_auth_header(self, tmp_path):
        dest = str(tmp_path / "payload.bin")
        resp = self._mock_urlopen()
        with patch("tools.tirith_security.urllib.request.urlopen",
                   return_value=resp) as mock_urlopen, \
             patch("agent.secret_scope.get_secret",
                   return_value="ghp_SECRET") as mock_secret:
            _tirith_mod._download_file("https://github.com/rel/tirith.tar.gz",
                                       dest, timeout=5)
        assert mock_secret.assert_called_once_with("GITHUB_TOKEN") is None
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "token ghp_SECRET"
        with open(dest, "rb") as f:
            assert f.read() == b"tirith-binary"

    def test_download_without_token_skips_auth_header(self, tmp_path):
        dest = str(tmp_path / "payload.bin")
        resp = self._mock_urlopen()
        with patch("tools.tirith_security.urllib.request.urlopen",
                   return_value=resp) as mock_urlopen, \
             patch("agent.secret_scope.get_secret", return_value=None):
            _tirith_mod._download_file("https://github.com/rel/checksums.txt",
                                       dest, timeout=5)
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") is None
        with open(dest, "rb") as f:
            assert f.read() == b"tirith-binary"


class TestVerifyCosign:
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_cosign_missing_returns_none(self, mock_which):
        from tools.tirith_security import _verify_cosign
        assert _verify_cosign("/tmp/cs", "/tmp/sig", "/tmp/cert") is None

    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    @patch("tools.tirith_security.subprocess.run")
    def test_cosign_rejected_returns_false(self, mock_run, mock_which):
        from tools.tirith_security import _verify_cosign
        mock_run.return_value = _mock_run(1, "", "signature verification failed")
        assert _verify_cosign("/tmp/cs", "/tmp/sig", "/tmp/cert") is False

    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    @patch("tools.tirith_security.subprocess.run",
           side_effect=OSError("cosign crashed"))
    def test_cosign_oserror_returns_none(self, mock_run, mock_which):
        from tools.tirith_security import _verify_cosign
        assert _verify_cosign("/tmp/cs", "/tmp/sig", "/tmp/cert") is None

    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    @patch("tools.tirith_security.subprocess.run",
           side_effect=subprocess.TimeoutExpired(cmd="cosign", timeout=15))
    def test_cosign_timeout_returns_none(self, mock_run, mock_which):
        from tools.tirith_security import _verify_cosign
        assert _verify_cosign("/tmp/cs", "/tmp/sig", "/tmp/cert") is None


class TestVerifyChecksum:
    def _write(self, tmp_path, digest, archive_name):
        archive = tmp_path / "tirith.tar.gz"
        archive.write_bytes(b"payload")
        checksums = tmp_path / "checksums.txt"
        checksums.write_text(f"{digest}  {archive_name}\n", encoding="utf-8")
        return str(archive), str(checksums)

    def test_checksum_match_returns_true(self, tmp_path):
        from tools.tirith_security import _verify_checksum
        digest = _tirith_mod.hashlib.sha256(b"payload").hexdigest()
        archive, checksums = self._write(tmp_path, digest, "tirith.tar.gz")
        assert _verify_checksum(archive, checksums, "tirith.tar.gz") is True

    def test_checksum_mismatch_returns_false(self, tmp_path):
        from tools.tirith_security import _verify_checksum
        archive, checksums = self._write(tmp_path, "0" * 64, "tirith.tar.gz")
        assert _verify_checksum(archive, checksums, "tirith.tar.gz") is False

    def test_checksum_no_entry_returns_false(self, tmp_path):
        from tools.tirith_security import _verify_checksum
        archive, checksums = self._write(tmp_path, "0" * 64, "other.tar.gz")
        assert _verify_checksum(archive, checksums, "tirith.tar.gz") is False


# ---------------------------------------------------------------------------
# Install branches
# ---------------------------------------------------------------------------

def _write_install_archive(tmp_path, member_name="tirith",
                           payload=b"#!/bin/sh\nexit 0\n"):
    """Write a real tar.gz archive + matching checksums file."""
    archive = tmp_path / "tirith-aarch64-apple-darwin.tar.gz"
    checksums = tmp_path / "checksums.txt"
    member = tarfile.TarInfo(member_name)
    member.mode = 0o755
    member.size = len(payload)
    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(member, io.BytesIO(payload))
    checksums.write_text(
        f"{_tirith_mod.hashlib.sha256(payload).hexdigest()}  "
        f"tirith-aarch64-apple-darwin.tar.gz\n",
        encoding="utf-8",
    )
    return archive, checksums


def _install_downloader(archive, checksums):
    """Download side effect that serves a real archive + checksums + cosign files."""
    def _download(url, dest, timeout=10):
        del timeout
        if url.endswith(".tar.gz"):
            _tirith_mod.shutil.copyfile(archive, dest)
        elif url.endswith("checksums.txt"):
            _tirith_mod.shutil.copyfile(checksums, dest)
        elif url.endswith("checksums.txt.sig") or url.endswith("checksums.txt.pem"):
            with open(dest, "wb") as f:
                f.write(b"")
        else:
            raise AssertionError(f"unexpected download URL: {url}")
    return _download


class TestExtractTirithBinary:
    def test_skips_path_traversal_member(self, tmp_path):
        """A '..'-containing member is skipped; the real 'tirith' is used."""
        from tools.tirith_security import _extract_tirith_binary
        tar_path = tmp_path / "a.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            bad = tarfile.TarInfo("../tirith")
            bad.size = 1
            tar.addfile(bad, io.BytesIO(b"x"))
            good = tarfile.TarInfo("bin/tirith")
            good.size = len(b"payload")
            tar.addfile(good, io.BytesIO(b"payload"))
        dest = tmp_path / "dest"
        dest.mkdir()
        with tarfile.open(tar_path, "r:gz") as tar:
            path, reason = _extract_tirith_binary(tar, str(dest),
                                                  lambda *a, **k: None)
        assert reason == ""
        assert path is not None
        assert os.path.isfile(path)
        with open(path, "rb") as f:
            assert f.read() == b"payload"

    def test_extractfile_none_fails(self):
        """extractfile returning None yields 'binary_extract_failed'."""
        from tools.tirith_security import _extract_tirith_binary
        member = MagicMock()
        member.name = "tirith"
        member.isfile.return_value = True
        tar = MagicMock()
        tar.getmembers.return_value = [member]
        tar.extractfile.return_value = None
        path, reason = _extract_tirith_binary(tar, "/dest", lambda *a, **k: None)
        assert path is None
        assert reason == "binary_extract_failed"


class TestInstallBranches:
    @patch("tools.tirith_security._detect_target", return_value=None)
    def test_install_unsupported_platform(self, mock_target):
        from tools.tirith_security import _install_tirith
        path, reason = _install_tirith()
        assert path is None
        assert reason == "unsupported_platform"

    @patch("tools.tirith_security._download_file",
           side_effect=OSError("network down"))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._detect_target")
    def test_install_download_failure(self, mock_target, mock_which, mock_dl):
        from tools.tirith_security import _install_tirith
        mock_target.return_value = "aarch64-apple-darwin"
        path, reason = _install_tirith()
        assert path is None
        assert reason == "download_failed"

    @patch("tools.tirith_security._verify_cosign", return_value=False)
    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    @patch("tools.tirith_security._download_file",
           side_effect=lambda url, dest, timeout=10: open(dest, "wb").write(b""))
    @patch("tools.tirith_security._detect_target")
    def test_install_cosign_verification_failed(self, mock_target, mock_dl,
                                                mock_which, mock_cosign):
        from tools.tirith_security import _install_tirith
        mock_target.return_value = "aarch64-apple-darwin"
        path, reason = _install_tirith()
        assert path is None
        assert reason == "cosign_verification_failed"

    @patch("tools.tirith_security._verify_checksum", return_value=False)
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._download_file",
           side_effect=lambda url, dest, timeout=10: open(dest, "wb").write(b""))
    @patch("tools.tirith_security._detect_target")
    def test_install_checksum_failed(self, mock_target, mock_dl,
                                     mock_which, mock_checksum):
        from tools.tirith_security import _install_tirith
        mock_target.return_value = "aarch64-apple-darwin"
        path, reason = _install_tirith()
        assert path is None
        assert reason == "checksum_failed"

    def _install_with_archive(self, tmp_path, monkeypatch, cosign_result,
                              target="aarch64-apple-darwin"):
        """Run a real archive install, patching cosign outcome."""
        archive, checksums = _write_install_archive(tmp_path)
        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("tools.tirith_security._download_file",
                   side_effect=_install_downloader(archive, checksums)), \
             patch("tools.tirith_security._detect_target", return_value=target), \
             patch("tools.tirith_security.shutil.which",
                   return_value="/usr/bin/cosign"), \
             patch("tools.tirith_security._verify_checksum",
                   return_value=True), \
             patch("tools.tirith_security._verify_cosign",
                   return_value=cosign_result):
            return _tirith_mod._install_tirith()

    def test_install_cosign_verified_succeeds(self, tmp_path, monkeypatch):
        path, reason = self._install_with_archive(tmp_path, monkeypatch,
                                                  cosign_result=True)
        assert reason == ""
        assert path == str(tmp_path / "hermes-home" / "bin" / "tirith")
        assert path is not None
        assert os.path.isfile(path)

    def test_install_cosign_none_proceeds_sha256_only(self, tmp_path, monkeypatch):
        path, reason = self._install_with_archive(tmp_path, monkeypatch,
                                                  cosign_result=None)
        assert reason == ""
        assert path == str(tmp_path / "hermes-home" / "bin" / "tirith")

    def test_install_cross_device_move_fallback_copy(self, tmp_path, monkeypatch):
        archive, checksums = _write_install_archive(tmp_path)
        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("tools.tirith_security._download_file",
                   side_effect=_install_downloader(archive, checksums)), \
             patch("tools.tirith_security._detect_target",
                   return_value="aarch64-apple-darwin"), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._verify_checksum", return_value=True), \
             patch("tools.tirith_security.shutil.move",
                   side_effect=OSError("cross-device")) as mock_move, \
             patch("tools.tirith_security.shutil.copy",
                   side_effect=lambda s, d: _tirith_mod.shutil.copyfile(s, d)) as mock_copy:
            path, reason = _tirith_mod._install_tirith()
        assert reason == ""
        mock_move.assert_called_once()
        assert mock_copy.called  # fell back to plain copy
        assert path == str(hermes_home / "bin" / "tirith")

    def test_install_cross_device_copy_failed(self, tmp_path, monkeypatch):
        archive, checksums = _write_install_archive(tmp_path)
        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("tools.tirith_security._download_file",
                   side_effect=_install_downloader(archive, checksums)), \
             patch("tools.tirith_security._detect_target",
                   return_value="aarch64-apple-darwin"), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._verify_checksum", return_value=True), \
             patch("tools.tirith_security.shutil.move",
                   side_effect=OSError("cross-device")), \
             patch("tools.tirith_security.shutil.copy",
                   side_effect=OSError("copy fail")), \
             patch("tools.tirith_security.os.unlink",
                   side_effect=OSError(13, "no unlink")) as mock_unlink:
            path, reason = _tirith_mod._install_tirith()
        assert path is None
        assert reason == "cross_device_copy_failed"
        # os.unlink(dest) is called to clean up partial dest; shutil.rmtree
        # in the finally block also calls os.unlink on temp files, so we
        # check the specific call rather than exact count.
        dest = str(hermes_home / "bin" / "tirith")
        mock_unlink.assert_any_call(dest)


# ---------------------------------------------------------------------------
# _resolve_tirith_path branches
# ---------------------------------------------------------------------------

class TestResolveTirithPath:
    def test_unsupported_platform_non_explicit(self):
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security.is_platform_supported",
                   return_value=False), \
             patch("tools.tirith_security.shutil.which") as mock_which:
            result = _tirith_mod._resolve_tirith_path("tirith")
        assert result == "tirith"
        assert _tirith_mod._resolved_path is _tirith_mod._INSTALL_FAILED
        assert _tirith_mod._install_failure_reason == "unsupported_platform"
        mock_which.assert_not_called()

    def test_explicit_path_found_via_which(self):
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security.shutil.which",
                   return_value="/custom/tirith"), \
             patch("os.path.isfile", return_value=False):
            result = _tirith_mod._resolve_tirith_path("custom-tirith")
        assert result == "/custom/tirith"
        assert _tirith_mod._resolved_path == "/custom/tirith"

    def test_default_path_found_on_path(self):
        _tirith_mod._resolved_path = None
        _tirith_mod._install_failure_reason = "download_failed"  # stale
        with patch("tools.tirith_security.shutil.which",
                   return_value="/usr/local/bin/tirith"), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear:
            result = _tirith_mod._resolve_tirith_path("tirith")
        assert result == "/usr/local/bin/tirith"
        assert _tirith_mod._resolved_path == "/usr/local/bin/tirith"
        assert _tirith_mod._install_failure_reason == ""
        mock_clear.assert_called_once()

    def test_default_hermes_bin_found(self, tmp_path):
        _tirith_mod._resolved_path = None
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tirith = bin_dir / "tirith"
        tirith.write_bytes(b"#!/bin/sh\n")
        os.chmod(str(tirith), 0o755)
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value=str(bin_dir)), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear:
            result = _tirith_mod._resolve_tirith_path("tirith")
        assert result == str(tirith)
        assert _tirith_mod._resolved_path == str(tirith)
        mock_clear.assert_called_once()

    def test_in_memory_cosign_missing_retries(self):
        _tirith_mod._resolved_path = _tirith_mod._INSTALL_FAILED
        _tirith_mod._install_failure_reason = "cosign_missing"
        with patch("tools.tirith_security.shutil.which") as mock_which, \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value="/nonexistent"), \
             patch("tools.tirith_security._read_failure_reason",
                   return_value=None), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear, \
             patch("tools.tirith_security._install_tirith",
                   return_value=("/auto/tirith", "")) as mock_install:
            mock_which.side_effect = lambda n: "/usr/bin/cosign" if n == "cosign" else None
            result = _tirith_mod._resolve_tirith_path("tirith")
        assert result == "/auto/tirith"
        mock_install.assert_called_once()
        # Cleared once for the resolved cosign_missing cause, then again on
        # successful re-install.
        assert mock_clear.call_count == 2

    def test_disk_marker_caches_failed(self):
        _tirith_mod._resolved_path = None
        _tirith_mod._install_failure_reason = ""
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value="/nonexistent"), \
             patch("tools.tirith_security._read_failure_reason",
                   return_value="download_failed"), \
             patch("tools.tirith_security._is_install_failed_on_disk",
                   return_value=True), \
             patch("tools.tirith_security._install_tirith") as mock_install:
            result = _tirith_mod._resolve_tirith_path("tirith")
        assert result == "tirith"
        assert _tirith_mod._resolved_path is _tirith_mod._INSTALL_FAILED
        assert _tirith_mod._install_failure_reason == "download_failed"
        mock_install.assert_not_called()


class TestBackgroundInstallBranches:
    def test_early_return_when_already_resolved(self):
        _tirith_mod._resolved_path = "/already/set"
        with patch("tools.tirith_security.shutil.which") as mock_which:
            _tirith_mod._background_install()
        mock_which.assert_not_called()
        assert _tirith_mod._resolved_path == "/already/set"

    def test_which_found(self):
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security.shutil.which",
                   return_value="/usr/bin/tirith"):
            _tirith_mod._background_install()
        assert _tirith_mod._resolved_path == "/usr/bin/tirith"
        assert _tirith_mod._install_failure_reason == ""

    def test_hermes_bin_found(self, tmp_path):
        _tirith_mod._resolved_path = None
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        tirith = bin_dir / "tirith"
        tirith.write_bytes(b"#!/bin/sh\n")
        os.chmod(str(tirith), 0o755)
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value=str(bin_dir)):
            _tirith_mod._background_install()
        assert _tirith_mod._resolved_path == str(tirith)

    def test_install_success(self):
        _tirith_mod._resolved_path = None
        _tirith_mod._install_failure_reason = "stale"
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value="/nonexistent"), \
             patch("tools.tirith_security._install_tirith",
                   return_value=("/auto/tirith", "")), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear:
            _tirith_mod._background_install()
        assert _tirith_mod._resolved_path == "/auto/tirith"
        assert _tirith_mod._install_failure_reason == ""
        mock_clear.assert_called_once()

    def test_install_failure_marks_failed(self):
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value="/nonexistent"), \
             patch("tools.tirith_security._install_tirith",
                   return_value=(None, "download_failed")), \
             patch("tools.tirith_security._mark_install_failed") as mock_mark:
            _tirith_mod._background_install()
        assert _tirith_mod._resolved_path is _tirith_mod._INSTALL_FAILED
        assert _tirith_mod._install_failure_reason == "download_failed"
        mock_mark.assert_called_once_with("download_failed")


class TestEnsureInstalledBranches:
    def test_cached_path_executable_returned(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        f = bin_dir / "tirith"
        f.write_bytes(b"#!/bin/sh\n")
        os.chmod(str(f), 0o755)
        _tirith_mod._resolved_path = str(f)
        with patch("tools.tirith_security._load_security_config",
                   return_value=_CFG):
            assert ensure_installed() == str(f)

    def test_cached_path_not_executable_returns_none(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        f = bin_dir / "tirith"
        f.write_bytes(b"#!/bin/sh\n")  # 0o644, not executable
        _tirith_mod._resolved_path = str(f)
        with patch("tools.tirith_security._load_security_config",
                   return_value=_CFG):
            assert ensure_installed() is None

    def test_unsupported_platform(self):
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security._load_security_config",
                   return_value=_CFG), \
             patch("tools.tirith_security.is_platform_supported",
                   return_value=False):
            assert ensure_installed() is None
        assert _tirith_mod._resolved_path is _tirith_mod._INSTALL_FAILED
        assert _tirith_mod._install_failure_reason == "unsupported_platform"

    def test_explicit_path_isfile_returned(self, tmp_path):
        f = tmp_path / "custom-tirith"
        f.write_bytes(b"#!/bin/sh\n")
        os.chmod(str(f), 0o755)
        cfg = dict(_CFG)
        cfg["tirith_path"] = str(f)
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security._load_security_config",
                   return_value=cfg), \
             patch("tools.tirith_security.is_platform_supported",
                   return_value=True):
            assert ensure_installed() == str(f)

    def test_explicit_path_via_which(self):
        cfg = dict(_CFG)
        cfg["tirith_path"] = "custom-tirith"
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security._load_security_config",
                   return_value=cfg), \
             patch("os.path.isfile", return_value=False), \
             patch("tools.tirith_security.shutil.which",
                   return_value="/opt/tirith"), \
             patch("tools.tirith_security.is_platform_supported",
                   return_value=True):
            assert ensure_installed() == "/opt/tirith"

    def test_explicit_path_missing(self):
        cfg = dict(_CFG)
        cfg["tirith_path"] = "/nope/tirith"
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security._load_security_config",
                   return_value=cfg), \
             patch("os.path.isfile", return_value=False), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security.is_platform_supported",
                   return_value=True):
            assert ensure_installed() is None
        assert _tirith_mod._resolved_path is _tirith_mod._INSTALL_FAILED
        assert _tirith_mod._install_failure_reason == "explicit_path_missing"

    def test_default_hermes_bin_found(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        f = bin_dir / "tirith"
        f.write_bytes(b"#!/bin/sh\n")
        os.chmod(str(f), 0o755)
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security._load_security_config",
                   return_value=_CFG), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value=str(bin_dir)), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear, \
             patch("tools.tirith_security.is_platform_supported",
                   return_value=True):
            assert ensure_installed() == str(f)
        mock_clear.assert_called_once()

    def test_cosign_missing_resolved_retries(self):
        _tirith_mod._resolved_path = _tirith_mod._INSTALL_FAILED
        _tirith_mod._install_failure_reason = "cosign_missing"
        with patch("tools.tirith_security._load_security_config",
                   return_value=_CFG), \
             patch("tools.tirith_security.shutil.which") as mock_which, \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value="/nonexistent"), \
             patch("tools.tirith_security._read_failure_reason",
                   return_value=None), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear, \
             patch("tools.tirith_security.threading.Thread") as MockThread, \
             patch("tools.tirith_security.is_platform_supported",
                   return_value=True):
            mock_which.side_effect = lambda n: "/usr/bin/cosign" if n == "cosign" else None
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            MockThread.return_value = mock_thread
            assert ensure_installed() is None
        mock_clear.assert_called_once()
        MockThread.assert_called_once()

    def test_cosign_missing_not_retryable(self):
        _tirith_mod._resolved_path = _tirith_mod._INSTALL_FAILED
        _tirith_mod._install_failure_reason = "cosign_missing"
        with patch("tools.tirith_security._load_security_config",
                   return_value=_CFG), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value="/nonexistent"), \
             patch("tools.tirith_security.is_platform_supported",
                   return_value=True):
            assert ensure_installed() is None

    def test_disk_marker_returns_none(self):
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security._load_security_config",
                   return_value=_CFG), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir",
                   return_value="/nonexistent"), \
             patch("tools.tirith_security._read_failure_reason",
                   return_value="download_failed"), \
             patch("tools.tirith_security._is_install_failed_on_disk",
                   return_value=True), \
             patch("tools.tirith_security.is_platform_supported",
                   return_value=True):
            assert ensure_installed() is None
        assert _tirith_mod._resolved_path is _tirith_mod._INSTALL_FAILED
        assert _tirith_mod._install_failure_reason == "download_failed"


# ---------------------------------------------------------------------------
# check_command_security edges + helpers
# ---------------------------------------------------------------------------

class TestCheckCommandSecurityEdges:
    @patch("tools.tirith_security._resolve_tirith_path", return_value=None)
    @patch("tools.tirith_security.is_platform_supported", return_value=True)
    @patch("tools.tirith_security._load_security_config", return_value=_CFG)
    def test_resolved_none_fail_open_allows(self, mock_cfg, mock_sup, mock_res):
        result = check_command_security("echo hi")
        assert result["action"] == "allow"
        assert "path unavailable" in result["summary"]

    @patch("tools.tirith_security._resolve_tirith_path", return_value=None)
    @patch("tools.tirith_security.is_platform_supported", return_value=True)
    @patch("tools.tirith_security._load_security_config", return_value=_CFG)
    def test_resolved_none_fail_closed_blocks(self, mock_cfg, mock_sup, mock_res):
        cfg = dict(_CFG)
        cfg["tirith_fail_open"] = False
        with patch("tools.tirith_security._load_security_config",
                   return_value=cfg):
            result = check_command_security("echo hi")
        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]

    @patch("tools.tirith_security._resolve_tirith_path",
           return_value="/usr/bin/tirith")
    @patch("tools.tirith_security.is_platform_supported", return_value=True)
    @patch("tools.tirith_security._load_security_config", return_value=_CFG)
    @patch("tools.tirith_security.subprocess.run",
           side_effect=subprocess.TimeoutExpired(cmd="tirith", timeout=5))
    def test_timeout_fail_open_allows(self, mock_run, mock_cfg, mock_sup, mock_res):
        result = check_command_security("slow command")
        assert result["action"] == "allow"
        assert "timed out" in result["summary"]

    @patch("tools.tirith_security._resolve_tirith_path",
           return_value="/usr/bin/tirith")
    @patch("tools.tirith_security.is_platform_supported", return_value=True)
    @patch("tools.tirith_security._load_security_config", return_value=_CFG)
    @patch("tools.tirith_security.subprocess.run", return_value=_mock_run(99, ""))
    def test_unknown_exit_code_fail_open_allows(self, mock_run, mock_cfg, mock_sup, mock_res):
        result = check_command_security("cmd")
        assert result["action"] == "allow"
        assert "exit code 99" in result["summary"]
        assert "fail-open" in result["summary"]

    @patch("tools.tirith_security._resolve_tirith_path",
           return_value="/usr/bin/tirith")
    @patch("tools.tirith_security.is_platform_supported", return_value=True)
    @patch("tools.tirith_security._load_security_config", return_value=_CFG)
    @patch("tools.tirith_security.subprocess.run", return_value=_mock_run(0, "NOT JSON"))
    def test_json_parse_fail_allow_logs_debug(self, mock_run, mock_cfg, mock_sup, mock_res, caplog):
        with caplog.at_level("DEBUG", logger="tools.tirith_security"):
            result = check_command_security("echo hi")
        assert result["action"] == "allow"
        assert result["summary"] == ""
        assert any("JSON parse failed" in r.message for r in caplog.records)

    @patch("tools.tirith_security._resolve_tirith_path",
           return_value="/usr/bin/tirith")
    @patch("tools.tirith_security.is_platform_supported", return_value=True)
    @patch("tools.tirith_security._load_security_config", return_value=_CFG)
    @patch("tools.tirith_security.subprocess.run", return_value=_mock_run(2, "NOT JSON"))
    def test_json_parse_fail_warn_sets_summary(self, mock_run, mock_cfg, mock_sup, mock_res):
        result = check_command_security("cmd")
        assert result["action"] == "warn"
        assert "details unavailable" in result["summary"]


class TestIsAppTldNonDict:
    @pytest.mark.parametrize("bad_input", [None, "lookalike_tld", 42, []])
    def test_non_dict_input_returns_false(self, bad_input):
        from tools.tirith_security import _is_app_tld_finding
        assert _is_app_tld_finding(bad_input) is False

