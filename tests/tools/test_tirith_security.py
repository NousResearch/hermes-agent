"""Tests for the tirith security scanning subprocess wrapper."""

import io
import json
import logging
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
    _tirith_mod._circuit_open_at = 0.0
    yield
    _tirith_mod._resolved_path = None
    _tirith_mod._install_thread = None
    _tirith_mod._install_failure_reason = ""
    _tirith_mod._crash_count = 0
    _tirith_mod._circuit_open = False
    _tirith_mod._circuit_open_at = 0.0


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
# Circuit breaker: half-open recovery
# ---------------------------------------------------------------------------

def _open_breaker(age_s):
    """Put the breaker in the open state as if it tripped ``age_s`` seconds ago."""
    _tirith_mod._crash_count = _tirith_mod._CRASH_LIMIT
    _tirith_mod._circuit_open = True
    _tirith_mod._circuit_open_at = time.monotonic() - age_s


class TestCircuitBreakerHalfOpen:
    @pytest.mark.parametrize("returncode, action", [(0, "allow"), (1, "block"), (2, "warn")])
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_completed_probe_after_retry_window_closes_breaker(self, mock_cfg, mock_run, returncode, action):
        """Once the retry window has elapsed, one real scan runs; any verdict (allow/block/warn)
        proves the binary healthy and closes the breaker, so the next command is scanned again."""
        mock_cfg.return_value = _CFG
        _open_breaker(age_s=_tirith_mod._CIRCUIT_RETRY_S + 1)
        mock_run.return_value = _mock_run(returncode, _json_stdout())

        result = check_command_security("echo hi")

        assert result["action"] == action
        assert mock_run.call_count == 1
        assert (_tirith_mod._circuit_open, _tirith_mod._crash_count) == (False, 0)
        # Breaker closed: the following command is scanned rather than short-circuited.
        check_command_security("echo again")
        assert mock_run.call_count == 2

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_open_breaker_probes_once_per_window_and_failed_probe_rearms(self, mock_cfg, mock_run):
        """Inside the window nothing spawns; after it exactly one probe runs, and a probe that
        fails re-arms the window so the next caller is fail-open without spawning again."""
        mock_cfg.return_value = _CFG
        _open_breaker(age_s=1)
        mock_run.side_effect = OSError("binary gone")

        assert check_command_security("echo hi")["summary"] == "tirith disabled (circuit breaker)"
        assert mock_run.call_count == 0

        _open_breaker(age_s=_tirith_mod._CIRCUIT_RETRY_S + 1)
        assert check_command_security("echo hi")["action"] == "allow"  # probe spawned and failed
        assert mock_run.call_count == 1
        assert _tirith_mod._circuit_open is True
        assert check_command_security("echo hi")["summary"] == "tirith disabled (circuit breaker)"
        assert mock_run.call_count == 1  # re-armed: no second probe inside the fresh window


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
            assert result == {"action": "allow", "findings": [], "summary": "",
                              "scanner_state": "unavailable"}
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


class TestEmojiVariationSelectorSuppression:
    """VS16 after an emoji-capable base is presentation, not obfuscation: no approval prompt."""

    _VS = [{"rule_id": "variation_selector", "severity": "medium"}]

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_emoji_only_variation_selector_warn_is_downgraded(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(2, _json_stdout(self._VS, "variation selector"))

        # SMP emoji, Dingbats/Misc Symbols, and BMP singletons outside those blocks (ℹ ▶).
        result = check_command_security('ls "🗞️ Journal/" "✅️ Projects/" "ℹ️ Info/" "▶️ Media/"')

        assert result == {"action": "allow", "findings": [], "summary": "", "scanner_state": "ran"}

    @pytest.mark.parametrize("command, findings", [
        ("printf 'a️'", _VS),            # VS16 after a letter
        ("printf '0️'", _VS),            # VS16 after a digit (keycap base)
        ("printf 'x󠄀'", _VS),        # a non-VS16 selector
        ('curl https://bit.ly/x --output "🗞️ Journal/file"',  # emoji path + another finding
         _VS + [{"rule_id": "shortened_url", "severity": "medium"}]),
    ])
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_other_selectors_or_mixed_findings_keep_warn(self, mock_cfg, mock_run, command, findings):
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "variation selector"))

        result = check_command_security(command)

        assert result["action"] == "warn"
        assert result["findings"] == findings


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
# Candidate header probe (INS1-564 / REQ-INS1-SAAS-066 AC-1, AC-2, AC-3, AC-5)
#
# Nothing is built, downloaded or installed here: the acceptance probe reads the
# candidate's own header, so a header is the whole fixture.
# ---------------------------------------------------------------------------

_ELF_MACHINE_X86_64 = 62
_ELF_MACHINE_AARCH64 = 183
_MACHO_CPUTYPE_X86_64 = 0x01000007
_MACHO_CPUTYPE_AARCH64 = 0x0100000C


def _elf_bytes(machine: int, *, length: int = 64) -> bytes:
    """An ELF64 little-endian header declaring ``machine`` at ``e_machine`` (offset 18)."""
    head = bytearray(b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8) + b"\x00\x00")
    head[18:20] = machine.to_bytes(2, "little")
    return bytes(head[:length]).ljust(length, b"\x00")


def _macho_bytes(cputype: int, *, length: int = 32) -> bytes:
    """A 64-bit little-endian Mach-O header — the measured foreign file's first octets, ``cf fa ed fe`` —
    declaring ``cputype`` at offset 4."""
    head = bytearray(b"\xcf\xfa\xed\xfe")
    head += cputype.to_bytes(4, "little") + (2).to_bytes(4, "little") + (0).to_bytes(4, "little")
    head += bytes(16)
    return bytes(head[:length]).ljust(length, b"\x00")


def _native_header() -> bytes:
    """A header declaring *this* process's architecture, in the format this OS loads."""
    target = _tirith_mod._detect_target()
    assert target, "the positive control needs a platform tirith ships a build for"
    arch, platform_slot = target.split("-", 1)
    if platform_slot == "apple-darwin":
        return _macho_bytes(_MACHO_CPUTYPE_AARCH64 if arch == "aarch64" else _MACHO_CPUTYPE_X86_64)
    return _elf_bytes(_ELF_MACHINE_AARCH64 if arch == "aarch64" else _ELF_MACHINE_X86_64)


def _write_candidate(directory, payload: bytes, name: str = "tirith") -> str:
    """A candidate carrying the execute bit at ``directory/name``."""
    path = os.path.join(str(directory), name)
    with open(path, "wb") as handle:
        handle.write(payload)
    os.chmod(path, 0o755)
    return path


def _pin_process(monkeypatch, system: str, machine: str) -> None:
    """Pin the process identity the probe compares the header against — the note measured a Mach-O
    inside linux/arm64. Only the identity is pinned: the header read stays real file I/O."""
    monkeypatch.setattr(_tirith_mod.platform, "system", lambda: system)
    monkeypatch.setattr(_tirith_mod.platform, "machine", lambda: machine)


def _refusal_lines(caplog) -> list:
    return [rec.message for rec in caplog.records if "scanner_unrunnable" in rec.message]


@pytest.fixture
def warned_fresh():
    """The once-per-class channel is process-wide: clear it so a case measures its own line."""
    _tirith_mod._warned_messages.clear()
    yield
    _tirith_mod._warned_messages.clear()


@pytest.fixture
def slots(tmp_path, monkeypatch):
    """Two empty slots — `PATH` and `$HERMES_HOME/bin/tirith` — that a case fills with fixtures, on a
    cold resolved state and a fresh once-per-class channel."""
    _tirith_mod._warned_messages.clear()
    _tirith_mod._resolved_path = None
    home_bin = tmp_path / "home-bin"
    home_bin.mkdir()
    monkeypatch.setattr(_tirith_mod, "_hermes_bin_dir", lambda: str(home_bin))
    monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
    yield home_bin
    _tirith_mod._warned_messages.clear()


class TestCandidateHeaderProbe:
    """A candidate is a scanner only when the file it names is a binary this process can execute:
    `_is_executable()`'s two facts stay necessary and become insufficient on their own."""

    def test_macho_under_a_linux_process_is_refused(self, slots, caplog, monkeypatch):
        """The measured case: a Mach-O at the fallback slot inside linux/arm64, reported as installed."""
        _pin_process(monkeypatch, "Linux", "arm64")
        candidate = _write_candidate(slots, _macho_bytes(_MACHO_CPUTYPE_AARCH64))

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert _tirith_mod._find_local_tirith() is None
            found, may_install = _tirith_mod._resolve_locally("tirith", warn_missing=False)

        assert found is None                        # never the resolved path
        assert may_install is True                  # what is reached instead is the install path
        assert _tirith_mod._cached_path() is None    # and it is never cached
        (line,) = _refusal_lines(caplog)
        assert candidate in line
        assert "mach-o/aarch64" in line and "aarch64-unknown-linux-gnu" in line

    def test_an_elf_for_another_machine_is_refused(self, slots, caplog, monkeypatch):
        """A valid foreign-architecture ELF for a different machine is refused: the check is about
        runnability, not about the file's name."""
        _pin_process(monkeypatch, "Linux", "aarch64")
        candidate = _write_candidate(slots, _elf_bytes(_ELF_MACHINE_X86_64))

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            found, _ = _tirith_mod._resolve_locally("tirith", warn_missing=False)

        assert found is None
        (line,) = _refusal_lines(caplog)
        assert candidate in line
        assert "elf/x86_64" in line and "aarch64-unknown-linux-gnu" in line

    @pytest.mark.parametrize("payload", [
        _elf_bytes(_ELF_MACHINE_AARCH64, length=8),       # an ELF cut before e_machine
        _macho_bytes(_MACHO_CPUTYPE_AARCH64, length=8),   # a Mach-O cut before cputype
    ])
    def test_a_file_shorter_than_its_own_header_is_refused(self, slots, caplog, monkeypatch, payload):
        """A truncated download is a refusal, not a crash."""
        _pin_process(monkeypatch, "Linux", "aarch64")
        candidate = _write_candidate(slots, payload)

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            found, _ = _tirith_mod._resolve_locally("tirith", warn_missing=False)

        assert found is None
        (line,) = _refusal_lines(caplog)
        assert candidate in line
        assert "truncated" in line and "aarch64-unknown-linux-gnu" in line

    def test_a_binary_of_this_architecture_is_accepted(self, slots):
        """Positive control — the check can pass: the resolution returns the path and the install path
        is not entered."""
        candidate = _write_candidate(slots, _native_header())

        with patch("tools.tirith_security._install_tirith") as install:
            assert _tirith_mod._find_local_tirith() == candidate
            found, may_install = _tirith_mod._resolve_locally("tirith", warn_missing=False)

        assert found == candidate and may_install is False
        assert _tirith_mod._cached_path() == candidate
        install.assert_not_called()

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_a_binary_of_this_architecture_scans_and_says_ran(self, mock_cfg, mock_run, slots):
        """The gate is a gate, not a ban: a valid build at a slot is scanned and returns its verdict."""
        candidate = _write_candidate(slots, _native_header())
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(0, _json_stdout(summary="clean"))

        result = check_command_security("echo hi")

        assert result == {"action": "allow", "findings": [], "summary": "clean",
                          "scanner_state": "ran"}
        assert mock_run.call_args[0][0][0] == candidate

    def test_a_candidate_with_no_binary_header_keeps_the_existing_rule(self, slots, caplog, monkeypatch):
        """A file the probe has no format to read (missing, unreadable, not a binary) keeps the module's
        existing state: the new refusal class is not invented over it (Beh 5)."""
        _pin_process(monkeypatch, "Linux", "aarch64")
        candidate = _write_candidate(slots, b"#!/bin/sh\nexit 0\n")

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert _tirith_mod._find_local_tirith() == candidate

        assert _refusal_lines(caplog) == []

    def test_a_big_endian_macho_header_is_refused_too(self, slots, caplog, monkeypatch):
        """The magic carries the file's byte order: `feedfacf` is read big-endian, not as this host's."""
        _pin_process(monkeypatch, "Linux", "aarch64")
        candidate = _write_candidate(
            slots, b"\xfe\xed\xfa\xcf" + _MACHO_CPUTYPE_AARCH64.to_bytes(4, "big") + bytes(24))

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            found, _ = _tirith_mod._resolve_locally("tirith", warn_missing=False)

        assert found is None
        (line,) = _refusal_lines(caplog)
        assert candidate in line and "mach-o/aarch64" in line

    def test_the_header_is_read_in_the_byte_order_the_file_declares(self):
        """`e_machine` follows `EI_DATA` and a Mach-O's `cputype` its magic — never this host's order."""
        big_endian_elf = (b"\x7fELF" + bytes([2, 2, 1, 0]) + bytes(8) + b"\x00\x00"
                          + _ELF_MACHINE_AARCH64.to_bytes(2, "big") + bytes(44))
        big_endian_macho = (b"\xfe\xed\xfa\xcf" + _MACHO_CPUTYPE_AARCH64.to_bytes(4, "big")
                            + bytes(24))

        assert _tirith_mod._read_elf_header(big_endian_elf)["declared"] == "elf/aarch64"
        assert _tirith_mod._read_macho_header(big_endian_macho, True, True)["declared"] == "mach-o/aarch64"

    def test_a_foreign_explicit_path_is_refused_as_not_found(self, tmp_path, monkeypatch):
        """The authoritative explicit path is a candidate too: a binary this process cannot run is never
        the resolved scanner."""
        _pin_process(monkeypatch, "Linux", "aarch64")
        candidate = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="custom-tirith")
        _tirith_mod._resolved_path = None
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)

        assert _tirith_mod._resolve_locally(candidate, warn_missing=False) == (None, False)
        assert _tirith_mod._cached_path() is None


class TestRefusedSlotFallsThrough:
    """A refused candidate is *not found*: the order the resolution already has is unchanged (Beh 2)."""

    def test_a_refused_path_hit_lets_the_next_slot_answer(self, tmp_path, monkeypatch):
        _pin_process(monkeypatch, "Linux", "aarch64")
        on_path = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="path-tirith")
        home_bin = tmp_path / "home-bin"
        home_bin.mkdir()
        valid = _write_candidate(home_bin, _elf_bytes(_ELF_MACHINE_AARCH64))
        monkeypatch.setattr(_tirith_mod, "_hermes_bin_dir", lambda: str(home_bin))
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: on_path)

        assert _tirith_mod._find_local_tirith() == valid

    def test_both_slots_refused_reach_the_install_path(self, tmp_path, monkeypatch, warned_fresh):
        _pin_process(monkeypatch, "Linux", "aarch64")
        home_bin = tmp_path / "home-bin"
        home_bin.mkdir()
        _write_candidate(home_bin, _macho_bytes(_MACHO_CPUTYPE_AARCH64))
        monkeypatch.setattr(_tirith_mod, "_hermes_bin_dir", lambda: str(home_bin))
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security._install_tirith",
                   return_value=("/auto/tirith", "")) as install:
            resolved = _tirith_mod._resolve_tirith_path("tirith")

        assert resolved == "/auto/tirith"
        install.assert_called_once()
        assert _tirith_mod._cached_path() == "/auto/tirith"
        del warned_fresh  # the fixture only orders the once-per-class reset


class TestCachedPathValidation:
    """A cached path is re-validated on the same rule (AC-3): replaced or truncated, it is not found
    instead of being handed back for the rest of the process."""

    @pytest.mark.parametrize("payload", [
        _macho_bytes(_MACHO_CPUTYPE_AARCH64),            # the pair is refilled with a foreign binary
        _elf_bytes(_ELF_MACHINE_AARCH64, length=8),      # the file is truncated in place
    ])
    def test_a_cached_path_that_stopped_passing_is_not_returned(self, tmp_path, monkeypatch,
                                                                warned_fresh, payload):
        _pin_process(monkeypatch, "Linux", "aarch64")
        cached = _write_candidate(tmp_path, _elf_bytes(_ELF_MACHINE_AARCH64))
        _tirith_mod._resolved_path = cached
        home_bin = tmp_path / "home-bin"
        home_bin.mkdir()
        monkeypatch.setattr(_tirith_mod, "_hermes_bin_dir", lambda: str(home_bin))
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
        with open(cached, "wb") as handle:
            handle.write(payload)
        os.chmod(cached, 0o755)

        with patch("tools.tirith_security._install_tirith",
                   return_value=("/auto/tirith", "")) as install:
            resolved = _tirith_mod._resolve_tirith_path("tirith")

        assert resolved == "/auto/tirith"          # the refused path is not what answers
        assert _tirith_mod._cached_path() == "/auto/tirith"
        install.assert_called_once()
        del warned_fresh

    def test_ensure_installed_drops_a_cached_path_that_no_longer_passes(self, tmp_path, monkeypatch):
        """The boot path treats it as not found too, and asks the slots again."""
        _pin_process(monkeypatch, "Linux", "aarch64")
        cached = _write_candidate(tmp_path, _elf_bytes(_ELF_MACHINE_AARCH64))
        _tirith_mod._resolved_path = cached
        with open(cached, "wb") as handle:
            handle.write(_macho_bytes(_MACHO_CPUTYPE_AARCH64))
        os.chmod(cached, 0o755)
        home_bin = tmp_path / "home-bin"
        home_bin.mkdir()
        monkeypatch.setattr(_tirith_mod, "_hermes_bin_dir", lambda: str(home_bin))
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)

        with patch("tools.tirith_security._load_security_config", return_value=_CFG), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            MockThread.return_value = mock_thread
            assert ensure_installed(log_failures=False) is None

        MockThread.assert_called_once()
        assert _tirith_mod._cached_path() is None


class TestRefusalIsTypedOnTheWayIn:
    """The refusal is produced where the resolution refuses, before any command is guarded (Beh 3)."""

    def test_the_boot_call_with_log_failures_false_still_names_the_refusal(self, slots, caplog, monkeypatch):
        """`unsupported_platform` keeps its silence; this class may not be swallowed by the boot call."""
        _pin_process(monkeypatch, "Linux", "arm64")
        candidate = _write_candidate(slots, _macho_bytes(_MACHO_CPUTYPE_AARCH64))

        with patch("tools.tirith_security._load_security_config", return_value=_CFG), \
             patch("tools.tirith_security.threading.Thread") as MockThread, \
             caplog.at_level("WARNING", logger="tools.tirith_security"):
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            MockThread.return_value = mock_thread
            assert ensure_installed(log_failures=False) is None

        (line,) = _refusal_lines(caplog)
        assert candidate in line
        assert "mach-o/aarch64" in line and "aarch64-unknown-linux-gnu" in line
        MockThread.assert_called_once()  # the install path is reached instead of the refused file

    def test_the_refusal_is_named_once_per_refused_path(self, slots, caplog, monkeypatch):
        """A slot a mount keeps refilling with the same foreign file does not repeat the line."""
        _pin_process(monkeypatch, "Linux", "aarch64")
        _write_candidate(slots, _macho_bytes(_MACHO_CPUTYPE_AARCH64))

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            for _ in range(5):
                _tirith_mod._find_local_tirith()

        assert len(_refusal_lines(caplog)) == 1

    def test_a_platform_with_no_build_stays_silent(self, slots, caplog, monkeypatch):
        """Nothing of this class is emitted where tirith has no build at all (AC-6)."""
        monkeypatch.setattr(_tirith_mod.platform, "system", lambda: "Windows")
        _write_candidate(slots, _macho_bytes(_MACHO_CPUTYPE_AARCH64))

        with patch("tools.tirith_security._load_security_config", return_value=_CFG), \
             caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert ensure_installed(log_failures=False) is None

        assert _refusal_lines(caplog) == []


# ---------------------------------------------------------------------------
# The guard never fails open in silence (INS1-565 / REQ-INS1-SAAS-066 AC-4)
# ---------------------------------------------------------------------------


class TestScannerStateMarker:
    """Every verdict says whether a scan produced it — `ran`, or why no scan ran."""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_the_breaker_allow_keeps_its_summary_verbatim_and_carries_disabled(self, mock_cfg,
                                                                              mock_run, warned_fresh):
        mock_cfg.return_value = _CFG
        _open_breaker(age_s=1)

        first = check_command_security("echo hi")
        second = check_command_security("echo hi")

        assert first["summary"] == "tirith disabled (circuit breaker)"
        assert first["scanner_state"] == "disabled"
        assert second["scanner_state"] == "disabled"  # present on the first call and it stays
        mock_run.assert_not_called()
        del warned_fresh

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_an_allow_without_a_scan_is_distinguishable_by_the_value_alone(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(0, _json_stdout(summary="clean"))
        scanned = check_command_security("echo hi")

        with patch("tools.tirith_security._resolve_tirith_path", return_value=None):
            unscanned = check_command_security("echo hi")

        assert scanned["scanner_state"] == "ran"
        assert unscanned["action"] == "allow" and unscanned["scanner_state"] == "unavailable"
        assert unscanned != scanned  # no prose needs to be interpreted

    @pytest.mark.parametrize("returncode, action", [(0, "allow"), (1, "block"), (2, "warn")])
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_a_completed_scan_is_unchanged_apart_from_the_marker(self, mock_cfg, mock_run,
                                                                 returncode, action):
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(returncode, _json_stdout(summary="scan summary"))

        result = check_command_security("echo hi")

        assert result["action"] == action
        assert result["summary"] == "scan summary"
        assert result["findings"] == []
        assert result["scanner_state"] == "ran"
        assert set(result) == {"action", "findings", "summary", "scanner_state"}

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_a_spawn_failure_carries_unavailable(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        mock_run.side_effect = OSError(8, "Exec format error")

        assert check_command_security("echo hi")["scanner_state"] == "unavailable"

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_fail_closed_still_blocks_and_says_why(self, mock_cfg, mock_run):
        """The policy still decides allow versus block; the marker only says no scan ran."""
        mock_cfg.return_value = {**_CFG, "tirith_fail_open": False}
        mock_run.side_effect = OSError(8, "Exec format error")

        result = check_command_security("echo hi")

        assert result["action"] == "block"
        assert result["scanner_state"] == "unavailable"

    @patch("tools.tirith_security._load_security_config")
    def test_the_configured_off_switch_carries_disabled(self, mock_cfg):
        mock_cfg.return_value = {**_CFG, "tirith_enabled": False}

        assert check_command_security("rm -rf /") == {"action": "allow", "findings": [],
                                                      "summary": "", "scanner_state": "disabled"}


class TestTurnLogNamesTheScannerState:
    """A state observable only by reading a `summary` string is a FAIL of AC-4."""

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_the_unavailable_state_is_named_once(self, mock_cfg, mock_run, caplog, warned_fresh):
        mock_cfg.return_value = _CFG
        mock_run.side_effect = FileNotFoundError("[Errno 2] No such file: '/opt/tirith'")

        with patch("tools.tirith_security._resolve_tirith_path", return_value="/opt/tirith"), \
             caplog.at_level("WARNING", logger="tools.tirith_security"):
            for _ in range(_tirith_mod._CRASH_LIMIT):
                assert check_command_security("echo hi")["scanner_state"] == "unavailable"

        named = [rec.message for rec in caplog.records if "scanner_state=unavailable" in rec.message]
        assert len(named) == 1
        del warned_fresh

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_entry_into_and_exit_from_disabled_are_named(self, mock_cfg, mock_run, caplog, warned_fresh):
        mock_cfg.return_value = _CFG
        mock_run.side_effect = OSError(8, "Exec format error")

        with patch("tools.tirith_security._resolve_tirith_path", return_value="/opt/tirith"), \
             caplog.at_level(logging.INFO, logger="tools.tirith_security"):
            for _ in range(_tirith_mod._CRASH_LIMIT):
                check_command_security("echo hi")
            disabled = check_command_security("echo hi")
            # the retry window has elapsed: the half-open probe runs for real and a completed scan closes
            _tirith_mod._circuit_open_at = time.monotonic() - _tirith_mod._CIRCUIT_RETRY_S - 1
            mock_run.side_effect = None
            mock_run.return_value = _mock_run(0, _json_stdout())
            recovered = check_command_security("echo hi")

        assert disabled["scanner_state"] == "disabled"
        assert recovered["scanner_state"] == "ran"
        text = "\n".join(rec.message for rec in caplog.records)
        assert "circuit breaker opened after 3 consecutive failures" in text
        assert "scanner_state=disabled" in text
        assert "circuit breaker half-open: probing after" in text
        assert "circuit breaker closed after successful probe" in text
        del warned_fresh

    def test_the_refusal_and_the_state_ride_the_same_turn_log(self, tmp_path, monkeypatch,
                                                              caplog, warned_fresh):
        """Nobody needs an `Exec format error` to learn either: a refused candidate is named on the way
        in, and the verdict that follows says no scan ran."""
        _pin_process(monkeypatch, "Linux", "arm64")
        home_bin = tmp_path / "home-bin"
        home_bin.mkdir()
        candidate = _write_candidate(home_bin, _macho_bytes(_MACHO_CPUTYPE_AARCH64))
        monkeypatch.setattr(_tirith_mod, "_hermes_bin_dir", lambda: str(home_bin))
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security._load_security_config", return_value=_CFG), \
             patch("tools.tirith_security._install_tirith", return_value=(None, "download_failed")), \
             patch("tools.tirith_security._mark_install_failed"), \
             patch("tools.tirith_security.subprocess.run") as mock_run, \
             caplog.at_level("WARNING", logger="tools.tirith_security"):
            mock_run.side_effect = OSError(8, "Exec format error")
            result = check_command_security("ls -la /tmp")

        assert result["scanner_state"] == "unavailable"
        text = "\n".join(rec.message for rec in caplog.records)
        assert "scanner_unrunnable" in text and candidate in text
        assert "scanner_state=unavailable" in text
        del warned_fresh

