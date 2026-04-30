"""Tests for the tirith security scanning subprocess wrapper."""

import json
import os
import subprocess
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
    yield
    _tirith_mod._resolved_path = None
    _tirith_mod._install_thread = None
    _tirith_mod._install_failure_reason = ""


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
    def test_exit_2_invalid_json_still_warns(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(2, "{broken")
        result = check_command_security("suspicious command")
        assert result["action"] == "warn"
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
    def test_permission_error_fail_open(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = PermissionError("Permission denied")
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
    def test_timeout_fail_open(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tirith", timeout=5)
        result = check_command_security("slow command")
        assert result["action"] == "allow"
        assert "timed out" in result["summary"]

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
    def test_unknown_exit_code_fail_open(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(99, "")
        result = check_command_security("cmd")
        assert result["action"] == "allow"
        assert "exit code 99" in result["summary"]

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
# Disabled + path expansion
# ---------------------------------------------------------------------------

class TestDisabled:
    @patch("tools.tirith_security._load_security_config")
    def test_disabled_returns_allow(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": False, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        result = check_command_security("rm -rf /")
        assert result["action"] == "allow"


class TestPathExpansion:
    def test_tilde_expanded_in_resolve(self):
        """_resolve_tirith_path should expand ~ in configured path."""
        from tools.tirith_security import _resolve_tirith_path
        _tirith_mod._resolved_path = None
        # Explicit path — won't auto-download, just expands and caches miss
        result = _resolve_tirith_path("~/bin/tirith")
        assert "~" not in result, "tilde should be expanded"
        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Findings cap + summary cap
# ---------------------------------------------------------------------------

class TestCaps:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_findings_capped_at_50(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        findings = [{"rule_id": f"rule_{i}"} for i in range(100)]
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "many findings"))
        result = check_command_security("cmd")
        assert len(result["findings"]) == 50

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_summary_capped_at_500(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        long_summary = "x" * 1000
        mock_run.return_value = _mock_run(2, _json_stdout([], long_summary))
        result = check_command_security("cmd")
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

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_type_error_propagates(self, mock_cfg, mock_run):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = TypeError("unexpected bug")
        with pytest.raises(TypeError):
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

    @patch("tools.tirith_security._load_security_config")
    def test_not_found_returns_none(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=False), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread
            result = ensure_installed()
            assert result is None
            # Should have launched background thread
            mock_thread.start.assert_called_once()
        _tirith_mod._resolved_path = None

    @patch("tools.tirith_security._load_security_config")
    def test_startup_prefetch_can_suppress_install_failure_logs(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _tirith_mod._resolved_path = None
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=False), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread
            result = ensure_installed(log_failures=False)
            assert result is None
            assert MockThread.call_args.kwargs["kwargs"] == {"log_failures": False}
            mock_thread.start.assert_called_once()
        _tirith_mod._resolved_path = None


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

    @patch("tools.tirith_security._mark_install_failed")
    @patch("tools.tirith_security._is_install_failed_on_disk", return_value=False)
    @patch("tools.tirith_security._install_tirith", return_value=(None, "download_failed"))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_failed_install_scan_uses_fail_open(self, mock_cfg, mock_run,
                                                 mock_which, mock_install,
                                                 mock_disk_check, mock_mark):
        """After cached miss, check_command_security hits OSError → fail_open."""
        _tirith_mod._resolved_path = None
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.side_effect = FileNotFoundError("No such file: tirith")
        # First command triggers install attempt + cached miss + scan
        result = check_command_security("echo hello")
        assert result["action"] == "allow"
        assert mock_install.call_count == 1

        # Second command: no install retry, just hits OSError → allow
        result = check_command_security("echo world")
        assert result["action"] == "allow"
        assert mock_install.call_count == 1  # still 1

        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Explicit path must not auto-download (Finding #2)
# ---------------------------------------------------------------------------

class TestExplicitPathNoAutoDownload:
    @patch("tools.tirith_security._install_tirith")
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_explicit_path_missing_no_download(self, mock_which, mock_install):
        """An explicit tirith_path that doesn't exist must NOT trigger download."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = None

        result = _resolve_tirith_path("/opt/custom/tirith")
        # Should cache failure, not call _install_tirith
        mock_install.assert_not_called()
        assert _tirith_mod._resolved_path is _INSTALL_FAILED
        assert "/opt/custom/tirith" in result

        _tirith_mod._resolved_path = None

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
    def test_cosign_pass(self, mock_which, mock_run):
        """cosign verify-blob exits 0 → returns True."""
        from tools.tirith_security import _verify_cosign
        mock_run.return_value = _mock_run(0, "Verified OK")
        result = _verify_cosign("/tmp/checksums.txt", "/tmp/checksums.txt.sig",
                                "/tmp/checksums.txt.pem")
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "verify-blob" in args
        assert "--certificate-identity-regexp" in args

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

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    def test_cosign_fail_aborts(self, mock_which, mock_run):
        """cosign verify-blob exits non-zero → returns False (abort install)."""
        from tools.tirith_security import _verify_cosign
        mock_run.return_value = _mock_run(1, "", "signature mismatch")
        result = _verify_cosign("/tmp/checksums.txt", "/tmp/checksums.txt.sig",
                                "/tmp/checksums.txt.pem")
        assert result is False

    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_cosign_not_found_returns_none(self, mock_which):
        """cosign not on PATH → returns None (proceed with SHA-256 only)."""
        from tools.tirith_security import _verify_cosign
        result = _verify_cosign("/tmp/checksums.txt", "/tmp/checksums.txt.sig",
                                "/tmp/checksums.txt.pem")
        assert result is None

    @patch("tools.tirith_security.subprocess.run",
           side_effect=subprocess.TimeoutExpired("cosign", 15))
    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    def test_cosign_timeout_returns_none(self, mock_which, mock_run):
        """cosign times out → returns None (proceed with SHA-256 only)."""
        from tools.tirith_security import _verify_cosign
        result = _verify_cosign("/tmp/checksums.txt", "/tmp/checksums.txt.sig",
                                "/tmp/checksums.txt.pem")
        assert result is None

    @patch("tools.tirith_security.subprocess.run",
           side_effect=OSError("exec format error"))
    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    def test_cosign_os_error_returns_none(self, mock_which, mock_run):
        """cosign OSError → returns None (proceed with SHA-256 only)."""
        from tools.tirith_security import _verify_cosign
        result = _verify_cosign("/tmp/checksums.txt", "/tmp/checksums.txt.sig",
                                "/tmp/checksums.txt.pem")
        assert result is None

    @patch("tools.tirith_security._verify_cosign", return_value=False)
    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/cosign")
    @patch("tools.tirith_security._download_file")
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_aborts_on_cosign_rejection(self, mock_target, mock_dl,
                                                 mock_which, mock_cosign):
        """_install_tirith returns None when cosign rejects the signature."""
        from tools.tirith_security import _install_tirith
        path, reason = _install_tirith()
        assert path is None
        assert reason == "cosign_verification_failed"

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

    @patch("tools.tirith_security.tarfile.open")
    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security._verify_cosign", return_value=None)
    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/cosign")
    @patch("tools.tirith_security._download_file")
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_proceeds_when_cosign_exec_fails(self, mock_target, mock_dl,
                                                       mock_which, mock_cosign,
                                                       mock_checksum, mock_tarfile):
        """_install_tirith falls back to SHA-256 when cosign exists but fails to execute."""
        from tools.tirith_security import _install_tirith
        mock_tar = MagicMock()
        mock_tar.__enter__ = MagicMock(return_value=mock_tar)
        mock_tar.__exit__ = MagicMock(return_value=False)
        mock_tar.getmembers.return_value = []
        mock_tarfile.return_value = mock_tar

        path, reason = _install_tirith()
        assert path is None
        assert reason == "binary_not_in_archive"  # got past cosign
        assert mock_checksum.called

    @patch("tools.tirith_security.tarfile.open")
    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/cosign")
    @patch("tools.tirith_security._download_file")
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_proceeds_when_cosign_artifacts_missing(self, mock_target,
                                                              mock_dl, mock_which,
                                                              mock_checksum, mock_tarfile):
        """_install_tirith proceeds with SHA-256 when .sig/.pem downloads fail."""
        from tools.tirith_security import _install_tirith
        import urllib.request

        def _dl_side_effect(url, dest, timeout=10):
            if url.endswith(".sig") or url.endswith(".pem"):
                raise urllib.request.URLError("404 Not Found")

        mock_dl.side_effect = _dl_side_effect
        mock_tar = MagicMock()
        mock_tar.__enter__ = MagicMock(return_value=mock_tar)
        mock_tar.__exit__ = MagicMock(return_value=False)
        mock_tar.getmembers.return_value = []
        mock_tarfile.return_value = mock_tar

        path, reason = _install_tirith()
        assert path is None
        assert reason == "binary_not_in_archive"  # got past cosign
        assert mock_checksum.called

    @patch("tools.tirith_security.tarfile.open")
    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security._verify_cosign", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/cosign")
    @patch("tools.tirith_security._download_file")
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_proceeds_when_cosign_passes(self, mock_target, mock_dl,
                                                   mock_which, mock_cosign,
                                                   mock_checksum, mock_tarfile):
        """_install_tirith proceeds only when cosign explicitly passes (True)."""
        from tools.tirith_security import _install_tirith
        # Mock tarfile — empty archive means "binary not found" return
        mock_tar = MagicMock()
        mock_tar.__enter__ = MagicMock(return_value=mock_tar)
        mock_tar.__exit__ = MagicMock(return_value=False)
        mock_tar.getmembers.return_value = []
        mock_tarfile.return_value = mock_tar

        path, reason = _install_tirith()
        assert path is None  # no binary in mock archive, but got past cosign
        assert reason == "binary_not_in_archive"
        assert mock_checksum.called  # reached SHA-256 step
        assert mock_cosign.called  # cosign was invoked


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

    def test_ensure_installed_skips_on_disk_marker(self):
        """ensure_installed skips network attempt when disk marker exists."""
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security._load_security_config",
                   return_value={"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._read_failure_reason", return_value="download_failed"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=True):

            result = ensure_installed()
            assert result is None
            assert _tirith_mod._resolved_path is _tirith_mod._INSTALL_FAILED
            assert _tirith_mod._install_failure_reason == "download_failed"

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

    def test_resolve_picks_up_background_result(self):
        """After background thread finishes, _resolve_tirith_path uses cached path."""
        from tools.tirith_security import _resolve_tirith_path
        # Simulate background thread having completed and set the path
        _tirith_mod._resolved_path = "/usr/local/bin/tirith"

        result = _resolve_tirith_path("tirith")
        assert result == "/usr/local/bin/tirith"

        _tirith_mod._resolved_path = None


# ---------------------------------------------------------------------------
# Disk failure marker persistence (P2)
# ---------------------------------------------------------------------------

class TestDiskFailureMarker:
    def test_mark_and_check(self):
        """Writing then reading the marker should work."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        marker = os.path.join(tmpdir, ".tirith-install-failed")
        with patch("tools.tirith_security._failure_marker_path", return_value=marker):
            from tools.tirith_security import (
                _mark_install_failed, _is_install_failed_on_disk, _clear_install_failed,
            )
            assert not _is_install_failed_on_disk()
            _mark_install_failed("download_failed")
            assert _is_install_failed_on_disk()
            _clear_install_failed()
            assert not _is_install_failed_on_disk()

    def test_expired_marker_ignored(self):
        """Marker older than TTL should be ignored."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        marker = os.path.join(tmpdir, ".tirith-install-failed")
        with patch("tools.tirith_security._failure_marker_path", return_value=marker):
            from tools.tirith_security import _mark_install_failed, _is_install_failed_on_disk
            _mark_install_failed("download_failed")
            # Backdate the file past 24h TTL
            old_time = time.time() - 90000  # 25 hours ago
            os.utime(marker, (old_time, old_time))
            assert not _is_install_failed_on_disk()

    def test_cosign_missing_marker_clears_when_cosign_appears(self):
        """Marker with 'cosign_missing' reason clears if cosign is now on PATH."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        marker = os.path.join(tmpdir, ".tirith-install-failed")
        with patch("tools.tirith_security._failure_marker_path", return_value=marker):
            from tools.tirith_security import _mark_install_failed, _is_install_failed_on_disk
            _mark_install_failed("cosign_missing")
            assert _is_install_failed_on_disk()  # cosign still absent

            # Now cosign appears on PATH
            with patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/cosign"):
                assert not _is_install_failed_on_disk()
            # Marker file should have been removed
            assert not os.path.exists(marker)

    def test_cosign_missing_marker_stays_when_cosign_still_absent(self):
        """Marker with 'cosign_missing' reason stays if cosign is still missing."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        marker = os.path.join(tmpdir, ".tirith-install-failed")
        with patch("tools.tirith_security._failure_marker_path", return_value=marker):
            from tools.tirith_security import _mark_install_failed, _is_install_failed_on_disk
            _mark_install_failed("cosign_missing")
            with patch("tools.tirith_security.shutil.which", return_value=None):
                assert _is_install_failed_on_disk()

    def test_non_cosign_marker_not_affected_by_cosign_presence(self):
        """Markers with other reasons are NOT cleared by cosign appearing."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        marker = os.path.join(tmpdir, ".tirith-install-failed")
        with patch("tools.tirith_security._failure_marker_path", return_value=marker):
            from tools.tirith_security import _mark_install_failed, _is_install_failed_on_disk
            _mark_install_failed("download_failed")
            with patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/cosign"):
                assert _is_install_failed_on_disk()  # still failed

    @patch("tools.tirith_security._mark_install_failed")
    @patch("tools.tirith_security._is_install_failed_on_disk", return_value=False)
    @patch("tools.tirith_security._install_tirith", return_value=(None, "cosign_missing"))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_sync_resolve_persists_failure(self, mock_which, mock_install,
                                            mock_disk_check, mock_mark):
        """Synchronous _resolve_tirith_path persists failure to disk."""
        from tools.tirith_security import _resolve_tirith_path
        _tirith_mod._resolved_path = None

        _resolve_tirith_path("tirith")
        mock_mark.assert_called_once_with("cosign_missing")

        _tirith_mod._resolved_path = None

    @patch("tools.tirith_security._clear_install_failed")
    @patch("tools.tirith_security._is_install_failed_on_disk", return_value=False)
    @patch("tools.tirith_security._install_tirith", return_value=("/installed/tirith", ""))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_sync_resolve_clears_marker_on_success(self, mock_which, mock_install,
                                                    mock_disk_check, mock_clear):
        """Successful install clears the disk failure marker."""
        from tools.tirith_security import _resolve_tirith_path
        _tirith_mod._resolved_path = None

        result = _resolve_tirith_path("tirith")
        assert result == "/installed/tirith"
        mock_clear.assert_called_once()

        _tirith_mod._resolved_path = None

    def test_sync_resolve_skips_install_on_disk_marker(self):
        """_resolve_tirith_path skips download when disk marker is recent."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = None

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._read_failure_reason", return_value="download_failed"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=True), \
             patch("tools.tirith_security._install_tirith") as mock_install:
            _resolve_tirith_path("tirith")
            mock_install.assert_not_called()
            assert _tirith_mod._resolved_path is _INSTALL_FAILED
            assert _tirith_mod._install_failure_reason == "download_failed"

        _tirith_mod._resolved_path = None

    def test_install_failed_still_checks_local_paths(self):
        """After _INSTALL_FAILED, a manual install on PATH is picked up."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = _INSTALL_FAILED

        with patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/tirith"), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear:
            result = _resolve_tirith_path("tirith")
            assert result == "/usr/local/bin/tirith"
            assert _tirith_mod._resolved_path == "/usr/local/bin/tirith"
            mock_clear.assert_called_once()

        _tirith_mod._resolved_path = None

    def test_install_failed_recovers_from_hermes_bin(self):
        """After _INSTALL_FAILED, manual install in HERMES_HOME/bin is picked up."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        import tempfile
        tmpdir = tempfile.mkdtemp()
        hermes_bin = os.path.join(tmpdir, "tirith")
        # Create a fake executable
        with open(hermes_bin, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(hermes_bin, 0o755)

        _tirith_mod._resolved_path = _INSTALL_FAILED

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value=tmpdir), \
             patch("tools.tirith_security._clear_install_failed") as mock_clear:
            result = _resolve_tirith_path("tirith")
            assert result == hermes_bin
            assert _tirith_mod._resolved_path == hermes_bin
            mock_clear.assert_called_once()

        _tirith_mod._resolved_path = None

    def test_install_failed_skips_network_when_local_absent(self):
        """After _INSTALL_FAILED, if local checks fail, network is NOT retried."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = _INSTALL_FAILED

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._install_tirith") as mock_install:
            result = _resolve_tirith_path("tirith")
            assert result == "tirith"  # fallback to configured path
            mock_install.assert_not_called()

        _tirith_mod._resolved_path = None

    def test_cosign_missing_disk_marker_allows_retry(self):
        """Disk marker with cosign_missing reason allows retry when cosign appears."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = None

        # _is_install_failed_on_disk sees "cosign_missing" + cosign on PATH → returns False
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=False), \
             patch("tools.tirith_security._install_tirith", return_value=("/new/tirith", "")) as mock_install, \
             patch("tools.tirith_security._clear_install_failed"):
            result = _resolve_tirith_path("tirith")
            mock_install.assert_called_once()  # network retry happened
            assert result == "/new/tirith"

        _tirith_mod._resolved_path = None

    def test_in_memory_cosign_missing_retries_when_cosign_appears(self):
        """In-memory _INSTALL_FAILED with cosign_missing retries when cosign appears."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = _INSTALL_FAILED
        _tirith_mod._install_failure_reason = "cosign_missing"

        def _which_side_effect(name):
            if name == "tirith":
                return None  # tirith not on PATH
            if name == "cosign":
                return "/usr/local/bin/cosign"  # cosign now available
            return None

        with patch("tools.tirith_security.shutil.which", side_effect=_which_side_effect), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=False), \
             patch("tools.tirith_security._install_tirith", return_value=("/new/tirith", "")) as mock_install, \
             patch("tools.tirith_security._clear_install_failed"):
            result = _resolve_tirith_path("tirith")
            mock_install.assert_called_once()  # network retry happened
            assert result == "/new/tirith"

        _tirith_mod._resolved_path = None

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

    def test_in_memory_cosign_missing_stays_when_cosign_still_absent(self):
        """In-memory cosign_missing is NOT retried when cosign is still absent."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = _INSTALL_FAILED
        _tirith_mod._install_failure_reason = "cosign_missing"

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._install_tirith") as mock_install:
            result = _resolve_tirith_path("tirith")
            assert result == "tirith"  # fallback
            mock_install.assert_not_called()

        _tirith_mod._resolved_path = None

    def test_disk_marker_reason_preserved_in_memory(self):
        """Disk marker reason is loaded into _install_failure_reason, not a generic tag."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _tirith_mod._resolved_path = None

        # First call: disk marker with cosign_missing is active, cosign still absent
        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._read_failure_reason", return_value="cosign_missing"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=True):
            _resolve_tirith_path("tirith")
            assert _tirith_mod._resolved_path is _INSTALL_FAILED
            assert _tirith_mod._install_failure_reason == "cosign_missing"

        # Second call: cosign now on PATH → in-memory retry fires
        def _which_side_effect(name):
            if name == "tirith":
                return None
            if name == "cosign":
                return "/usr/local/bin/cosign"
            return None

        with patch("tools.tirith_security.shutil.which", side_effect=_which_side_effect), \
             patch("tools.tirith_security._hermes_bin_dir", return_value="/nonexistent"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=False), \
             patch("tools.tirith_security._install_tirith", return_value=("/new/tirith", "")) as mock_install, \
             patch("tools.tirith_security._clear_install_failed"):
            result = _resolve_tirith_path("tirith")
            mock_install.assert_called_once()
            assert result == "/new/tirith"

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

    def test_failure_marker_respects_hermes_home(self):
        """_failure_marker_path must use HERMES_HOME, not hardcoded ~/.hermes."""
        from tools.tirith_security import _failure_marker_path
        with patch.dict(os.environ, {"HERMES_HOME": "/custom/hermes"}):
            result = _failure_marker_path()
        assert result == "/custom/hermes/.tirith-install-failed"

    def test_conftest_isolation_prevents_real_home_writes(self):
        """The conftest autouse fixture sets HERMES_HOME; verify it's active."""
        hermes_home = os.getenv("HERMES_HOME")
        assert hermes_home is not None, "HERMES_HOME should be set by conftest"
        assert "hermes_test" in hermes_home, "Should point to test temp dir"

    def test_get_hermes_home_fallback(self):
        """Without HERMES_HOME set, falls back to the active OS home."""
        from tools.tirith_security import _get_hermes_home
        with patch.dict(os.environ, {}, clear=True):
            # Remove HERMES_HOME entirely. With HOME also absent, expanduser
            # falls back to the account database; compute expected under the
            # same environment instead of after patch.dict restores HOME.
            os.environ.pop("HERMES_HOME", None)
            expected = os.path.join(os.path.expanduser("~"), ".hermes")
            result = _get_hermes_home()
        assert result == expected
