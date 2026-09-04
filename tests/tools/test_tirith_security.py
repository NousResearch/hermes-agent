"""Tests for the tirith security scanning subprocess wrapper."""

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timedelta, timezone
from http.client import HTTPMessage
from unittest.mock import MagicMock, patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.x509.oid import NameOID

from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import lazy_deps as _lazy_deps
import tools.tirith_security as _tirith_mod
from tools.tirith_security import check_command_security, ensure_installed


_REAL_ALLOW_LAZY_INSTALLS = _lazy_deps._allow_lazy_installs


def _state(path: str = "tirith"):
    return _tirith_mod._runtime_state(path)


@pytest.fixture(autouse=True)
def _reset_resolved_path():
    """Pre-set cached path to skip auto-install in scan tests.
    Tests that specifically test ensure_installed / resolve behavior
    reset this to None themselves.
    """
    _tirith_mod._reset_runtime_states_for_tests()
    _state().resolved_path = "tirith"
    with _tirith_mod._in_process_update_state_lock:
        _tirith_mod._in_process_update_states.clear()
    # The global test fixture disables runtime installs. Most tests in this
    # module exercise Tirith's installer mechanics, so opt them in explicitly;
    # policy tests below override this nested patch with False.
    with patch("tools.lazy_deps._allow_lazy_installs", return_value=True):
        yield
    _tirith_mod._reset_runtime_states_for_tests()
    with _tirith_mod._in_process_update_state_lock:
        _tirith_mod._in_process_update_states.clear()


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


def _write_release_certificate(path, identities):
    """Write a minimal certificate with cosign-like URI SAN identities."""
    key = ed25519.Ed25519PrivateKey.generate()
    now = datetime.now(timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(minutes=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.UniformResourceIdentifier(identity) for identity in identities]
            ),
            critical=False,
        )
        .sign(key, algorithm=None)
    )
    path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def _mock_missing_tirith(monkeypatch):
    monkeypatch.setattr(_tirith_mod, "is_platform_supported", lambda: True)
    monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_tirith_mod.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(_tirith_mod, "_read_failure_reason", lambda: None)
    monkeypatch.setattr(_tirith_mod, "_clear_install_failed", lambda *_args: None)


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
        assert _state().resolved_path is None

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
        _state().resolved_path = None
        assert ensure_installed() is None

    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/tirith")
    @patch("tools.tirith_security._load_security_config")
    def test_found_on_path_returns_immediately(self, mock_cfg, mock_which):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _state().resolved_path = None
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = ensure_installed()
        assert result == "/usr/local/bin/tirith"
        _state().resolved_path = None

    def test_config_opt_out_prevents_tirith_download_thread(
        self, tmp_path, monkeypatch
    ):
        """Exercise the real config loader, not only a mocked policy helper."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "security:\n  allow_lazy_installs: false\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
        monkeypatch.delenv("HERMES_LAZY_INSTALL_TARGET", raising=False)
        _state().resolved_path = None
        _state().install_thread = None

        cfg = {
            "tirith_enabled": True,
            "tirith_path": "tirith",
            "tirith_timeout": 5,
            "tirith_fail_open": True,
        }
        monkeypatch.setattr(
            _lazy_deps, "_allow_lazy_installs", _REAL_ALLOW_LAZY_INSTALLS
        )
        monkeypatch.setattr(_tirith_mod, "_load_security_config", lambda: cfg)
        _mock_missing_tirith(monkeypatch)
        thread_factory = MagicMock()
        monkeypatch.setattr(_tirith_mod.threading, "Thread", thread_factory)

        assert ensure_installed(log_failures=False) is None

        thread_factory.assert_not_called()
        assert _state().resolved_path is None
        assert not (hermes_home / "bin").exists()


class TestLazyInstallPolicy:
    def test_low_level_installer_honors_global_opt_out(self):
        """No direct Tirith installer call may bypass the global policy."""
        with (
            patch("tools.lazy_deps._allow_lazy_installs", return_value=False),
            patch("tools.tirith_security._detect_target") as mock_target,
            patch("tools.tirith_security._download_file") as mock_download,
        ):
            result = _tirith_mod._install_tirith(log_failures=False)

        assert result == (None, "lazy_installs_disabled")
        mock_target.assert_not_called()
        mock_download.assert_not_called()

    def test_installer_rechecks_opt_out_before_replacement(
        self, tmp_path, monkeypatch
    ):
        """A live policy change during download prevents filesystem mutation."""
        source = tmp_path / "verified-tirith"
        source.write_bytes(b"verified release")
        hermes_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        policy_states = iter((True, False))
        monkeypatch.setattr(
            _lazy_deps, "_allow_lazy_installs", lambda: next(policy_states)
        )
        monkeypatch.setattr(
            _tirith_mod,
            "_download_verified_tirith",
            lambda *_args, **_kwargs: (str(source), "", True, (0, 4, 1)),
        )
        replace = MagicMock()
        monkeypatch.setattr(_tirith_mod, "_atomic_replace_binary", replace)

        assert _tirith_mod._install_tirith(log_failures=False) == (
            None,
            "lazy_installs_disabled",
        )
        replace.assert_not_called()
        assert not hermes_home.exists()

    def test_resolver_policy_transition_does_not_cache_install_failure(
        self, monkeypatch
    ):
        """A policy change during install must remain immediately reversible."""
        _state().resolved_path = None
        _state().install_failure_reason = ""

        _mock_missing_tirith(monkeypatch)
        policy_states = iter((True, False))
        monkeypatch.setattr(
            _lazy_deps, "_allow_lazy_installs", lambda: next(policy_states)
        )
        mark_failure = MagicMock()
        monkeypatch.setattr(_tirith_mod, "_mark_install_failed", mark_failure)

        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"

        mark_failure.assert_not_called()
        assert _state().resolved_path is None
        assert _state().install_failure_reason == ""

        monkeypatch.setattr(_lazy_deps, "_allow_lazy_installs", lambda: True)
        install = MagicMock(return_value=("/tmp/tirith", ""))
        monkeypatch.setattr(_tirith_mod, "_install_tirith", install)

        assert _tirith_mod._resolve_tirith_path("tirith") == "/tmp/tirith"

        install.assert_called_once_with()

    def test_background_policy_transition_does_not_cache_install_failure(
        self, monkeypatch
    ):
        """The background path must not persist a mid-install policy decision."""
        _state().resolved_path = None
        _state().install_failure_reason = ""

        _mock_missing_tirith(monkeypatch)
        policy_states = iter((True, False))
        monkeypatch.setattr(
            _lazy_deps, "_allow_lazy_installs", lambda: next(policy_states)
        )
        mark_failure = MagicMock()
        monkeypatch.setattr(_tirith_mod, "_mark_install_failed", mark_failure)

        _tirith_mod._background_install(log_failures=False)

        mark_failure.assert_not_called()
        assert _state().resolved_path is None
        assert _state().install_failure_reason == ""

        monkeypatch.setattr(_lazy_deps, "_allow_lazy_installs", lambda: True)
        install = MagicMock(return_value=("/tmp/tirith", ""))
        monkeypatch.setattr(_tirith_mod, "_install_tirith", install)

        _tirith_mod._background_install(log_failures=False)

        install.assert_called_once_with(log_failures=False)
        assert _state().resolved_path == "/tmp/tirith"

    def test_local_binary_is_discovered_when_lazy_installs_are_disabled(
        self, tmp_path, monkeypatch
    ):
        """The opt-out disables downloads, not discovery of an installed binary."""
        _state().resolved_path = None
        _state().install_failure_reason = ""
        local_tirith = tmp_path / "tirith"
        local_tirith.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        local_tirith.chmod(0o700)

        policy = MagicMock(return_value=False)
        monkeypatch.setattr(_lazy_deps, "_allow_lazy_installs", policy)
        monkeypatch.setattr(_tirith_mod, "is_platform_supported", lambda: True)
        monkeypatch.setattr(
            _tirith_mod.shutil, "which", lambda _name: str(local_tirith)
        )
        monkeypatch.setattr(
            _tirith_mod, "_clear_install_failed", lambda *_args: None
        )

        assert _tirith_mod._resolve_tirith_path("tirith") == str(local_tirith)

        policy.assert_not_called()


# ---------------------------------------------------------------------------
# Managed-cache execution boundary
# ---------------------------------------------------------------------------

class TestManagedCacheExecutionBoundary:
    def _assert_cached_managed_binary_mode_drift_is_never_spawned(
        self, fail_open, expected_action, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = home / "bin" / "tirith"
        managed.parent.mkdir(parents=True)
        managed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        managed.chmod(0o755)
        monkeypatch.setenv("HERMES_HOME", str(home))
        _state().resolved_path = str(managed)

        # Simulate trust drift after successful resolution. The cached string
        # must not bypass the current filesystem proof.
        managed.chmod(0o775)
        monkeypatch.setattr(
            _tirith_mod,
            "_load_security_config",
            lambda: {
                "tirith_enabled": True,
                "tirith_path": "tirith",
                "tirith_timeout": 5,
                "tirith_fail_open": fail_open,
            },
        )
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
        monkeypatch.setattr(_tirith_mod, "is_platform_supported", lambda: True)
        run = MagicMock()
        monkeypatch.setattr(_tirith_mod.subprocess, "run", run)

        result = check_command_security("echo guarded")

        assert result["action"] == expected_action
        assert _state().resolved_path is _tirith_mod._INSTALL_FAILED
        assert _state().install_failure_reason == "managed_cache_untrusted"
        run.assert_not_called()

    @pytest.mark.linux_only
    @pytest.mark.parametrize(
        "fail_open, expected_action",
        [(True, "allow"), (False, "block")],
    )
    def test_cached_managed_binary_mode_drift_is_never_spawned_linux(
        self, fail_open, expected_action, tmp_path, monkeypatch
    ):
        self._assert_cached_managed_binary_mode_drift_is_never_spawned(
            fail_open, expected_action, tmp_path, monkeypatch
        )

    @pytest.mark.macos_only
    @pytest.mark.parametrize(
        "fail_open, expected_action",
        [(True, "allow"), (False, "block")],
    )
    def test_cached_managed_binary_mode_drift_is_never_spawned_macos(
        self, fail_open, expected_action, tmp_path, monkeypatch
    ):
        self._assert_cached_managed_binary_mode_drift_is_never_spawned(
            fail_open, expected_action, tmp_path, monkeypatch
        )

    @pytest.mark.require_symlinks
    def test_path_alias_cannot_hide_untrusted_managed_binary(
        self, tmp_path, monkeypatch, caplog
    ):
        home = tmp_path / "home"
        managed = home / "bin" / "tirith"
        managed.parent.mkdir(parents=True)
        managed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        managed.chmod(0o777)
        alias = tmp_path / "path" / "tirith"
        alias.parent.mkdir()
        alias.symlink_to(managed)
        monkeypatch.setenv("HERMES_HOME", str(home))
        _state().resolved_path = None
        monkeypatch.setattr(
            _tirith_mod,
            "_load_security_config",
            lambda: {
                "tirith_enabled": True,
                "tirith_path": "tirith",
                "tirith_timeout": 5,
                "tirith_fail_open": False,
            },
        )
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: str(alias))
        monkeypatch.setattr(_tirith_mod, "is_platform_supported", lambda: True)
        run = MagicMock()
        monkeypatch.setattr(_tirith_mod.subprocess, "run", run)

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            result = check_command_security("echo guarded")
            repeated = check_command_security("echo guarded again")

        assert result["action"] == "block"
        assert repeated["action"] == "block"
        assert _state().install_failure_reason == "managed_cache_untrusted"
        trust_warnings = [
            record
            for record in caplog.records
            if "failed ownership, mode, ACL, or link validation" in record.message
        ]
        assert len(trust_warnings) == 1
        run.assert_not_called()

    @pytest.mark.require_symlinks
    def test_trusted_managed_path_alias_is_normalized_before_execution(
        self, tmp_path, monkeypatch
    ):
        home = tmp_path / "home"
        managed = home / "bin" / "tirith"
        managed.parent.mkdir(parents=True)
        managed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        managed.chmod(0o755)
        alias = tmp_path / "path" / "tirith"
        alias.parent.mkdir()
        alias.symlink_to(managed)
        monkeypatch.setenv("HERMES_HOME", str(home))
        _state().resolved_path = None
        monkeypatch.setattr(
            _tirith_mod,
            "_load_security_config",
            lambda: {
                "tirith_enabled": True,
                "tirith_path": "tirith",
                "tirith_timeout": 5,
                "tirith_fail_open": False,
            },
        )
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: str(alias))
        monkeypatch.setattr(_tirith_mod, "_schedule_managed_update", lambda *_a, **_kw: None)
        run = MagicMock(return_value=_mock_run(0, _json_stdout()))
        monkeypatch.setattr(_tirith_mod.subprocess, "run", run)

        result = check_command_security("echo safe")

        assert result["action"] == "allow"
        assert _state().resolved_path == str(managed)
        assert run.call_args.args[0][0] == str(managed)

    @pytest.mark.macos_only
    def test_case_alias_is_recognized_as_managed_file(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        managed = home / "bin" / "tirith"
        managed.parent.mkdir(parents=True)
        managed.write_text("scanner", encoding="utf-8")
        managed.chmod(0o755)
        alias = managed.with_name("TIRITH")
        if not alias.exists():
            pytest.skip("test filesystem is case-sensitive")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert alias != managed
        assert alias.samefile(managed)
        assert _tirith_mod._is_managed_tirith_location(str(alias))
        assert _tirith_mod._validated_tirith_path(str(alias)) == str(managed)


# ---------------------------------------------------------------------------
# Unsupported managed-install platform (Windows etc.)
# ---------------------------------------------------------------------------

class TestUnsupportedPlatform:
    """Manager support must not disable an operator-provided scanner."""

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
             patch("tools.tirith_security.platform.machine", return_value=machine), \
             patch("tools.tirith_security.is_termux", return_value=False):
            assert _tirith_mod.is_platform_supported() is expected

    @pytest.mark.parametrize(
        "machine, expected",
        [
            ("aarch64", "aarch64-unknown-linux-musl"),
            ("arm64", "aarch64-unknown-linux-musl"),
            ("x86_64", None),
        ],
    )
    def test_termux_uses_only_published_musl_target(self, machine, expected):
        with patch("tools.tirith_security.platform.system", return_value="Linux"), \
             patch("tools.tirith_security.platform.machine", return_value=machine), \
             patch("tools.tirith_security.is_termux", return_value=True):
            assert _tirith_mod._detect_target() == expected


    @patch("tools.tirith_security._load_security_config")
    def test_check_command_security_unsupported_allows_silently(self, mock_cfg):
        """Default fail-open remains silent when no local scanner exists."""
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _state().resolved_path = None
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.shutil.which", return_value=None) as mock_which, \
             patch("tools.tirith_security.subprocess.run") as mock_run, \
             patch("tools.tirith_security._warn_once") as mock_warn:
            result = check_command_security("rm -rf /")
            assert result == {"action": "allow", "findings": [], "summary": ""}
            mock_run.assert_not_called()
            mock_warn.assert_not_called()
            mock_which.assert_called_once_with("tirith")

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_explicit_path_scans_on_unsupported_platform(
        self, mock_cfg, mock_run, tmp_path
    ):
        """A working explicit scanner remains authoritative everywhere."""
        binary = tmp_path / "tirith"
        binary.write_text("scanner", encoding="utf-8")
        binary.chmod(0o700)
        mock_cfg.return_value = {"tirith_enabled": True,
                                 "tirith_path": str(binary),
                                 "tirith_timeout": 5,
                                 "tirith_fail_open": False}
        mock_run.return_value = _mock_run(
            1,
            _json_stdout([{"rule_id": "homograph_url"}], "blocked"),
        )
        _state().resolved_path = None
        with patch("tools.tirith_security.is_platform_supported", return_value=False):
            result = check_command_security("curl https://example.test")

        assert result["action"] == "block"
        assert mock_run.call_args.args[0][0] == str(binary)

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_default_path_scans_on_unsupported_platform(
        self, mock_cfg, mock_run, tmp_path
    ):
        """Manager support does not suppress the ordinary PATH contract."""
        binary = tmp_path / "tirith"
        binary.write_text("scanner", encoding="utf-8")
        binary.chmod(0o700)
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        mock_run.return_value = _mock_run(2, _json_stdout(summary="review"))
        _state().resolved_path = None
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.shutil.which",
                   return_value=str(binary)), \
             patch("tools.tirith_security.threading.Thread") as mock_thread:
            result = check_command_security("curl https://example.test")

        assert result["action"] == "warn"
        assert mock_run.call_args.args[0][0] == str(binary)
        mock_thread.assert_not_called()

    @pytest.mark.windows_only
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_native_windows_explicit_scanner_remains_usable(
        self, mock_cfg, mock_run
    ):
        """Native Windows may scan with an external binary without manager support."""
        mock_cfg.return_value = {
            "tirith_enabled": True,
            "tirith_path": sys.executable,
            "tirith_timeout": 5,
            "tirith_fail_open": False,
        }
        mock_run.return_value = _mock_run(
            1,
            _json_stdout([{"rule_id": "homograph_url"}], "blocked"),
        )
        state = _state(sys.executable)
        state.resolved_path = None

        assert not _tirith_mod.is_platform_supported()
        result = check_command_security("curl https://example.test")

        assert result["action"] == "block"
        assert mock_run.call_args.args[0][0] == sys.executable

    @patch("tools.tirith_security._load_security_config")
    def test_unsupported_missing_scanner_honors_fail_closed(self, mock_cfg):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": False}
        _state().resolved_path = None
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security.subprocess.run") as mock_run:
            result = check_command_security("echo guarded")

        assert result == {
            "action": "block",
            "findings": [],
            "summary": "tirith path unavailable (fail-closed)",
        }
        mock_run.assert_not_called()

    @patch("tools.tirith_security._load_security_config")
    def test_ensure_installed_honors_explicit_path(self, mock_cfg, tmp_path):
        binary = tmp_path / "tirith"
        binary.write_text("scanner", encoding="utf-8")
        binary.chmod(0o700)
        mock_cfg.return_value = {"tirith_enabled": True,
                                 "tirith_path": str(binary),
                                 "tirith_timeout": 5,
                                 "tirith_fail_open": True}
        state = _state(str(binary))
        state.resolved_path = None
        state.install_failure_reason = "explicit_path_missing"
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.threading.Thread") as mock_thread:
            assert ensure_installed() == str(binary)

        mock_thread.assert_not_called()
        assert state.install_failure_reason == ""

    @patch("tools.tirith_security._load_security_config")
    def test_ensure_installed_honors_path(self, mock_cfg, tmp_path):
        binary = tmp_path / "tirith"
        binary.write_text("scanner", encoding="utf-8")
        binary.chmod(0o700)
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        _state().resolved_path = None
        with patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.shutil.which",
                   return_value=str(binary)), \
             patch("tools.tirith_security.threading.Thread") as mock_thread:
            assert ensure_installed() == str(binary)

        mock_thread.assert_not_called()


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
        _state().resolved_path = None

        # First call: tries install, fails
        _resolve_tirith_path("tirith")
        assert mock_install.call_count == 1
        assert _state().resolved_path is _INSTALL_FAILED
        mock_mark.assert_called_once_with("download_failed")  # reason persisted

        # Second call: hits the cache, does NOT call _install_tirith again
        _resolve_tirith_path("tirith")
        assert mock_install.call_count == 1  # still 1, not 2

        _state().resolved_path = None


# ---------------------------------------------------------------------------
# Explicit path must not auto-download (Finding #2)
# ---------------------------------------------------------------------------

class TestExplicitPathNoAutoDownload:
    def test_explicit_path_recovery_clears_stale_failure_reason(self, tmp_path):
        binary = tmp_path / "tirith"
        binary.write_text("scanner", encoding="utf-8")
        binary.chmod(0o700)
        state = _state(str(binary))
        state.resolved_path = _tirith_mod._INSTALL_FAILED
        state.install_failure_reason = "explicit_path_missing"

        assert _tirith_mod._resolve_tirith_path(str(binary)) == str(binary)
        assert state.install_failure_reason == ""

    @patch("tools.tirith_security._install_tirith")
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_tilde_explicit_path_missing_no_download(self, mock_which, mock_install):
        """An explicit ~/path that doesn't exist must NOT trigger download."""
        from tools.tirith_security import _resolve_tirith_path, _INSTALL_FAILED
        _state("~/bin/tirith").resolved_path = None

        result = _resolve_tirith_path("~/bin/tirith")
        mock_install.assert_not_called()
        assert _state("~/bin/tirith").resolved_path is _INSTALL_FAILED
        assert result is not None
        assert "~" not in result  # tilde still expanded

        _state("~/bin/tirith").resolved_path = None

    @patch("tools.tirith_security._mark_install_failed")
    @patch("tools.tirith_security._is_install_failed_on_disk", return_value=False)
    @patch("tools.tirith_security._install_tirith", return_value=("/auto/tirith", ""))
    @patch("tools.tirith_security.shutil.which", return_value=None)
    def test_default_path_does_auto_download(self, mock_which, mock_install,
                                              mock_disk_check, mock_mark):
        """The default bare 'tirith' SHOULD trigger auto-download."""
        from tools.tirith_security import _resolve_tirith_path
        _state().resolved_path = None

        result = _resolve_tirith_path("tirith")
        mock_install.assert_called_once()
        assert result == "/auto/tirith"

        _state().resolved_path = None


# ---------------------------------------------------------------------------
# Cosign provenance verification (P1)
# ---------------------------------------------------------------------------

class TestCosignVerification:
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security.shutil.which", return_value="/usr/bin/cosign")
    @patch(
        "tools.tirith_security._release_identity_from_certificate",
        return_value=(
            (0, 4, 1),
            "https://github.com/sheeki03/tirith/.github/workflows/"
            "release.yml@refs/tags/v0.4.1",
        ),
    )
    def test_cosign_identity_pinned_to_exact_release_tag(
        self, mock_identity, mock_which, mock_run
    ):
        """Verification binds the manifest to one authenticated stable tag."""
        del mock_identity, mock_which
        from tools.tirith_security import _verify_cosign
        mock_run.return_value = _mock_run(0, "Verified OK")
        assert _verify_cosign(
            "/tmp/checksums.txt", "/tmp/sig", "/tmp/cert"
        ) == (True, (0, 4, 1))
        args = mock_run.call_args[0][0]
        idx = args.index("--certificate-identity")
        assert args[idx + 1].endswith("release.yml@refs/tags/v0.4.1")
        assert "--certificate-identity-regexp" not in args

    @pytest.mark.parametrize(
        "identity, expected",
        [
            (
                "https://github.com/sheeki03/tirith/.github/workflows/"
                "release.yml@refs/tags/v0.4.1",
                (0, 4, 1),
            ),
            (
                "https://github.com/sheeki03/tirith/.github/workflows/"
                "release.yml@refs/tags/v0.4.1-rc.1",
                None,
            ),
            (
                "https://github.com/sheeki03/tirith/.github/workflows/"
                "release.yml@refs/tags/v00.4.1",
                None,
            ),
            (
                "https://github.com/sheeki03/tirith/.github/workflows/"
                "other.yml@refs/tags/v0.4.1",
                None,
            ),
        ],
    )
    def test_certificate_identity_accepts_only_stable_release_tag(
        self, tmp_path, identity, expected
    ):
        certificate = tmp_path / "checksums.txt.pem"
        _write_release_certificate(certificate, [identity])

        parsed = _tirith_mod._release_identity_from_certificate(str(certificate))

        assert (parsed[0] if parsed else None) == expected

    def test_certificate_with_ambiguous_release_identities_is_rejected(
        self, tmp_path
    ):
        prefix = (
            "https://github.com/sheeki03/tirith/.github/workflows/"
            "release.yml@refs/tags/v"
        )
        certificate = tmp_path / "checksums.txt.pem"
        _write_release_certificate(certificate, [prefix + "0.4.1", prefix + "0.4.2"])

        assert _tirith_mod._release_identity_from_certificate(str(certificate)) is None

    def test_upstream_base64_wrapped_certificate_is_accepted(self, tmp_path):
        identity = (
            "https://github.com/sheeki03/tirith/.github/workflows/"
            "release.yml@refs/tags/v0.4.1"
        )
        certificate = tmp_path / "checksums.txt.pem"
        _write_release_certificate(certificate, [identity])
        certificate.write_bytes(base64.b64encode(certificate.read_bytes()) + b"\n")

        assert _tirith_mod._release_identity_from_certificate(str(certificate)) == (
            (0, 4, 1),
            identity,
        )

    @pytest.mark.parametrize("malformation", ["trailing", "bundle", "double-wrap"])
    def test_noncanonical_certificate_framing_is_rejected(
        self, malformation, tmp_path
    ):
        identity = (
            "https://github.com/sheeki03/tirith/.github/workflows/"
            "release.yml@refs/tags/v0.4.1"
        )
        certificate = tmp_path / "checksums.txt.pem"
        _write_release_certificate(certificate, [identity])
        raw = certificate.read_bytes()
        malformed = {
            "trailing": raw + b"unexpected",
            "bundle": raw + raw,
            "double-wrap": base64.b64encode(base64.b64encode(raw)),
        }[malformation]
        certificate.write_bytes(malformed)

        assert _tirith_mod._release_identity_from_certificate(str(certificate)) is None


    @patch(
        "tools.tirith_security._extract_release_archive",
        return_value=(None, "binary_not_in_archive"),
    )
    @patch("tools.tirith_security._verify_checksum", return_value=True)
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._download_file")
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_install_proceeds_without_cosign(self, mock_target, mock_dl,
                                              mock_which, mock_checksum,
                                              mock_extract):
        """_install_tirith proceeds with SHA-256 only when cosign is not on PATH."""
        from tools.tirith_security import _install_tirith

        path, reason = _install_tirith()
        # Reaches extraction (no binary in mock archive), but got past cosign
        assert path is None
        assert reason == "binary_not_in_archive"
        assert mock_checksum.called  # SHA-256 verification ran
        mock_extract.assert_called_once()

    @patch("tools.tirith_security._extract_release_archive")
    @patch("tools.tirith_security._verify_checksum")
    @patch("tools.tirith_security.shutil.which", return_value=None)
    @patch("tools.tirith_security._download_file")
    @patch("tools.tirith_security._detect_target", return_value="aarch64-apple-darwin")
    def test_replacement_requires_cosign(self, mock_target, mock_dl,
                                         mock_which, mock_checksum,
                                         mock_extract):
        """Automatic replacement must not downgrade to checksum-only trust."""
        del mock_target, mock_dl, mock_which
        from tools.tirith_security import _install_tirith

        path, reason = _install_tirith(
            expected_existing_sha256="0" * 64,
            current_version=(0, 4, 0),
        )

        assert path is None
        assert reason == "cosign_missing"
        mock_checksum.assert_not_called()
        mock_extract.assert_not_called()

    def test_unsigned_initial_download_cannot_race_into_replacement(
        self, tmp_path, monkeypatch
    ):
        """A concurrently created managed binary forces fail-closed replacement."""
        hermes_home = tmp_path / "hermes-home"
        verified = tmp_path / "verified-tirith"
        verified.write_bytes(b"new tirith")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(
            _tirith_mod, "_detect_target", lambda: "aarch64-apple-darwin"
        )

        def unsigned_download(*_args, **_kwargs):
            existing = hermes_home / "bin" / "tirith"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"concurrent tirith")
            return str(verified), "", False, None

        monkeypatch.setattr(
            _tirith_mod, "_download_verified_tirith", unsigned_download
        )

        path, reason = _tirith_mod._install_tirith(log_failures=False)

        assert path is None
        assert reason == "cosign_required_for_replacement"
        assert (hermes_home / "bin" / "tirith").read_bytes() == b"concurrent tirith"

    def test_initial_install_cannot_clobber_late_concurrent_winner(
        self, tmp_path, monkeypatch
    ):
        """The filesystem commit enforces no-clobber after the final path check."""
        hermes_home = tmp_path / "hermes-home"
        verified = tmp_path / "verified-tirith"
        verified.write_bytes(b"new tirith")
        destination = hermes_home / "bin" / "tirith"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(
            _tirith_mod, "_detect_target", lambda: "aarch64-apple-darwin"
        )
        monkeypatch.setattr(
            _tirith_mod,
            "_download_verified_tirith",
            lambda *_args, **_kwargs: (str(verified), "", False, None),
        )
        real_directory_check = _tirith_mod._managed_install_directory_is_real

        def concurrent_install_after_path_check():
            destination.write_bytes(b"concurrent tirith")
            destination.chmod(0o755)
            return real_directory_check()

        monkeypatch.setattr(
            _tirith_mod,
            "_managed_install_directory_is_real",
            concurrent_install_after_path_check,
        )

        path, reason = _tirith_mod._install_tirith(log_failures=False)

        assert path is None
        assert reason == "install_replace_failed"
        assert destination.read_bytes() == b"concurrent tirith"
        assert os.access(destination, os.X_OK)


class TestReleaseDownloadLimits:
    def test_github_token_is_not_forwarded_to_release_asset_redirect(
        self, tmp_path, monkeypatch
    ):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.return_value = b""
        secure_open = MagicMock(return_value=response)
        monkeypatch.setattr(_tirith_mod, "open_credentialed_url", secure_open)
        monkeypatch.setattr(
            "agent.secret_scope.get_secret",
            lambda key: "secret-token" if key == "GITHUB_TOKEN" else None,
        )

        _tirith_mod._download_file(
            "https://github.com/sheeki03/tirith/releases/latest/download/checksums.txt",
            str(tmp_path / "checksums.txt"),
            max_bytes=4,
        )

        initial_request = secure_open.call_args.args[0]
        assert secure_open.call_args.kwargs == {"timeout": 10}
        assert initial_request.get_header("Authorization") == "token secret-token"
        redirected_request = (
            _tirith_mod.urllib.request.HTTPRedirectHandler().redirect_request(
                initial_request,
                io.BytesIO(),
                302,
                "Found",
                HTTPMessage(),
                "https://release-assets.githubusercontent.com/checksums.txt",
            )
        )
        assert redirected_request is not None
        assert redirected_request.get_header("Authorization") is None

    def test_download_rejects_response_over_limit(self, tmp_path, monkeypatch):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        response.read.side_effect = [b"12345", b""]
        monkeypatch.setattr(
            _tirith_mod,
            "open_credentialed_url",
            MagicMock(return_value=response),
        )
        destination = tmp_path / "metadata"

        with pytest.raises(ValueError, match="exceeds 4-byte limit"):
            _tirith_mod._download_file(
                "https://example.invalid/checksums.txt",
                str(destination),
                max_bytes=4,
            )

        assert not destination.exists()


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
        def _download(url, dest, timeout=10, *, max_bytes):
            del timeout, max_bytes
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
        assert path is not None
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

    def test_install_rejects_oversized_tirith_member(self, tmp_path):
        member = tarfile.TarInfo("tirith")
        member.size = _tirith_mod._MAX_TIRITH_BINARY_BYTES + 1
        archive = MagicMock()
        archive.__iter__.return_value = iter([member])

        path, reason = _tirith_mod._extract_tirith_binary(
            archive, str(tmp_path), lambda *_args: None
        )

        assert path is None
        assert reason == "archive_member_too_large"
        archive.extractfile.assert_not_called()

    def test_install_rejects_too_many_archive_members(self, tmp_path):
        members = [
            tarfile.TarInfo(f"extra-{index}")
            for index in range(_tirith_mod._MAX_RELEASE_ARCHIVE_MEMBERS + 1)
        ]
        archive = MagicMock()
        archive.__iter__.return_value = iter(members)

        path, reason = _tirith_mod._extract_tirith_binary(
            archive, str(tmp_path), lambda *_args: None
        )

        assert path is None
        assert reason == "too_many_archive_members"
        archive.extractfile.assert_not_called()

    def test_install_caps_pax_metadata_before_tarfile_yields_it(
        self, tmp_path, monkeypatch
    ):
        archive = tmp_path / "pax-metadata.tar.gz"
        with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
            metadata_heavy = tarfile.TarInfo("nested/" + "a" * 16_384)
            metadata_heavy.size = 0
            tar.addfile(metadata_heavy)
            binary = tarfile.TarInfo("tirith")
            binary.size = 1
            tar.addfile(binary, io.BytesIO(b"x"))

        monkeypatch.setattr(
            _tirith_mod,
            "_MAX_RELEASE_ARCHIVE_UNPACKED_BYTES",
            4 * 1024,
        )

        path, reason = _tirith_mod._extract_release_archive(
            str(archive), str(tmp_path), lambda *_args: None
        )

        assert path is None
        assert reason == "archive_too_large"
        assert not (tmp_path / "tirith").exists()


# ---------------------------------------------------------------------------
# Background install / non-blocking startup (P2)
# ---------------------------------------------------------------------------

class TestBackgroundInstall:
    def test_ensure_installed_non_blocking(self):
        """ensure_installed must return immediately when download needed."""
        _state().resolved_path = None

        with patch("tools.tirith_security._load_security_config",
                   return_value={"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}), \
             patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._managed_tirith_path", return_value="/nonexistent/tirith"), \
             patch("tools.tirith_security._is_install_failed_on_disk", return_value=False), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            mock_thread = MagicMock()
            mock_thread.is_alive.return_value = False
            MockThread.return_value = mock_thread

            result = ensure_installed()
            assert result is None  # not available yet
            MockThread.assert_called_once()
            mock_thread.start.assert_called_once()

        _state().resolved_path = None

    def test_resolve_returns_default_when_thread_alive(self):
        """_resolve_tirith_path returns default while background thread runs."""
        from tools.tirith_security import _resolve_tirith_path
        _state().resolved_path = None
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        _state().install_thread = mock_thread

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._managed_tirith_path", return_value="/nonexistent/tirith"):
            result = _resolve_tirith_path("tirith")
            assert result == "tirith"  # returns configured default, doesn't block

        _state().install_thread = None
        _state().resolved_path = None

    def test_approval_path_starts_missing_install_in_background(self):
        """A first command must not synchronously download Tirith."""
        _state().resolved_path = None
        _state().install_thread = None
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = False

        with (
            patch(
                "tools.tirith_security._load_security_config",
                return_value={
                    "tirith_enabled": True,
                    "tirith_path": "tirith",
                    "tirith_timeout": 5,
                    "tirith_fail_open": True,
                },
            ),
            patch("tools.tirith_security.is_platform_supported", return_value=True),
            patch("tools.tirith_security.shutil.which", return_value=None),
            patch(
                "tools.tirith_security._managed_tirith_path",
                return_value="/nonexistent/tirith",
            ),
            patch("tools.tirith_security._read_failure_reason", return_value=None),
            patch("tools.tirith_security.threading.Thread", return_value=mock_thread),
            patch("tools.tirith_security._install_tirith") as install,
            patch("tools.tirith_security.subprocess.run") as run,
        ):
            result = check_command_security("echo hi")

        assert result["action"] == "allow"
        assert "unavailable" in result["summary"]
        mock_thread.start.assert_called_once_with()
        install.assert_not_called()
        run.assert_not_called()


class TestRuntimeProfileIsolation:
    def test_interleaved_profiles_keep_distinct_managed_paths(
        self, tmp_path, monkeypatch
    ):
        homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
        managed_paths = []
        for home in homes:
            home.mkdir(mode=0o700)
            managed = home / "bin" / "tirith"
            managed.parent.mkdir(mode=0o700)
            managed.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            managed.chmod(0o700)
            managed_paths.append(managed)

        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
        monkeypatch.setattr(_tirith_mod, "is_platform_supported", lambda: True)
        monkeypatch.setattr(
            _tirith_mod, "_uses_image_managed_tirith_root", lambda: False
        )
        monkeypatch.setattr(
            _tirith_mod, "_schedule_managed_update", lambda *_args, **_kwargs: None
        )

        resolved = []
        states = []
        for home in (homes[0], homes[1], homes[0]):
            token = set_hermes_home_override(home)
            try:
                resolved.append(_tirith_mod._resolve_tirith_path("tirith"))
                states.append(_state())
            finally:
                reset_hermes_home_override(token)

        assert resolved == [
            str(managed_paths[0]),
            str(managed_paths[1]),
            str(managed_paths[0]),
        ]
        assert states[0] is states[2]
        assert states[0] is not states[1]

    def test_background_update_inherits_profile_context(
        self, tmp_path, monkeypatch
    ):
        profile_home = tmp_path / "profile-b"
        profile_home.mkdir()
        seen = {}

        def record_update(path, *, log_failures=True):
            seen["home"] = get_hermes_home()
            seen["path"] = path
            seen["log_failures"] = log_failures

        monkeypatch.setattr(_tirith_mod, "_background_update", record_update)
        monkeypatch.setattr(_tirith_mod, "is_platform_supported", lambda: True)
        monkeypatch.setattr(_tirith_mod, "_is_managed_tirith", lambda _path: True)
        monkeypatch.setattr(
            _tirith_mod, "_tirith_auto_install_allowed", lambda: True
        )
        monkeypatch.setattr(_tirith_mod, "_update_is_due", lambda: True)

        token = set_hermes_home_override(profile_home)
        try:
            state = _state()
            _tirith_mod._schedule_managed_update(
                "/managed/tirith", "tirith", log_failures=False, state=state
            )
            worker = state.update_thread
        finally:
            reset_hermes_home_override(token)

        assert worker is not None
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert seen == {
            "home": profile_home,
            "path": "/managed/tirith",
            "log_failures": False,
        }

    def test_background_install_inherits_profile_context(
        self, tmp_path, monkeypatch
    ):
        profile_home = tmp_path / "profile-b"
        profile_home.mkdir()
        seen = {}

        def record_install(*, log_failures=True, state=None):
            seen["home"] = get_hermes_home()
            seen["log_failures"] = log_failures
            seen["state"] = state

        monkeypatch.setattr(_tirith_mod, "_background_install", record_install)
        monkeypatch.setattr(
            _tirith_mod,
            "_load_security_config",
            lambda: {
                "tirith_enabled": True,
                "tirith_path": "tirith",
                "tirith_timeout": 5,
                "tirith_fail_open": True,
            },
        )
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
        monkeypatch.setattr(_tirith_mod, "is_platform_supported", lambda: True)
        monkeypatch.setattr(
            _tirith_mod, "_uses_image_managed_tirith_root", lambda: False
        )
        monkeypatch.setattr(
            _tirith_mod, "_tirith_auto_install_allowed", lambda: True
        )
        monkeypatch.setattr(_tirith_mod, "_read_failure_reason", lambda: None)

        token = set_hermes_home_override(profile_home)
        try:
            state = _state()
            assert ensure_installed(log_failures=False) is None
            worker = state.install_thread
        finally:
            reset_hermes_home_override(token)

        assert worker is not None
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert seen == {
            "home": profile_home,
            "log_failures": False,
            "state": state,
        }

    def test_circuit_breakers_are_scoped_per_profile(
        self, tmp_path, monkeypatch
    ):
        profile_a = tmp_path / "profile-a"
        profile_b = tmp_path / "profile-b"
        profile_a.mkdir()
        profile_b.mkdir()
        monkeypatch.setattr(_tirith_mod, "_load_security_config", lambda: _CFG)
        monkeypatch.setattr(
            _tirith_mod,
            "_resolve_tirith_path",
            lambda *_args, **_kwargs: "tirith",
        )
        run = MagicMock(return_value=_mock_run(0, _json_stdout()))
        monkeypatch.setattr(_tirith_mod.subprocess, "run", run)

        token = set_hermes_home_override(profile_a)
        try:
            state_a = _state()
            state_a.crash_count = _tirith_mod._CRASH_LIMIT
            state_a.circuit_open = True
            state_a.circuit_opened_at = time.monotonic()
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(profile_b)
        try:
            assert check_command_security("echo profile-b")["action"] == "allow"
            state_b = _state()
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(profile_a)
        try:
            blocked_profile = check_command_security("echo profile-a")
        finally:
            reset_hermes_home_override(token)

        assert state_a is not state_b
        assert blocked_profile["action"] == "allow"
        assert "circuit breaker" in blocked_profile["summary"]
        run.assert_called_once()


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
        _state().resolved_path = _INSTALL_FAILED
        _state().install_failure_reason = "cosign_exec_failed"

        with patch("tools.tirith_security.shutil.which", return_value=None), \
             patch("tools.tirith_security._managed_tirith_path", return_value="/nonexistent/tirith"), \
             patch("tools.tirith_security._install_tirith") as mock_install:
            result = _resolve_tirith_path("tirith")
            assert result == "tirith"  # fallback
            mock_install.assert_not_called()

        _state().resolved_path = None


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
    @patch("tools.tirith_security._resolve_tirith_path", return_value="tirith")
    @patch("tools.tirith_security._load_security_config")
    def test_repeated_spawn_failure_logs_once(
        self, mock_cfg, _mock_resolve, mock_run, caplog
    ):
        mock_cfg.return_value = {
            "tirith_enabled": True, "tirith_path": "tirith",
            "tirith_timeout": 5, "tirith_fail_open": True,
        }
        mock_run.side_effect = FileNotFoundError("[WinError 2]")
        # Fresh dedupe state — clear any keys left by other tests.
        _tirith_mod._reset_spawn_warning_state()

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


class TestCircuitBreakerRecovery:
    @pytest.mark.parametrize(
        "returncode, expected_action",
        [(0, "allow"), (1, "block"), (2, "warn")],
    )
    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_recognized_verdict_resets_prior_failures(
        self, mock_cfg, mock_run, returncode, expected_action
    ):
        mock_cfg.return_value = _CFG
        _state().crash_count = _tirith_mod._CRASH_LIMIT - 1
        mock_run.return_value = _mock_run(returncode, _json_stdout([], "review"))

        result = check_command_security("echo review")

        assert result["action"] == expected_action
        assert _state().crash_count == 0
        assert not _state().circuit_open

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_open_circuit_honors_fail_closed(self, mock_cfg, mock_run):
        mock_cfg.return_value = {**_CFG, "tirith_fail_open": False}
        _state().crash_count = _tirith_mod._CRASH_LIMIT
        _state().circuit_open = True
        _state().circuit_opened_at = time.monotonic()

        result = check_command_security("echo blocked")

        assert result["action"] == "block"
        assert "fail-closed" in result["summary"]
        mock_run.assert_not_called()

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_open_circuit_makes_half_open_recovery_probe(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        _state().crash_count = _tirith_mod._CRASH_LIMIT
        _state().circuit_open = True
        _state().circuit_opened_at = 100.0
        mock_run.return_value = _mock_run(0, _json_stdout())

        with patch(
            "tools.tirith_security.time.monotonic",
            return_value=100.0 + _tirith_mod._CIRCUIT_RETRY_SECONDS,
        ):
            result = check_command_security("echo recovered")

        assert result["action"] == "allow"
        assert not _state().circuit_open
        assert _state().circuit_opened_at is None
        mock_run.assert_called_once()

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_half_open_recovery_probe_is_single_flight(
        self, mock_cfg, mock_run
    ):
        mock_cfg.return_value = _CFG
        state = _state()
        state.resolved_path = None
        state.crash_count = _tirith_mod._CRASH_LIMIT
        state.circuit_open = True
        state.circuit_opened_at = 100.0
        entered = threading.Event()
        release = threading.Event()
        first_result = {}
        current_time = [100.0 + _tirith_mod._CIRCUIT_RETRY_SECONDS]

        def slow_success(*_args, **_kwargs):
            entered.set()
            assert release.wait(5), "test did not release half-open probe"
            return _mock_run(0, _json_stdout())

        mock_run.side_effect = slow_success

        with (
            patch(
                "tools.tirith_security.time.monotonic",
                side_effect=lambda: current_time[0],
            ),
            patch(
                "tools.tirith_security.shutil.which",
                return_value="/usr/local/bin/tirith",
            ),
            patch(
                "tools.tirith_security._validated_tirith_path",
                side_effect=lambda path: path,
            ),
            patch("tools.tirith_security._schedule_managed_update"),
        ):
            worker = threading.Thread(
                target=lambda: first_result.update(
                    result=check_command_security("echo first")
                )
            )
            worker.start()
            assert entered.wait(5), "half-open probe did not start"

            # Even after several additional cooldown periods, the explicit
            # in-flight claim prevents another probe from starting.
            current_time[0] = 100.0 + 3 * _tirith_mod._CIRCUIT_RETRY_SECONDS
            second = check_command_security("echo second")
            assert second["action"] == "allow"
            assert "circuit breaker" in second["summary"]
            assert mock_run.call_count == 1

            release.set()
            worker.join(5)

        assert not worker.is_alive()
        assert first_result["result"]["action"] == "allow"
        assert not _state().circuit_open

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_programming_error_releases_half_open_claim(self, mock_cfg, mock_run):
        mock_cfg.return_value = _CFG
        state = _state()
        state.crash_count = _tirith_mod._CRASH_LIMIT
        state.circuit_open = True
        state.circuit_opened_at = 100.0
        mock_run.side_effect = [
            AttributeError("unexpected bug"),
            _mock_run(0, _json_stdout()),
        ]

        with patch(
            "tools.tirith_security.time.monotonic",
            return_value=100.0 + _tirith_mod._CIRCUIT_RETRY_SECONDS,
        ):
            with pytest.raises(AttributeError, match="unexpected bug"):
                check_command_security("echo broken probe")

            assert state.circuit_open
            assert not state.circuit_probe_in_flight
            recovered = check_command_security("echo retry probe")

        assert recovered["action"] == "allow"
        assert not state.circuit_open
        assert mock_run.call_count == 2


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
    def test_non_app_finding_beyond_display_cap_preserves_warn(
        self, mock_cfg, mock_run
    ):
        """Suppression must consider findings that are not returned to callers."""
        mock_cfg.return_value = _CFG
        findings = [
            {"rule_id": "lookalike_tld", "value": ".app"}
            for _ in range(_tirith_mod._MAX_FINDINGS)
        ]
        findings.append({"rule_id": "shortened_url", "severity": "medium"})
        mock_run.return_value = _mock_run(2, _json_stdout(findings, "mixed"))

        result = check_command_security("curl https://bit.ly/test.app")

        assert result["action"] == "warn"
        assert len(result["findings"]) == _tirith_mod._MAX_FINDINGS

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
        ({"rule_id": "lookalike_tld", "value": ".apple"}, False),
        ({"rule_id": "lookalike_tld", "description": "Domain uses '.application' TLD"}, False),
        ({"rule_id": "lookalike_tld", "message": "Docs mention .app; domain uses '.zip' TLD"}, False),
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
