"""Tests for the tirith security scanning subprocess wrapper."""

import json
import logging
import os
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pm
import pytest

import tools.tirith_security as _tirith_mod
from tools.tirith_security import check_command_security, ensure_installed


@pytest.fixture(autouse=True)
def _no_real_scanner(monkeypatch):
    """Nothing in this file may resolve a scanner on the machine it runs on. The module now reads a
    candidate's own header, so a rule that used to be free of I/O (`shutil.which` hit, PM's installed
    binary) would otherwise open a path under the real hermes home — which `home_io_guard` refuses, and
    which the developer's own `PATH` decides. A case that wants a scanner patches the slot itself."""
    monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(pm, "installed_package", MagicMock(return_value=None))


def _reset_state():
    _tirith_mod._install_attempted.clear()
    _tirith_mod._install_threads.clear()
    _tirith_mod._crash_count = 0
    _tirith_mod._circuit_open = False
    _tirith_mod._circuit_open_at = 0.0


@pytest.fixture(autouse=True)
def _reset_tirith_state():
    _reset_state()
    yield
    _reset_state()


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
        assert ensure_installed() is None

    @patch("tools.tirith_security.shutil.which", return_value="/usr/local/bin/tirith")
    @patch("tools.tirith_security._load_security_config")
    def test_found_on_path_returns_immediately(self, mock_cfg, mock_which):
        mock_cfg.return_value = {"tirith_enabled": True, "tirith_path": "tirith",
                                 "tirith_timeout": 5, "tirith_fail_open": True}
        assert ensure_installed() == "/usr/local/bin/tirith"


# ---------------------------------------------------------------------------
# Unsupported platform (Windows etc.) — silent fast-path everywhere
# ---------------------------------------------------------------------------

class TestUnsupportedPlatform:
    """When PM has no tirith build for this OS+arch, the entire subsystem
    must stay silent: no install thread, no spawn attempts, no CLI banner. Pattern-matching
    guards still cover the gap; tirith content scanning is just absent."""

    @pytest.mark.parametrize("target, expected", [
        ("linux-x64", True),
        ("win32-x64", False),
        (RuntimeError("unsupported architecture: riscv64"), False),
    ])
    def test_is_platform_supported(self, target, expected):
        # Table inputs, not a host fake: support is PM's per-target mapping.
        current_target = MagicMock(side_effect=[target])
        with patch("pm.current_target", current_target):
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

    def test_explicit_path_still_honored_on_unsupported_platform(self, tmp_path):
        """If a user explicitly configured a tirith_path (e.g. they built it
        themselves under WSL), the unsupported-platform short-circuit must
        NOT override that — explicit config wins."""
        custom = tmp_path / "tirith"
        custom.write_text("#!/bin/sh\nexit 0\n")
        custom.chmod(0o755)
        with patch("tools.tirith_security.is_platform_supported", return_value=False):
            assert _tirith_mod._resolve_tirith_path(str(custom)) == str(custom)


# ---------------------------------------------------------------------------
# PM-provisioned binary: one install attempt per home, explicit paths never download
# ---------------------------------------------------------------------------

_BARE_CFG = {"tirith_enabled": True, "tirith_path": "tirith",
             "tirith_timeout": 5, "tirith_fail_open": True}


@pytest.fixture
def pm_tirith(monkeypatch):
    """Nothing on PATH, nothing installed yet, lazy installs allowed."""
    import pm

    installed = MagicMock(return_value=None)
    ensure = MagicMock()
    monkeypatch.setattr("tools.tirith_security.shutil.which", lambda _name: None)
    monkeypatch.setattr(pm, "installed_package", installed)
    monkeypatch.setattr(pm, "ensure", ensure)
    monkeypatch.setattr(pm, "lazy_installs_allowed", lambda: True)
    return ensure, installed


class TestPmInstall:
    def test_default_path_installs_through_pm(self, pm_tirith):
        """The default bare 'tirith' is provisioned by PM on a cold scan."""
        ensure, installed = pm_tirith
        ensure.side_effect = lambda *_a, **_k: setattr(
            installed, "return_value", MagicMock(binary="/pm/tirith"))

        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"
        for thread in _tirith_mod._install_threads.values():
            thread.join(5)
        ensure.assert_called_once_with("tirith")
        assert _tirith_mod._resolve_tirith_path("tirith") == "/pm/tirith"

    def test_failed_install_is_not_retried(self, pm_tirith):
        """After a failed install, subsequent resolves fall back without retrying."""
        ensure, _ = pm_tirith
        ensure.side_effect = RuntimeError("download failed")

        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"
        for thread in _tirith_mod._install_threads.values():
            thread.join(5)
        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"
        assert ensure.call_count == 1

    def test_tilde_explicit_path_missing_no_download(self, pm_tirith):
        """An explicit ~/path that doesn't exist must NOT trigger an install."""
        ensure, _ = pm_tirith

        result = _tirith_mod._resolve_tirith_path("~/bin/tirith")

        ensure.assert_not_called()
        assert "~" not in result  # tilde still expanded

    def test_install_proceeds_without_cosign(self, tmp_path):
        """Provenance is optional without cosign: SHA-256 verification alone proceeds."""
        with patch("tools.tirith_security.shutil.which", return_value=None):
            verified, reason = _tirith_mod.verify_release_provenance(tmp_path, MagicMock())
        assert (verified, reason) == (False, "")


# ---------------------------------------------------------------------------
# Background install / non-blocking startup (P2)
# ---------------------------------------------------------------------------

class TestBackgroundInstall:
    def test_ensure_installed_non_blocking(self, pm_tirith):
        """ensure_installed must return immediately when an install is needed."""
        with patch("tools.tirith_security._load_security_config", return_value=_BARE_CFG), \
             patch("tools.tirith_security.is_platform_supported", return_value=True), \
             patch("tools.tirith_security.threading.Thread") as MockThread:
            assert ensure_installed() is None  # not available yet
            MockThread.assert_called_once()
            MockThread.return_value.start.assert_called_once()

    def test_scan_does_not_wait_on_startup_install(self, pm_tirith):
        """A scan during the startup install returns the default instead of installing again."""
        ensure, _ = pm_tirith
        with patch("tools.tirith_security._load_security_config", return_value=_BARE_CFG), \
             patch("tools.tirith_security.is_platform_supported", return_value=True), \
             patch("tools.tirith_security.threading.Thread"):
            ensure_installed()

        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"
        ensure.assert_not_called()


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

        assert result == {"action": "allow", "findings": [], "summary": "",
                          "scanner_state": "ran"}

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


# ---------------------------------------------------------------------------
# Candidate header probe — the resolution accepts only a binary this process can
# run (REQ-INS1-SAAS-066 Beh 1, 2, 3, 5)
#
# Nothing is built, downloaded or installed here: the acceptance probe reads the
# candidate's own header, so a header is the whole fixture.
# ---------------------------------------------------------------------------

_ELF_MACHINE_X86_64 = 62
_ELF_MACHINE_AARCH64 = 183
_MACHO_CPUTYPE_X86_64 = 0x01000007
_MACHO_CPUTYPE_AARCH64 = 0x0100000C
# The process identity the probe compares a header against, as PM names it — pinned rather than read
# from the host, because the measured case is a Mach-O inside aarch64/linux and every case below needs
# that identity on whatever machine the suite runs.
_TARGET_LINUX_AARCH64 = "aarch64-unknown-linux-gnu"


def _elf_bytes(machine: int, *, length: int = 64) -> bytes:
    """An ELF64 little-endian header declaring ``machine`` at ``e_machine`` (offset 18)."""
    head = bytearray(b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8) + b"\x00\x00")
    head[18:20] = machine.to_bytes(2, "little")
    return bytes(head[:length]).ljust(length, b"\x00")


def _macho_bytes(cputype: int, *, length: int = 32) -> bytes:
    """A 64-bit little-endian Mach-O header — the measured foreign file's first octets, ``cf fa ed fe``
    — declaring ``cputype`` at offset 4."""
    head = bytearray(b"\xcf\xfa\xed\xfe")
    head += cputype.to_bytes(4, "little") + (2).to_bytes(4, "little") + (0).to_bytes(4, "little")
    head += bytes(16)
    return bytes(head[:length]).ljust(length, b"\x00")


def _native_header() -> bytes:
    """A header declaring the pinned process's own architecture, in the format that platform loads."""
    arch, platform_slot = _TARGET_LINUX_AARCH64.split("-", 1)
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


def _refusal_lines(caplog) -> list:
    return [rec.message for rec in caplog.records if "scanner_unrunnable" in rec.message]


def _pin_process(monkeypatch, target: str) -> None:
    """Pin the target PM names for this process — the identity the probe compares a header against.
    Only the identity is pinned: the header read stays real file I/O. `raising=False`: the resolution's
    identity source is this change's own seam, and a case run against the parent revision must fail on
    behaviour (the probe never refuses) rather than on a missing attribute."""
    monkeypatch.setattr(_tirith_mod, "_process_target", lambda: target, raising=False)


@pytest.fixture
def warned_fresh():
    """The once-per-class channel is process-wide: clear it so a case measures its own line."""
    _tirith_mod._warned_messages.clear()
    yield
    _tirith_mod._warned_messages.clear()


@pytest.fixture
def slots(tmp_path, monkeypatch):
    """Both local slots of the resolution empty — `PATH` and PM's own installed binary — on a pinned
    linux/aarch64 process and a fresh once-per-class channel. A case fills either one: `shutil.which`
    for the first slot, ``slots.installed.return_value = MagicMock(binary=…)`` for the second."""
    _tirith_mod._warned_messages.clear()
    _pin_process(monkeypatch, _TARGET_LINUX_AARCH64)
    monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: None)
    installed, ensure = MagicMock(return_value=None), MagicMock()
    monkeypatch.setattr(pm, "installed_package", installed)
    monkeypatch.setattr(pm, "ensure", ensure)
    monkeypatch.setattr(pm, "lazy_installs_allowed", lambda: True)
    yield SimpleNamespace(installed=installed, ensure=ensure)
    _tirith_mod._warned_messages.clear()


class TestCandidateHeaderProbe:
    """A candidate is a scanner only when the file it names is a binary this process can execute:
    the execute bit stays necessary and becomes insufficient on its own."""

    def test_macho_under_a_linux_process_is_refused(self, slots, tmp_path, caplog):
        """The measured case: a Mach-O inside aarch64/linux, handed back as the scanner."""
        candidate = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64))
        slots.installed.return_value = MagicMock(binary=candidate)

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert _tirith_mod._local_tirith("tirith") is None
            assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"

        slots.ensure.assert_called_once_with("tirith")  # what is reached instead is the install path
        (line,) = _refusal_lines(caplog)
        assert candidate in line
        assert "mach-o/aarch64" in line and _TARGET_LINUX_AARCH64 in line

    def test_an_elf_for_another_machine_is_refused(self, slots, tmp_path, caplog):
        candidate = _write_candidate(tmp_path, _elf_bytes(_ELF_MACHINE_X86_64))
        slots.installed.return_value = MagicMock(binary=candidate)

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert _tirith_mod._local_tirith("tirith") is None

        (line,) = _refusal_lines(caplog)
        assert candidate in line and "elf/x86_64" in line

    @pytest.mark.parametrize("payload", [
        _elf_bytes(_ELF_MACHINE_AARCH64, length=8),       # an ELF cut before e_machine
        _macho_bytes(_MACHO_CPUTYPE_AARCH64, length=8),   # a Mach-O cut before cputype
    ])
    def test_a_file_shorter_than_its_own_header_is_refused(self, slots, tmp_path, caplog, payload):
        candidate = _write_candidate(tmp_path, payload)
        slots.installed.return_value = MagicMock(binary=candidate)

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert _tirith_mod._local_tirith("tirith") is None

        (line,) = _refusal_lines(caplog)
        assert candidate in line and "truncated" in line

    def test_a_binary_of_this_architecture_is_accepted(self, slots, tmp_path):
        """Positive control — the gate can pass: the resolution returns that path and the install path
        is not entered."""
        candidate = _write_candidate(tmp_path, _native_header())
        slots.installed.return_value = MagicMock(binary=candidate)

        assert _tirith_mod._local_tirith("tirith") == candidate
        assert _tirith_mod._resolve_tirith_path("tirith") == candidate
        slots.ensure.assert_not_called()

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security.is_platform_supported", return_value=True)
    @patch("tools.tirith_security._load_security_config")
    def test_a_binary_of_this_architecture_scans_and_says_ran(self, mock_cfg, _supported, mock_run,
                                                              slots, tmp_path):
        """The gate is a gate, not a ban: a valid build at a slot is scanned and returns its verdict."""
        candidate = _write_candidate(tmp_path, _native_header())
        slots.installed.return_value = MagicMock(binary=candidate)
        mock_cfg.return_value = _CFG
        mock_run.return_value = _mock_run(0, _json_stdout(summary="clean"))

        result = check_command_security("echo hi")

        assert result == {"action": "allow", "findings": [], "summary": "clean",
                          "scanner_state": "ran"}
        assert mock_run.call_args[0][0][0] == candidate

    def test_a_candidate_with_no_binary_header_keeps_the_existing_rule(self, slots, tmp_path, caplog):
        """A file the probe has no format to read (missing, unreadable, not a binary) keeps the module's
        existing state: the new refusal class is not invented over it (Beh 5)."""
        candidate = _write_candidate(tmp_path, b"#!/bin/sh\nexit 0\n")
        slots.installed.return_value = MagicMock(binary=candidate)

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert _tirith_mod._local_tirith("tirith") == candidate

        assert _refusal_lines(caplog) == []

    def test_a_big_endian_macho_header_is_refused_too(self, slots, tmp_path, caplog):
        """The magic carries the file's byte order: `feedfacf` is read big-endian, not as this host's."""
        candidate = _write_candidate(
            tmp_path, b"\xfe\xed\xfa\xcf" + _MACHO_CPUTYPE_AARCH64.to_bytes(4, "big") + bytes(24))
        slots.installed.return_value = MagicMock(binary=candidate)

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert _tirith_mod._local_tirith("tirith") is None

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

    def test_a_foreign_explicit_path_is_refused_as_not_found(self, slots, tmp_path, caplog):
        """The authoritative explicit path is a candidate too: a binary this process cannot run is never
        the resolved scanner."""
        candidate = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64),
                                     name="custom-tirith")

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert _tirith_mod._local_tirith(candidate) is None
            assert _tirith_mod._resolve_tirith_path(candidate) == candidate

        slots.ensure.assert_not_called()  # an explicit path never downloads a replacement
        (line,) = _refusal_lines(caplog)
        assert candidate in line


class TestRefusedSlotFallsThrough:
    """A refused candidate is *not found*: the order the resolution already has is unchanged (Beh 2)."""

    def test_a_refused_path_hit_lets_the_pm_slot_answer(self, slots, tmp_path, monkeypatch):
        on_path = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="path-tirith")
        valid = _write_candidate(tmp_path, _native_header(), name="pm-tirith")
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: on_path)
        slots.installed.return_value = MagicMock(binary=valid)

        assert _tirith_mod._local_tirith("tirith") == valid

    def test_both_slots_refused_reach_the_install_path(self, slots, tmp_path, monkeypatch):
        on_path = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="path-tirith")
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: on_path)

        assert _tirith_mod._local_tirith("tirith") is None
        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"
        slots.ensure.assert_called_once_with("tirith")


class TestTheInstallResultIsProbedToo:
    """PM owns durable selection; what it hands back is a candidate like any other, so the acceptance
    rule is applied once more on the way out of the install path (Beh 1, Beh 2)."""

    def test_the_install_result_is_not_handed_back_when_it_is_refused(self, slots, tmp_path):
        refused = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="pm-tirith")
        slots.installed.return_value = MagicMock(binary=refused)

        assert _tirith_mod._resolve_tirith_path("tirith") == "tirith"

    def test_ensure_installed_does_not_return_a_refused_binary(self, slots, tmp_path):
        refused = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="pm-tirith")
        slots.installed.return_value = MagicMock(binary=refused)

        with patch("tools.tirith_security._load_security_config", return_value=_CFG), \
             patch("tools.tirith_security.is_platform_supported", return_value=True), \
             patch("tools.tirith_security.threading.Thread"):
            assert ensure_installed() is None


class TestRefusalIsTypedOnTheWayIn:
    """The refusal is produced where the resolution refuses, before any command is guarded (Beh 3)."""

    def test_the_boot_call_with_log_failures_false_still_names_the_refusal(self, slots, tmp_path, caplog):
        """`unsupported_platform` keeps its silence; this class may not be swallowed by the boot call."""
        candidate = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="pm-tirith")
        slots.installed.return_value = MagicMock(binary=candidate)

        with patch("tools.tirith_security._load_security_config", return_value=_CFG), \
             patch("tools.tirith_security.is_platform_supported", return_value=True), \
             patch("tools.tirith_security.threading.Thread") as MockThread, \
             caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert ensure_installed(log_failures=False) is None

        (line,) = _refusal_lines(caplog)
        assert candidate in line
        assert "mach-o/aarch64" in line and _TARGET_LINUX_AARCH64 in line
        MockThread.assert_called_once()  # the install path is reached instead of the refused file

    def test_the_refusal_is_named_once_per_refused_path(self, slots, tmp_path, caplog):
        """A slot a mount keeps refilling with the same foreign file does not repeat the line."""
        candidate = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="pm-tirith")
        slots.installed.return_value = MagicMock(binary=candidate)

        with caplog.at_level("WARNING", logger="tools.tirith_security"):
            for _ in range(5):
                _tirith_mod._local_tirith("tirith")

        assert len(_refusal_lines(caplog)) == 1

    def test_a_platform_with_no_build_and_no_candidate_stays_silent(self, slots, caplog):
        """Where there is nothing to refuse, the boot call stays as silent as it was (Beh 3)."""
        with patch("tools.tirith_security._load_security_config", return_value=_CFG), \
             patch("tools.tirith_security.is_platform_supported", return_value=False), \
             patch("tools.tirith_security.threading.Thread") as MockThread, \
             caplog.at_level("WARNING", logger="tools.tirith_security"):
            assert ensure_installed(log_failures=False) is None

        assert _refusal_lines(caplog) == []
        MockThread.assert_not_called()


# ---------------------------------------------------------------------------
# The guard never fails open in silence (REQ-INS1-SAAS-066 Beh 4)
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
    """A state observable only by reading a `summary` string is a failure of Beh 4."""

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

    @patch("tools.tirith_security.subprocess.run")
    @patch("tools.tirith_security._load_security_config")
    def test_entry_into_and_exit_from_disabled_are_named(self, mock_cfg, mock_run, caplog,
                                                         warned_fresh):
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

    def test_the_refusal_and_the_state_ride_the_same_turn_log(self, slots, tmp_path, caplog,
                                                             monkeypatch):
        """Nobody needs an `Exec format error` to learn either: a refused candidate is named on the way
        in, and the verdict that follows says no scan ran."""
        on_path = _write_candidate(tmp_path, _macho_bytes(_MACHO_CPUTYPE_AARCH64), name="path-tirith")
        monkeypatch.setattr(_tirith_mod.shutil, "which", lambda _name: on_path)

        with patch("tools.tirith_security._load_security_config", return_value=_CFG), \
             patch("tools.tirith_security.is_platform_supported", return_value=True), \
             patch("tools.tirith_security.subprocess.run") as mock_run, \
             caplog.at_level("WARNING", logger="tools.tirith_security"):
            mock_run.side_effect = OSError(8, "Exec format error")
            result = check_command_security("ls -la /tmp")

        assert result["scanner_state"] == "unavailable"
        text = "\n".join(rec.message for rec in caplog.records)
        assert "scanner_unrunnable" in text and on_path in text
        assert "scanner_state=unavailable" in text
